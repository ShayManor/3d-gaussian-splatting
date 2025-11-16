from logging import log, WARNING, INFO, DEBUG

import cv2
import numpy as np
from lightglue import SuperPoint, LightGlue
from tqdm import tqdm

from src.video.calibrate import Calibrator


class VideoSFM:
    """
    Class representing one video and getting the structure from frames
    """

    def __init__(self, device="cuda", matcher="opencv"):
        self.extractor = SuperPoint(max_num_keypoints=None).eval().to(device)  # More robust to changes
        self.matcher = LightGlue(features="superpoint", depth_confidence=-1, width_confidence=-1).eval().to(device)
        self.calibrator = Calibrator(matcher)


    def process_video_frames(self, frames, video_path, stride=5):
        """
        Extract poses from series of video frames
        :param frames: Series of frames
        :param video_path: path to video
        :param stride: Number of bytes per row in image
        :return: Poses (representation of position and orientation compared to frame)
        """
        poses = [np.identity(4)]
        K = self.calibrator.identify_intrinsics(
            frames[: min(50, len(frames))], video_path
        )

        points_3d = []
        point_colors = []

        frame_indices = list(range(0, len(frames), stride))

        if not frame_indices:
            return {
                "poses": np.empty((0, 4, 4)),
                "intrinsics": self.calibrator.identify_intrinsics([], video_path),
                "points_3d": np.empty((0, 3)),
                "colors": np.empty((0, 3)),
                "frame_indices": np.empty((0,), int),
            }


        prev_frame_idx = 0
        pose_frame_indices = [frame_indices[0]]
        n_dupes = 0

        for i, frame_idx in enumerate(
            tqdm(frame_indices[1:], total=len(frame_indices) - 1, desc="Processing pairs"), 1
        ):
            if self._too_similar(frames[prev_frame_idx], frames[frame_idx]):
                # prev_frame_idx = frame_idx
                n_dupes += 1
                continue
            matches = self.calibrator.extract_all_matches(
                [frames[prev_frame_idx], frames[frame_idx]]
            )

            if not matches or len(matches[0]["pts1"]) < 30:
                poses.append(poses[-1])
                log(
                    WARNING, f"Not enough matches ({len(matches[0]['pts1'])}) in frame!"
                )
                # prev_frame_idx = frame_idx
                n_dupes += 1
                continue

            pts1, pts2 = matches[0]["pts1"], matches[0]["pts2"]

            R, t, inliers = self.estimate_pose_from_matches(pts1, pts2, K)
            if R is None:
                # prev_frame_idx = frame_idx
                n_dupes += 1
                continue

            # Quick parralax check and drop small candidates
            ninl = int(np.count_nonzero(inliers)) if inliers is not None else 0
            flow_med = float(np.median(np.linalg.norm(pts2[inliers] - pts1[inliers], axis=1))) if ninl else 0.0

            # TODO: Make this configurable
            MIN_INLIERS = 100
            MIN_FLOW_PX = 4.0
            if ninl < MIN_INLIERS or flow_med < MIN_FLOW_PX:
                # increase baseline and try the next candidate, do not triangulate
                # prev_frame_idx = frame_idx  # advance anchor
                n_dupes += 1
                continue

            pts1 = pts1[inliers]
            pts2 = pts2[inliers]

            # if len(pts1) < 30:
            #     poses.append(poses[-1])
            #     continue

            # Builds absolute pose
            pose = np.eye(4)
            pose[:3, :3] = R
            pose[:3, 3] = t.squeeze()
            absolute_pose = pose @ poses[-1]  # world to camera
            poses.append(absolute_pose)
            pose_frame_indices.append(frame_idx)

            # Triangulate new points
            if len(poses) > 1:
                new_points = self.triangulate_points(
                    pts1, pts2, K, poses[-2], poses[-1]
                )
                if new_points is not None and len(new_points):
                    points_3d.extend(new_points)
                    # Extract colors from frame
                    colors = self._extract_point_colors(frames[frame_idx], pts2)
                    point_colors.extend(colors)
                else:
                    log(WARNING, f"Triangulation failed at frame {i}")

            prev_frame_idx = frame_idx

        log(INFO, f"SFM complete: {len(poses)} poses, {len(points_3d)} 3D points")
        if n_dupes > 0.3 * (len(frame_indices) - 1):
            log(WARNING, f"Number of dupes (skip failures): {n_dupes}")
        else:
            log(INFO, f"Number of dupes (skip failures): {n_dupes}")

        poses = np.array(poses)

        log(DEBUG, "SFM poses[0]:\n", poses[0])
        if len(poses) > 1:
            log(DEBUG, "SFM poses[1]:\n", poses[1])
        log(DEBUG, "SFM #3D points:", len(points_3d))

        return {
            "poses": np.array(poses),
            "intrinsics": K,
            "points_3d": np.array(points_3d) if points_3d else np.empty((0, 3)),
            "colors": np.array(point_colors) if point_colors else np.empty((0, 3)),
            "frame_indices": np.array(pose_frame_indices, dtype=int),
        }

    def estimate_pose_from_matches(self, pts1, pts2, K):
        """
        Estimate the relative pose from the points
        :return: Essential (3x3) matrix, mask for inline/outline
        """
        E, mask = cv2.findEssentialMat(
            pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0
        )
        if E is None or mask is None:
            return None, None, None

        mask = mask.ravel().astype(bool)
        if mask.sum() < 8:
            return None, None, None

        # homography-vs-essential degeneracy check on inliers
        H, mH = cv2.findHomography(
            pts1[mask], pts2[mask],
            cv2.RANSAC, 3.0, confidence=0.999
        )
        inl_F = int(mask.sum())
        inl_H = int(mH.ravel().sum()) if mH is not None else 0
        if inl_H >= 1.5 * inl_F:
            return None, None, None

        # Recover pose directly from E (essential matrix)
        ok, R, t, pose_mask = cv2.recoverPose(
            E, pts1, pts2, K, mask=mask.astype(np.uint8)
        )
        if not ok:
            return None, None, None

        inliers = pose_mask.ravel().astype(bool)
        return R, t, inliers

    def triangulate_points(self, pts1, pts2, K, pose1, pose2):
        """
        Triangulates 3D points from the given 2D points
        :param pts1: First 2d point
        :param pts2: Second 2d point
        :param K: Camera intrinsics
        :param pose1: Position and orientation with respect to camera
        :param pose2: Position and orientation with respect to camera
        :return: Triangulated 3D points
        """
        # Projection Matrices
        # P1 = K @ pose1[:3, :]
        # P2 = K @ pose2[:3, :]
        #
        # W2C1 = pose1
        # W2C2 = pose2
        # C2W1 = np.linalg.inv(W2C1)
        # T21 = W2C2 @ C2W1  # 4x4
        # R = T21[:3, :3]
        # t = T21[:3, 3:4]

        # Triangulate
        pts1_h = cv2.undistortPoints(pts1.reshape(-1, 1, 2), K, None).reshape(-1, 2)
        pts2_h = cv2.undistortPoints(pts2.reshape(-1, 1, 2), K, None).reshape(-1, 2)

        x1 = pts1_h.T.astype(np.float64)  # shape 2xN
        x2 = pts2_h.T.astype(np.float64)  # shape 2xN

        # Pose is world-to-camera
        # Tcw1 = np.linalg.inv(pose1)[:3, :4]  # 3x4
        # Tcw2 = np.linalg.inv(pose2)[:3, :4]  # 3x4

        # Relative pose from cam1 to cam2: T21 = Tcw2 * Twc1
        # Twc1 = np.vstack([np.linalg.inv(pose1)[:3, :4], [0, 0, 0, 1]])
        # Twc2 = np.vstack([np.linalg.inv(pose2)[:3, :4], [0, 0, 0, 1]])
        T21 = pose2 @ np.linalg.inv(pose1)  # W2C2 * C2W1
        R = T21[:3, :3]
        t = T21[:3, 3:4]  # column vector

        P0 = np.hstack([np.eye(3), np.zeros((3, 1))])  # normalized projection for cam1
        P1 = np.hstack([R, t])  # normalized projection for cam2

        # Triangulated in normal space
        Xh = cv2.triangulatePoints(P0, P1, x1, x2)  # 4xN
        X = (Xh[:3] / Xh[3]).T  # Nx3 in cam1 frame

        z1 = X[:, 2]  # Depth in cam1 coords

        # Transform X to cam2: X2 = R*X + t
        X2 = (R @ X.T + t).T
        z2 = X2[:, 2]

        cheirality_mask = (z1 > 0) & (z2 > 0)

        # Reproj in pixels
        Xh1 = np.hstack([X, np.ones((len(X), 1))])  # Nx4 in cam1
        # cam1: [I|0]
        proj1 = (K @ Xh1[:, :3].T).T
        proj1 = proj1[:, :2] / proj1[:, 2:3]

        # cam2: [R|t]
        Xh2 = (R @ X.T + t).T  # Nx3
        proj2 = (K @ Xh2.T).T
        proj2 = proj2[:, :2] / proj2[:, 2:3]

        mask = self._filter_triangulated_points(
            points_3d=X,  # in cam1 coords
            pts1=pts1,
            pts2=pts2,
            proj1=proj1,
            proj2=proj2,
            z1=z1,
            z2=z2,
        )

        if not mask.any():
            return None

        X_cam1 = X[mask]
        # pose1: world-to-camera => invert to get cam-to-world
        Twc1 = np.linalg.inv(pose1)  # 4x4
        X_h = np.hstack([X_cam1, np.ones((X_cam1.shape[0], 1))])  # Nx4
        X_world = (Twc1 @ X_h.T).T[:, :3]  # Nx3 in world coords

        return X_world

    def _too_similar(self, img1, img2,
                     corr_t=0.995, mad_t=0.6, flow_px=0.6, max_corners=400):
        """
        Function to check if frames are too similar and skip - written by AI because I can't be bothered
        TODO: Improve this
        """
        # downscale + grayscale
        g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        h, w = g1.shape
        s = 256.0 / max(h, w)
        if s < 1.0:
            g1 = cv2.resize(g1, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
            g2 = cv2.resize(g2, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)

        # global histogram correlation (very cheap)
        hist1 = cv2.calcHist([g1], [0], None, [32], [0, 256]);
        cv2.normalize(hist1, hist1)
        hist2 = cv2.calcHist([g2], [0], None, [32], [0, 256]);
        cv2.normalize(hist2, hist2)
        corr = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        if corr >= corr_t:
            return True  # near-identical appearance

        # mean absolute difference after blur (robust to noise)
        b1 = cv2.GaussianBlur(g1, (3, 3), 0)
        b2 = cv2.GaussianBlur(g2, (3, 3), 0)
        mad = float(np.mean(np.abs(b1.astype(np.float32) - b2.astype(np.float32))))
        if mad <= mad_t:
            return True  # photometrically too close

        # tiny KLT probe to measure motion
        p0 = cv2.goodFeaturesToTrack(g1, maxCorners=max_corners, qualityLevel=0.01, minDistance=7,
                                     useHarrisDetector=True, k=0.04)
        if p0 is None or len(p0) < 20:
            return True  # no texture, treat as unusable
        p1, st, _ = cv2.calcOpticalFlowPyrLK(g1, g2, p0, None, winSize=(21, 21), maxLevel=2,
                                             criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 1e-3))
        if p1 is None or st is None:
            return True
        st = st.reshape(-1).astype(bool)
        flow = np.linalg.norm(p1[st] - p0[st], axis=2).reshape(-1)
        return float(np.median(flow)) < flow_px

    def _filter_triangulated_points(
        self,
        points_3d,
        pts1,
        pts2,
        proj1,
        proj2,
        z1,
        z2,
        max_reproj_error=6.0,
    ):
        """
        Filters the 3D points based on reprojection error and depth
        :param points_3d: 3D points in cam1
        :param pts1: First 2D point Nx2 pixels
        :param pts2: Second 2D point Nx2 pixels
        :param proj1: Projection 1 Nx2 pixels predicted
        :param proj2: Projection 2 Nx2 pixels predicted
        :param z1: N depths in cam1
        :param z2: N depths in cam2
        :param max_reproj_error: Threshold for reproj error - Needs to be configurable
        :return: mask to filter points
        """
        # cheirality - no magnitude check
        cheirality = (z1 > 0) & (z2 > 0)

        # Pixel reprojection errors
        e1 = np.linalg.norm(proj1 - pts1, axis=1)
        e2 = np.linalg.norm(proj2 - pts2, axis=1)
        reproj_ok = (e1 < max_reproj_error) & (e2 < max_reproj_error)

        final_mask = cheirality & reproj_ok
        return final_mask


    def _extract_point_colors(self, frame, pts):
        """
        Extracts RGB colors at locations
        :param frame: Frame to get colors from
        :param pts: Points on the frame
        :return: Colors matrix
        """

        colors = []
        h, w = frame.shape[:2]
        for pt in pts:
            x, y = int(pt[0]), int(pt[1])
            if 0 <= x < w and 0 <= y < h:
                color = frame[y, x] / 255.0  # Normalize
                colors.append(color)
            else:
                log(WARNING, "Error in extracting colors")
                colors.append([0.5, 0.5, 0.5])  # Default gray
        return colors
