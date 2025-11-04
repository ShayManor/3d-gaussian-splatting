from logging import log, WARNING, INFO

import cv2
import numpy as np
from lightglue import SuperPoint, LightGlue

from src.video.calibrate import Calibrator


class VideoSFM:
    """
    Class representing one video and getting the structure from frames
    """

    def __init__(self, device="cuda"):
        self.extractor = SuperPoint().eval().to(device)  # More robust to changes
        self.matcher = LightGlue(features="superpoint").eval().to(device)
        self.calibrator = Calibrator()

    def process_video_frames(self, frames, video_path, stride=10):
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
        point_map_ids = {}

        frame_indices = list(range(0, len(frames), stride))

        prev_feats = None
        prev_frame_idx = 0

        for i, frame_idx in enumerate(frame_indices[1:], 1):
            matches = self.calibrator.extract_all_matches(
                [frames[prev_frame_idx], frames[frame_idx]]
            )

            if not matches or len(matches[0]["pts1"]) < 30:
                poses.append(poses[-1])
                log(
                    WARNING, f"Not enough matches ({len(matches[0]['pts1'])}) in frame!"
                )
                continue

            pts1, pts2 = matches[0]["pts1"], matches[0]["pts2"]

            R, t, inliers = self.estimate_pose_from_matches(pts1, pts2, K)
            if R is None:
                poses.append(poses[-1])
                continue

            pts1 = pts1[inliers]  # No clue what this does
            pts2 = pts2[inliers]

            if len(pts1) < 30:
                poses.append(poses[-1])
                continue

            # Builds absolute pose
            pose = np.eye(4)
            pose[:3, :3] = R
            pose[:3, 3] = t.squeeze()
            absolute_pose = pose @ poses[-1]  # world to camera
            poses.append(absolute_pose)

            # Triangulate new points
            if len(poses) > 1:
                new_points = self.triangulate_points(
                    pts1, pts2, K, poses[-2], poses[-1]
                )
                if new_points is not None:
                    points_3d.extend(new_points)
                    # Extract colors from frame
                    colors = self._extract_point_colors(frames[frame_idx], pts2)
                    point_colors.extend(colors)
                else:
                    log(WARNING, f"Triangulation failed at frame {i}")

            prev_frame_idx = frame_idx

        log(INFO, f"SFM complete: {len(poses)} poses, {len(points_3d)} 3D points")
        return {
            "poses": np.array(poses),
            "intrinsics": K,
            "points_3d": np.array(points_3d) if points_3d else np.empty((0, 3)),
            "colors": np.array(point_colors) if point_colors else np.empty((0, 3)),
            "frame_indices": frame_indices,
        }

    def estimate_pose_from_matches(self, pts1, pts2, K):
        """
        Estimate the relative pose from the points
        :return: Essential (3x3) matrix, mask for inline/outline
        """
        E, mask = cv2.findEssentialMat(
            pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0
        )
        if E is None:
            return None, None, None

        _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, K, mask=mask)
        inliners = (mask_pose.ravel() == 1)
        return R, t, inliners

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
        Twc1 = np.vstack([np.linalg.inv(pose1)[:3, :4], [0, 0, 0, 1]])
        Twc2 = np.vstack([np.linalg.inv(pose2)[:3, :4], [0, 0, 0, 1]])
        T21 = np.linalg.inv(Twc1) @ Twc2
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

        return X[mask] if mask.any() else None

        # points_4d = cv2.triangulatePoints(P1, P2, pts1_h, pts2_h)  # Quaternions
        # points_3d = points_4d[:3] / points_4d[3]
        # log(INFO, f"Before filtering: {points_3d.shape[1]} points")
        # Filter outliers based on reprojection error and depth
        # mask = self._filter_triangulated_points(points_3d.T, pts1, pts2, P1, P2)
        # log(
        #     INFO,
        #     f"After filtering: {mask.sum()} points (kept {mask.sum()}/{len(mask)})",
        # )
        # return points_3d.T[mask] if mask.any() else None

    def _filter_triangulated_points(
        self,
        points_3d,
        pts1,
        pts2,
        proj1,
        proj2,
        z1,
        z2,
        max_reproj_error=2.5,  # Unsure of this value
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

        # Logs
        if len(z1) > 0:
            z1min, z1max = float(np.min(z1)), float(np.max(z1))
        else:
            z1min = z1max = 0.0
        log(INFO,
            f"  Depth cheirality: {cheirality.sum()}/{len(cheirality)} points have z>0 in both views (z1 range: [{z1min:.2f}, {z1max:.2f}])")

        if len(e1) > 0:
            emin, emax = float(np.min(np.minimum(e1, e2))), float(np.max(np.maximum(e1, e2)))
        else:
            emin = emax = 0.0
        log(INFO, f"  Reproj filter: {reproj_ok.sum()}/{len(reproj_ok)} pass (errors range: [{emin:.2f}, {emax:.2f}])")

        final_mask = cheirality & reproj_ok
        log(INFO, f"  Final: {final_mask.sum()}/{len(final_mask)} points pass both filters")
        return final_mask

        # # Check depths
        # depths1 = (
        #     P1[2:3] @ np.hstack([points_3d, np.ones((len(points_3d), 1))]).T
        # ).flatten()
        # depths2 = (
        #     P2[2:3] @ np.hstack([points_3d, np.ones((len(points_3d), 1))]).T
        # ).flatten()
        #
        # depth_mask = (
        #     (depths1 > min_depth)
        #     & (depths1 < max_depth)
        #     & (depths2 > min_depth)
        #     & (depths2 < max_depth)
        # )
        #
        # log(
        #     INFO,
        #     f"  Depth filter: {depth_mask.sum()}/{len(depth_mask)} points pass (depths range: [{depths1.min():.2f}, {depths1.max():.2f}])",
        # )
        #
        # # Reprojection errors
        # points_h = np.hstack([points_3d, np.ones((len(points_3d), 1))])
        #
        # proj1 = (P1 @ points_h.T).T
        # proj1 = proj1[:, :2] / proj1[:, 2:3]
        #
        # proj2 = (P2 @ points_h.T).T
        # proj2 = proj2[:, :2] / proj2[:, 2:3]
        #
        # errors1 = np.linalg.norm(proj1 - pts1, axis=1)
        # errors2 = np.linalg.norm(proj2 - pts2, axis=1)
        #
        # error_mask = (errors1 < max_reproj_error) & (errors2 < max_reproj_error)
        # log(
        #     INFO,
        #     f"  Reproj filter: {error_mask.sum()}/{len(error_mask)} points pass (errors range: [{errors1.min():.2f}, {errors1.max():.2f}])",
        # )
        # final_mask = depth_mask & error_mask
        # log(
        #     INFO,
        #     f"  Final: {final_mask.sum()}/{len(final_mask)} points pass both filters",
        # )
        #
        # return final_mask

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
