from logging import log, WARNING

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

            R, t = self.estimate_pose_from_matches(pts1, pts2, K)
            if R is None:
                poses.append(poses[-1])
                continue

            # Builds absolute pose
            pose = np.eye(4)
            pose[:3, :3] = R
            pose[:3, 3] = t.squeeze()
            absolute_pose = poses[-1] @ pose
            poses.append(absolute_pose)

            # Triangulate new points (only every 5 frames for efficiency)
            if i % 5 == 0 and len(poses) > 1:
                new_points = self.triangulate_points(
                    pts1, pts2, K, poses[-2], poses[-1]
                )
                if new_points is not None:
                    points_3d.extend(new_points)
                    # Extract colors from frame
                    colors = self._extract_point_colors(frames[frame_idx], pts2)
                    point_colors.extend(colors)

            prev_frame_idx = frame_idx

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
            return None, None

        _, R, t, _ = cv2.recoverPose(E, pts1, pts2, K, mask=mask)
        return R, t

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
        P1 = K @ pose1[:3, :]
        P2 = K @ pose2[:3, :]

        # Triangulate
        pts1_h = cv2.undistortPoints(pts1.reshape(-1, 1, 2), K, None)
        pts2_h = cv2.undistortPoints(pts2.reshape(-1, 1, 2), K, None)

        points_4d = cv2.triangulatePoints(P1, P2, pts1_h, pts2_h)  # Quaternions
        points_3d = points_4d[:3] / points_4d[3]

        # Filter outliers based on reprojection error and depth
        mask = self._filter_triangulated_points(points_3d.T, pts1, pts2, P1, P2)
        return points_3d.T[mask] if mask.any() else None

    def _filter_triangulated_points(
        self,
        points_3d,
        pts1,
        pts2,
        P1,
        P2,
        max_reproj_error=5.0,  # Unsure of this value
        min_depth=0.1,  # Unsure of this value
        max_depth=100.0,  # Unsure of this value
    ):
        """
        Filters the 3D points based on reprojection error and depth
        :param points_3d: 3D points
        :param pts1: First 2D point
        :param pts2: Second 2D point
        :param P1: Projection 1
        :param P2: Projection 2
        :param max_reproj_error: Threshold for reproj error - Needs to be configurable
        :param min_depth: Minimum threshold for depth - Needs to be configurable
        :param max_depth: Max threshold for depth - Needs to be configurable
        :return:
        """
        # Check depths
        depths1 = (
            P1[2:3] @ np.hstack([points_3d, np.ones((len(points_3d), 1))]).T
        ).flatten()
        depths2 = (
            P2[2:3] @ np.hstack([points_3d, np.ones((len(points_3d), 1))]).T
        ).flatten()

        depth_mask = (
            (depths1 > min_depth)
            & (depths1 < max_depth)
            & (depths2 > min_depth)
            & (depths2 < max_depth)
        )

        # Reprojection errors
        points_h = np.hstack([points_3d, np.ones((len(points_3d), 1))])

        proj1 = (P1 @ points_h.T).T
        proj1 = proj1[:, :2] / proj1[:, 2:3]

        proj2 = (P2 @ points_h.T).T
        proj2 = proj2[:, :2] / proj2[:, 2:3]

        errors1 = np.linalg.norm(proj1 - pts1, axis=1)
        errors2 = np.linalg.norm(proj2 - pts2, axis=1)

        error_mask = (errors1 < max_reproj_error) & (errors2 < max_reproj_error)
        return depth_mask & error_mask

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
