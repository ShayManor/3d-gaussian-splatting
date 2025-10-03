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
        K = self.calibrator.identify_intrinsics(frames[:min(50, len(frames))], video_path)

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

            if not matches or len(matches[0]['pts1']) < 30:
                poses.append(poses[-1])
                log(WARNING, f"Not enough matches ({len(matches[0]['pts1'])}) in frame!")
                continue

            pts1, pts2 = matches[0]['pts1'], matches[0]['pts2']

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
                'poses': np.array(poses),
                'intrinsics': K,
                'points_3d': np.array(points_3d) if points_3d else np.empty((0, 3)),
                'colors': np.array(point_colors) if point_colors else np.empty((0, 3)),
                'frame_indices': frame_indices
            }


    def estimate_pose_from_matches(self, pts1, pts2, K):
        """
        Estimate the relative pose from the points
        :return: Essential (3x3) matrix, mask for inline/outline
        """
        E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        if E is None:
            return None, None

        _, R, t, _ = cv2.recoverPose(E, pts1, pts2, K, mask=mask)
        return R, t
