import pickle
from logging import INFO, WARNING, log
from pathlib import Path
from typing import List, Dict

from sklearn.neighbors import KDTree

import numpy as np
from tqdm import tqdm

from src.RANSAC import _align_video_to_reference
from src.video.video_loader import VideoLoader
from src.video.video_sfm import VideoSFM


class MultiVideoProcessor:
    """
    Processes multiple videos in order and turns them into one coordinate system.
    """

    def __init__(self, cache="./cache", device="cuda", max_frames_per_video=100_000, matcher="opencv"):
        self.cache_dir = Path(cache)
        self.use_cache = True
        self.matcher = matcher
        if self.cache_dir is None:
            self.use_cache = False
        else:
            self.cache_dir.mkdir(exist_ok=True)
        self.max_frames_per_video = max_frames_per_video
        self.device = device

    def process_videos(
        self, video_paths: List[str], stride: int = 10, use_cache: bool = True
    ) -> Dict:
        """
        Processes multiple videos into one coordinate system
        :param video_paths: Ordered list of video paths to process
        :param stride: Sampling for videos. TODO: Make this configurable per video
        :param use_cache: Whether or not to use cache for efficiency with the tradeoff of device storage
        :return: Merged videos in coordinate system
        """
        all_video_data = []

        for video_path in tqdm(video_paths, desc="Processing videos"):
            video_cache = self.cache_dir / f"{Path(video_path).stem}_sfm.pkl"
            if use_cache and video_cache.exists():
                log(INFO, f"Using cached SFM from {self.cache_dir}")
                with open(video_cache, "rb") as f:
                    video_data = pickle.load(f)
            else:
                video_data = self._process_single_video(video_path, stride, self.matcher)
                with open(video_cache, "wb") as f:
                    pickle.dump(video_data, f)

            all_video_data.append(video_data)

        merged_data = self._merge_videos(all_video_data)
        return merged_data

    def _process_single_video(self, video_path: str, stride: int, matcher) -> Dict:
        """
        Processes a single video with SFM
        :param video_path: Path to individual video
        :param stride: Stride (sampling amount)
        :return: Processes SFM video
        """
        loader = VideoLoader(video_path, self.use_cache)
        processor = VideoSFM(self.device, matcher=matcher)

        num_frames = min(loader.total_frames, self.max_frames_per_video)
        frame_indices = list(range(0, num_frames, stride))

        log(INFO, f"Loading {num_frames} frames from {video_path}")
        frames = []
        for idx in tqdm(frame_indices, desc="Loading frames", leave=False):
            frame = loader.get_frame(idx)
            if frame is not None:
                frames.append(frame)

        # Run SFM
        sfm_data = processor.process_video_frames(frames, video_path, stride=1)
        sfm_data["video_path"] = video_path
        sfm_data["fps"] = loader.fps
        local_pose_ids = sfm_data["frame_indices"]

        sfm_data["frame_indices"] = np.array(frame_indices, dtype=int)[local_pose_ids]

        print("RAW SFM pose[0]:\n", sfm_data["poses"][0])
        print("RAW SFM pose[1]:\n", sfm_data["poses"][1])

        return sfm_data

    def _merge_videos(self, video_data_list: List[Dict]) -> Dict:
        """
        Merges multiple videos using feature matching
        :param video_data_list:
        :return: Merged video features
        """
        merged = {
            "points_3d": video_data_list[0]["points_3d"],
            "colors": video_data_list[0]["colors"],
            "all_poses": [video_data_list[0]["poses"]],
            "all_intrinsics": [video_data_list[0]["intrinsics"]],
            "frame_indices": [video_data_list[0]["frame_indices"]],
            "video_info": [
                {
                    "path": video_data_list[0]["video_path"],
                    "fps": video_data_list[0]["fps"],
                    "num_frames": len(video_data_list[0]["poses"]),
                }
            ],
        }

        if (video_data_list[0] == 1):
            return merged


        # Align and merge videos
        for i in range(1, len(video_data_list)):
            aligned_data = _align_video_to_reference(
                video_data_list[i], merged
            )

            # Merge points
            merged['points_3d'] = np.vstack([
                merged['points_3d'],
                aligned_data['points_3d']
            ])
            merged['colors'] = np.vstack([
                merged['colors'],
                aligned_data['colors']
            ])
            merged['all_poses'].append(aligned_data['poses'])
            merged['all_intrinsics'].append(video_data_list[i]['intrinsics'])
            merged['video_info'].append({
                'path': video_data_list[i]['video_path'],
                'fps': video_data_list[i]['fps'],
                'num_frames': len(aligned_data['poses'])
            })

        # Dedupe
        merged = self._remove_duplicate_points(merged)

        return merged

    def _remove_duplicate_points(
        self, merged_data: Dict, threshold: float = 0.01
    ) -> Dict:
        """
        Removes duplicate points that are very close in space
        :param merged_data: Data from two videos
        :param threshold: Minimum allowed similarity value
        :return: deduped merged data
        """
        points = merged_data['points_3d']
        colors = merged_data['colors']

        if len(points) == 0:
            log(WARNING, "No points found!")
            return merged_data

        tree = KDTree(points)

        # Find points that are too close
        keep_mask = np.ones(len(points), dtype=bool)
        for i in range(len(points)):
            if keep_mask[i]:
                # Find neighbors within threshold
                indices = tree.query_radius([points[i]], r=threshold)[0]
                # Keep and mark
                for i in indices[1:]:
                    keep_mask[i] = False

        merged_data['points_3d'] = points[keep_mask]
        merged_data['colors'] = colors[keep_mask]

        log(INFO, f"Removed {(~keep_mask).sum()} duplicate points")

        return merged_data

