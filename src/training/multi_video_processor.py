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

    def __init__(self, cache="./cache", device="cuda", max_frames_per_video=100_000, matcher="sift"):
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
            video_data = self._maybe_load_cache(video_cache) if use_cache else None
            if video_data is None:
                video_data = self._process_single_video(video_path, stride, self.matcher)
                with open(video_cache, "wb") as f:
                    pickle.dump(video_data, f)

            all_video_data.append(video_data)

        merged_data = self._merge_videos(all_video_data)
        return merged_data

    def _maybe_load_cache(self, video_cache: Path):
        """
        Load and validate a per-video SfM cache. Returns the dict if it has
        plausible content (>=2 poses and >0 3D points), else None so the caller
        can fall back to re-running SfM. Caches written by failed runs are
        common; treating them as authoritative leads to training from random
        gaussians on a bad scene_extent.
        """
        if not video_cache.exists():
            return None
        try:
            with open(video_cache, "rb") as f:
                cached = pickle.load(f)
        except Exception as e:
            log(WARNING, f"Failed to read cache {video_cache} ({e}); re-running SfM")
            return None
        n_poses = len(cached.get("poses", []))
        n_pts = len(cached.get("points_3d", []))
        if n_poses >= 2 and n_pts > 0:
            log(INFO, f"Using cached SFM from {video_cache} ({n_poses} poses, {n_pts} pts)")
            return cached
        log(
            WARNING,
            f"Discarding degenerate cache {video_cache} "
            f"({n_poses} poses, {n_pts} pts) and re-running SfM",
        )
        return None

    def _process_single_video(self, video_path: str, stride: int, matcher) -> Dict:
        """
        Processes a single video with SFM
        :param video_path: Path to individual video
        :param stride: Stride (sampling amount)
        :return: Processes SFM video
        """

        ### Opencv2 video capture ###
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

        n_poses = len(sfm_data["poses"])
        n_pts = len(sfm_data["points_3d"])
        log(INFO, f"SfM result: {n_poses} poses, {n_pts} 3D points")
        if n_poses < 2 or n_pts == 0:
            log(
                WARNING,
                "SfM produced too few poses/points — every candidate pair was rejected. "
                "Check the per-gate breakdown logged by VideoSFM above. Common causes: "
                "video too still (lower MIN_FLOW_PX or use larger stride), too low resolution, "
                "or weak texture (try matcher='loftr' if a GPU is available).",
            )

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

        if len(video_data_list) == 1:
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

