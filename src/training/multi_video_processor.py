import pickle
from logging import INFO, log
from pathlib import Path
from typing import List, Dict

from tqdm import tqdm


class MultiVideoProcessor:
    """
    Processes multiple videos in order and turns them into one coordinate system.
    """
    def __init__(self, cache='./cache', device='cuda', max_frames_per_video=100_000):
        self.cache_dir = Path(cache)
        self.cache_dir.mkdir(exist_ok=True)
        self.max_frames_per_video = max_frames_per_video
        self.device = device

    def process_videos(self, video_paths: List[str], stride: int = 10,
                       use_cache: bool = True) -> Dict:
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
            if use_cache and self.cache_dir.exists():
                log(INFO, f"Using cached SFM from {self.cache_dir}")
                with open(video_cache, 'rb') as f:
                    video_data = pickle.load(f)
            else:
                video_data = self._process_single_video(video_path, stride)
                with open(video_cache, 'wb') as f:
                    pickle.dump(video_data, f)

            all_video_data.append(video_data)

        merged_data = self._merge_videos(all_video_data)
        return merged_data

    def _process_single_video(self, video_path: str, stride: int) -> Dict:
        pass

    def _merge_videos(self, video_data_list: List[Dict]) -> Dict:
        pass

    def _remove_duplicate_points(self, merged_data: Dict, threshold: float = 0.01) -> Dict:
        pass

    def _align_video_to_reference(self, video_data: Dict, reference: Dict) -> Dict:
        pass
