import numpy as np
from lightglue import SuperPoint, LightGlue

from src.calibrate import Calibrator


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
        :return: Poses
        """
        poses = [np.identity(4)]
        K = self.calibrator.identify_intrinsics(frames, video_path)
