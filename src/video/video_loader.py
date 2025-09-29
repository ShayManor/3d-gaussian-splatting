import cv2


class VideoLoader:
    """
    Efficiently loads video frames
    """

    def __init__(self, video_path, cache_frames=False):
        self.video_path = video_path
        self.cache_frames = cache_frames
        self.capture = cv2.VideoCapture(str(video_path))
        self.total_frames = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.capture.get(cv2.CAP_PROP_FPS)
        self.frame_cache = {} if cache_frames else None

    def get_frame(self, idx):
        """
        Gets the frame efficiently
        :param idx: index of frame
        :return: frame
        """
        if self.frame_cache and idx in self.frame_cache:
            return self.frame_cache[idx]

        self.capture.set(cv2.CAP_PROP_POS_FRAMES, idx)
        data, frame = self.capture.read()

        if data and self.frame_cache:
            self.frame_cache[idx] = frame

        return frame if data else None

    def get_frame_generator(self, indices):
        """
        Generator for memory-efficient frame loading
        :param indices: indices for loading
        :return: Generator
        """
        for idx in indices:
            frame = self.get_frame(idx)
            if frame:
                yield frame

    def __del__(self):
        """
        Efficient deletion for this object
        """
        if hasattr(self, 'capture'):
            self.capture.release()
