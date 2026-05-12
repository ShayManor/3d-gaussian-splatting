import cv2


class VideoLoader:
    """
    Efficiently loads video frames. Pass `cache_frames=True` (or call
    `preload(indices)`) to keep decoded frames in CPU RAM — random-access reads
    on H.264 require re-decoding from the prior keyframe, which costs 30–80 ms
    per call. Sequential pre-decode is ~1 ms per frame.
    """

    def __init__(self, video_path, cache_frames=False):
        self.video_path = video_path
        self.capture = cv2.VideoCapture(str(video_path))
        self.total_frames = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.capture.get(cv2.CAP_PROP_FPS)
        # Empty dict means caching enabled but nothing decoded yet; None means
        # don't cache at all (every get_frame seeks). preload() lazily switches
        # a non-caching loader into a caching one.
        self.frame_cache = {} if cache_frames else None

    def preload(self, indices):
        """
        Sequentially decode the frames at `indices` into the cache. One linear
        pass through the file — no seeks — which is ~50× faster than calling
        get_frame() for each index on a random-access H.264 stream.
        """
        wanted = set(int(i) for i in indices)
        if not wanted:
            return
        if self.frame_cache is None:
            self.frame_cache = {}
        max_wanted = max(wanted)
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        for i in range(max_wanted + 1):
            ok, frame = self.capture.read()
            if not ok:
                break
            if i in wanted:
                self.frame_cache[i] = frame

    def get_frame(self, idx):
        if self.frame_cache is not None and idx in self.frame_cache:
            return self.frame_cache[idx]

        self.capture.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = self.capture.read()
        if not ok:
            raise RuntimeError("Can not read video")

        if self.frame_cache is not None:
            self.frame_cache[idx] = frame
        return frame

    def get_frame_generator(self, indices):
        for idx in indices:
            frame = self.get_frame(idx)
            if frame is not None:
                yield frame

    def __del__(self):
        if hasattr(self, "capture"):
            self.capture.release()
