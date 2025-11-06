import cv2
from src.video.video_sfm import VideoSFM

VIDEO = "../data/input.mp4"


def test_process_video_frames_smoke():
    cap = cv2.VideoCapture(VIDEO)
    assert cap.isOpened()
    frames = []
    # sample ~120 frames to keep runtime low
    for _ in range(120):
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    assert len(frames) >= 30

    sfm = VideoSFM(device="cpu")
    out = sfm.process_video_frames(frames, VIDEO, stride=4)

    # shape and health checks
    assert isinstance(out, dict)
    for k in ("poses", "intrinsics", "points_3d", "colors", "frame_indices"):
        assert k in out

    poses = out["poses"]
    pts3d = out["points_3d"]
    K = out["intrinsics"]

    assert poses.ndim == 3 and poses.shape[1:] == (4, 4)
    assert K.shape == (3, 3)
    assert pts3d.ndim == 2 and pts3d.shape[1] == 3
    # non-trivial result
    assert len(poses) >= 10
    assert pts3d.shape[0] >= 500
