import cv2
from src.video.video_sfm import VideoSFM

VIDEO = "../data/input.mp4"


def test_process_video_frames_smoke():
    cap = cv2.VideoCapture(VIDEO)
    assert cap.isOpened()
    frames = []
    # 60 fps source: sample 360 frames at stride=12 below = 200 ms baselines.
    # Smaller baselines starve SIFT of inliers; LoFTR was tolerant there but the
    # default matcher is now SIFT, which is what production uses.
    for _ in range(360):
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    assert len(frames) >= 60

    sfm = VideoSFM(device="cpu")
    out = sfm.process_video_frames(frames, VIDEO, stride=12)

    assert isinstance(out, dict)
    for k in ("poses", "intrinsics", "points_3d", "colors", "frame_indices"):
        assert k in out

    poses = out["poses"]
    pts3d = out["points_3d"]
    K = out["intrinsics"]

    assert poses.ndim == 3 and poses.shape[1:] == (4, 4)
    assert K.shape == (3, 3)
    assert pts3d.ndim == 2 and pts3d.shape[1] == 3
    assert len(poses) >= 5
    assert pts3d.shape[0] >= 200

    # Scale-drift guard: relative camera distances must not be uniform.
    import numpy as np
    centers = np.array([-p[:3, :3].T @ p[:3, 3] for p in poses])
    ds = np.linalg.norm(np.diff(centers, axis=0), axis=1)
    assert ds.std() / ds.mean() > 0.05, (
        f"consecutive camera distances are uniform "
        f"(std/mean={ds.std()/ds.mean():.3f}) — scale drift regression"
    )
