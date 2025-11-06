import cv2
import numpy as np
from src.video.calibrate import Calibrator
from src.video.video_sfm import VideoSFM
images = ['../data/img1s.png', '../data/img2s.png']

def _reproj_err(K, R, t, X, pts):
    X2 = (R @ X.T + t).T
    proj = (K @ X2.T).T
    proj = proj[:, :2] / proj[:, 2:3]
    return np.median(np.linalg.norm(proj - pts, axis=1))


def test_pose_recovery_and_triangulation():
    im1, im2 = images
    cal = Calibrator()
    K = cal.identify_intrinsics_cheap(im1)
    im1 = cv2.imread(im1)
    im2 = cv2.imread(im2)

    m = cal.extract_all_matches([im1, im2])[0]
    pts1, pts2 = m["pts1"], m["pts2"]
    assert len(pts1) >= 50

    sfm = VideoSFM(device="cpu")
    R, t, inliers = sfm.estimate_pose_from_matches(pts1, pts2, K)
    assert R is not None and t is not None
    pts1_i, pts2_i = pts1[inliers], pts2[inliers]
    assert len(pts1_i) >= 30

    # minimal triangulation check
    X = sfm.triangulate_points(
        pts1_i,
        pts2_i,
        K,
        pose1=np.eye(4),
        pose2=np.block([[R, t], [0, 0, 0, 1]]),
    )
    assert X is not None and X.ndim == 2 and X.shape[1] == 3
    # sanity reprojection (camera 2)
    e2_med = _reproj_err(K, R, t, X, pts2_i[: len(X)])
    assert e2_med <= 3.0
