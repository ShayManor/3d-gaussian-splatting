import cv2
import numpy as np
import pytest
from src.video.calibrate import Calibrator
images = ['../data/img1s.png', '../data/img2s.png']

def _inlier_ratio(mask):
    if mask is None:
        return 0.0
    m = mask.ravel().astype(bool)
    return float(np.mean(m)) if m.size else 0.0

def test_calibrator_matches_basic():
    im1, im2 = images
    im1 = cv2.imread(im1)
    im2 = cv2.imread(im2)
    cal = Calibrator()
    matches = cal.extract_all_matches([im1, im2])
    assert matches and "pts1" in matches[0] and "pts2" in matches[0]
    pts1, pts2 = matches[0]["pts1"], matches[0]["pts2"]
    assert isinstance(pts1, np.ndarray) and isinstance(pts2, np.ndarray)
    assert pts1.shape == pts2.shape and pts1.ndim == 2 and pts1.shape[1] == 2
    assert len(pts1) >= 30  # small viewpoint change should yield dozens

def test_epipolar_geometry():
    im1, im2 = images
    im1 = cv2.imread(im1)
    im2 = cv2.imread(im2)
    cal = Calibrator()
    m = cal.extract_all_matches([im1, im2])[0]
    pts1, pts2 = m["pts1"], m["pts2"]
    assert len(pts1) >= 30
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 1.0, 0.999)
    assert F is not None
    assert _inlier_ratio(mask) >= 0.5  # geometry should support many inliers

def test_planar_fallback_homography():
    im1, im2 = images
    im1 = cv2.imread(im1)
    im2 = cv2.imread(im2)
    cal = Calibrator()
    m = cal.extract_all_matches([im1, im2])[0]
    pts1, pts2 = m["pts1"], m["pts2"]
    assert len(pts1) >= 30
    _, mF = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 1.0, 0.999)
    _, mH = cv2.findHomography(pts1, pts2, cv2.RANSAC, 2.0, confidence=0.999)
    rF = (mF.ravel().mean() if mF is not None else 0.0)
    rH = (mH.ravel().mean() if mH is not None else 0.0)
    assert max(rF, rH) >= 0.5  # accept strong F or strong H (planar scenes)
