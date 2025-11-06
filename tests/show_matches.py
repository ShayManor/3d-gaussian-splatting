#!/usr/bin/env python3
import argparse
import cv2
import numpy as np

def preprocess(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    g = clahe.apply(g)
    g = cv2.GaussianBlur(g, (0,0), 0.8)
    g = cv2.addWeighted(g, 1.6, cv2.GaussianBlur(g,(0,0),1.2), -0.6, 0)
    return g

def detect_and_match(im1, im2, ratio=0.75):
    orb = cv2.ORB_create(nfeatures=4000, scaleFactor=1.2, nlevels=8, edgeThreshold=15, WTA_K=2)
    bf  = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    g1, g2 = preprocess(im1), preprocess(im2)
    k1, d1 = orb.detectAndCompute(g1, None)
    k2, d2 = orb.detectAndCompute(g2, None)
    if d1 is None or d2 is None or len(d1) == 0 or len(d2) == 0:
        return [], k1, k2, np.empty((0,2), np.float32), np.empty((0,2), np.float32)
    raw12 = bf.knnMatch(d1, d2, k=2)
    good  = [m[0] for m in raw12 if len(m)==2 and m[0].distance < ratio*m[1].distance]
    # mutual check
    raw21 = bf.knnMatch(d2, d1, k=1)
    rev   = {m[0].queryIdx: m[0].trainIdx for m in raw21 if len(m)==1}
    good  = [m for m in good if rev.get(m.trainIdx, -1) == m.queryIdx]
    pts1  = np.float32([k1[m.queryIdx].pt for m in good]) if good else np.empty((0,2), np.float32)
    pts2  = np.float32([k2[m.trainIdx].pt for m in good]) if good else np.empty((0,2), np.float32)
    return good, k1, k2, pts1, pts2

def default_K_for(img, scale=1.2):
    h, w = img.shape[:2]
    f = scale * max(h, w)
    return np.array([[f, 0, w/2.0],
                     [0, f, h/2.0],
                     [0, 0, 1.0]], dtype=np.float64)

def triangulate_and_metrics(K, R, t, pts1, pts2):
    # undistort to normalized coords
    x1 = cv2.undistortPoints(pts1.reshape(-1,1,2), K, None).reshape(-1,2).T
    x2 = cv2.undistortPoints(pts2.reshape(-1,1,2), K, None).reshape(-1,2).T
    P0 = np.hstack([np.eye(3), np.zeros((3,1))])
    P1 = np.hstack([R, t])
    Xh = cv2.triangulatePoints(P0, P1, x1, x2).T
    X  = Xh[:, :3] / Xh[:, 3:4]     # in cam1 coords
    z1 = X[:, 2]
    X2 = (R @ X.T + t).T
    z2 = X2[:, 2]
    che = float(np.mean((z1 > 0) & (z2 > 0))) if len(X) else 0.0
    # reprojection in cam2
    proj = (K @ (R @ X.T + t)).T
    proj = proj[:, :2] / proj[:, 2:3]
    err2 = np.median(np.linalg.norm(proj - pts2[:len(X)], axis=1)) if len(X) else np.inf
    return X, che, float(err2)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img1", required=False, default="../data/img1s.png")
    ap.add_argument("--img2", required=False, default="../data/img2s.png")
    ap.add_argument("--out_raw", default="matches_raw.jpg")
    ap.add_argument("--out_inl", default="matches_inliers_E.jpg")
    ap.add_argument("--ratio", type=float, default=0.75)
    ap.add_argument("--px", type=float, default=0.8, help="RANSAC threshold (pixels) for Essential")
    ap.add_argument("--fscale", type=float, default=1.2, help="fx,fy ≈ fscale*max(W,H) if no calibration")
    args = ap.parse_args()

    im1 = cv2.imread(args.img1, cv2.IMREAD_COLOR)
    im2 = cv2.imread(args.img2, cv2.IMREAD_COLOR)
    if im1 is None or im2 is None:
        raise SystemExit("Could not read input images.")

    K = default_K_for(im1, scale=args.fscale)

    good, k1, k2, pts1, pts2 = detect_and_match(im1, im2, ratio=args.ratio)
    if len(good) == 0:
        raise SystemExit("No matches after ratio test.")
    raw_vis = cv2.drawMatches(im1, k1, im2, k2, good[:120], None,
                              flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imwrite(args.out_raw, raw_vis)

    # Essential + recoverPose for geometric inliers
    E, mE = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=args.px)
    if E is None or mE is None:
        raise SystemExit("Essential RANSAC failed.")
    ok, R, t, mPose = cv2.recoverPose(E, pts1, pts2, K, mask=mE)
    inl = mPose.ravel().astype(bool)
    inlier_matches = [m for m, keep in zip(good, inl) if keep]
    inlier_ratio = float(np.mean(inl)) if inl.size else 0.0

    inl_vis = cv2.drawMatches(im1, k1, im2, k2, inlier_matches[:120], None,
                              flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imwrite(args.out_inl, inl_vis)

    # triangulate and metrics
    X, che, med_reproj = triangulate_and_metrics(K, R, t, pts1[inl], pts2[inl])

    print(f"raw_matches={len(good)}  E_inliers={int(inl.sum())}  ratio={inlier_ratio:.2f}")
    print(f"cheirality={che:.2f}  median_reproj_px={med_reproj:.2f}")
    print(f"saved: {args.out_raw}, {args.out_inl}")

if __name__ == "__main__":
    main()
