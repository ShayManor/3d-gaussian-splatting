#!/usr/bin/env python3
import argparse
import logging

import cv2
import numpy as np

# use your codebase
from src.video.calibrate import Calibrator
from src.video.video_sfm import VideoSFM
logging.basicConfig(level=logging.INFO)

def draw_lines(img1, img2, pts1, pts2, max_draw=120):
    """Side-by-side draw without cv2.drawMatches (we only have coords)."""
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    H = max(h1, h2)
    canvas = np.zeros((H, w1 + w2, 3), dtype=np.uint8)
    canvas[:h1, :w1] = img1
    canvas[:h2, w1:w1 + w2] = img2
    color = (0, 255, 0)
    for (x1, y1), (x2, y2) in zip(pts1[:max_draw], pts2[:max_draw]):
        p1 = (int(x1), int(y1))
        p2 = (int(x2) + w1, int(y2))
        cv2.circle(canvas, p1, 2, (255, 0, 0), -1)
        cv2.circle(canvas, p2, 2, (255, 0, 0), -1)
        cv2.line(canvas, p1, p2, color, 1, cv2.LINE_AA)
    return canvas

def default_K_for_image(img, scale=1.2):
    h, w = img.shape[:2]
    f = scale * max(h, w)
    return np.array([[f, 0, w/2.0],
                     [0, f, h/2.0],
                     [0, 0, 1.0]], dtype=np.float64)

def triangulate_metrics(K, R, t, pts1, pts2):
    if len(pts1) == 0:
        return np.empty((0,3)), 0.0, float("inf")
    x1 = cv2.undistortPoints(pts1.reshape(-1,1,2), K, None).reshape(-1,2).T
    x2 = cv2.undistortPoints(pts2.reshape(-1,1,2), K, None).reshape(-1,2).T
    P0 = np.hstack([np.eye(3), np.zeros((3,1))])
    P1 = np.hstack([R, t])
    Xh = cv2.triangulatePoints(P0, P1, x1, x2).T
    X  = Xh[:, :3] / Xh[:, 3:4]
    z1 = X[:, 2]
    X2 = (R @ X.T + t).T
    z2 = X2[:, 2]
    che = float(np.mean((z1 > 0) & (z2 > 0))) if len(X) else 0.0
    proj = (K @ (R @ X.T + t)).T
    proj = proj[:, :2] / proj[:, 2:3]
    med_err2 = float(np.median(np.linalg.norm(proj - pts2[:len(X)], axis=1))) if len(X) else float("inf")
    return X, che, med_err2


def debug_gaussian_render(K, R, t, pts1, pts2, X, im1):
    """Debug: render triangulated points as Gaussians to verify SFM"""
    import torch
    from src.gaussians.gaussian_model import GaussianModel
    from src.gaussians.gaussian_rasterizer import GaussianRasterizer

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize Gaussians from triangulated points
    n_pts = len(X)
    gaussians = GaussianModel(n_gaussians=n_pts, device=device)

    with torch.no_grad():
        # Set positions
        gaussians.xyz.data = torch.tensor(X, device=device, dtype=torch.float32)

        # Set scales (uniform small size)
        gaussians.scaling.data = torch.ones(n_pts, 3, device=device) * np.log(0.01)

        # Set opacities (slightly visible)
        gaussians.opacity.data = torch.logit(torch.ones(n_pts, 1, device=device) * 0.5)

    # Setup rasterizer
    K_torch = torch.tensor(K, device=device, dtype=torch.float32)
    rasterizer = GaussianRasterizer(K=K_torch, device=device, backend="gsplat")

    # Create viewpoint (identity pose for first camera)
    h, w = im1.shape[:2]
    pose = np.eye(4)  # First camera at origin

    # Projection matrix helper
    znear, zfar = 0.01, 100.0
    P = torch.zeros(4, 4, device=device)
    P[0, 0] = 2 * K[0, 0] / w
    P[1, 1] = 2 * K[1, 1] / h
    P[0, 2] = 2 * K[0, 2] / w - 1
    P[1, 2] = 2 * K[1, 2] / h - 1
    P[2, 2] = zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    P[3, 2] = 1

    viewpoint = {
        "world_view_transform": torch.tensor(pose, device=device, dtype=torch.float32),
        "projection_matrix": P,
        "image_width": w,
        "image_height": h,
    }

    gaussian_params = {
        "means3D": gaussians.get_xyz,
        "scales": gaussians.get_scaling,
        "rotations": gaussians.get_rotation,
        "opacities": gaussians.get_opacity,
        "shs": gaussians.get_features,
    }

    # Render
    with torch.no_grad():
        out = rasterizer.backend.render_with_depth(
            gaussian_params,
            viewpoint,
            bg_color=torch.zeros(3, device=device),
            render_mode="RGB",
            device=device,
        )
        img = out["render"]
        if img.shape[0] == 3:
            img = img.permute(1, 2, 0)

    # Save outputs
    cv2.imwrite("debug_init_render.png",
                (img.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8))
    cv2.imwrite("debug_init_gt.png", im1)
    print("Saved debug_init_render.png and debug_init_gt.png")
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img1", default="../data/Image.png")
    ap.add_argument("--img2", default="../data/Image2.png")
    ap.add_argument("--out_raw", default="matches_raw.jpg")
    ap.add_argument("--out_inl", default="matches_inliers_E.jpg")
    ap.add_argument("--px", type=float, default=1.2, help="Essential RANSAC pixel threshold")
    ap.add_argument("--fscale", type=float, default=1.2, help="fx,fy ≈ fscale*max(W,H) if no calibration")
    args = ap.parse_args()

    im1 = cv2.imread(args.img1, cv2.IMREAD_COLOR)
    im2 = cv2.imread(args.img2, cv2.IMREAD_COLOR)
    if im1 is None or im2 is None:
        raise SystemExit("Could not read input images.")

    # intrinsics from your calibrator (fast path). If missing, fall back to size-based K.
    cal = Calibrator()
    try:
        K = cal.identify_intrinsics_cheap(args.img1)
    except Exception:
        K = default_K_for_image(im1, scale=args.fscale)

    # matches from your calibrator path (uses your ORB/LoFTR config inside)
    matches = cal.extract_all_matches([im1, im2])
    if not matches:
        raise SystemExit("No matches from Calibrator.")
    pts1 = matches[0]["pts1"].astype(np.float32)
    pts2 = matches[0]["pts2"].astype(np.float32)

    raw_vis = draw_lines(im1, im2, pts1, pts2)
    cv2.imwrite(args.out_raw, raw_vis)

    sfm = VideoSFM(device="cpu")
    R, t, inliers = sfm.estimate_pose_from_matches(pts1, pts2, K)
    if R is None or t is None or inliers is None or not inliers.any():
        raise SystemExit("Essential/pose failed.")
    inl = inliers.astype(bool)

    # inlier-only viz using your coordinates
    inl_vis = draw_lines(im1, im2, pts1[inl], pts2[inl])
    cv2.imwrite(args.out_inl, inl_vis)

    # triangulation and metrics via your SFM code + a metrics helper
    # (your triangulate_points returns only X; we compute cheirality+reproj here)
    X = sfm.triangulate_points(pts1[inl], pts2[inl], K, pose1=np.eye(4),
                               pose2=np.block([[R, t], [0, 0, 0, 1]]))
    if X is not None and len(X) > 0:
        debug_gaussian_render(K, R, t, pts1[inl], pts2[inl], X, im1)
    if X is None:
        X = np.empty((0,3))
        che, med = 0.0, float("inf")
    else:
        _, che, med = triangulate_metrics(K, R, t, pts1[inl], pts2[inl])

    raw_cnt = len(pts1)
    inl_cnt = int(inl.sum())
    inl_ratio = 0.0 if raw_cnt == 0 else inl_cnt / float(raw_cnt)

    print(f"raw_matches={raw_cnt}  E_inliers={inl_cnt}  ratio={inl_ratio:.2f}")
    print(f"cheirality={che:.2f}  median_reproj_px={med:.2f}")
    print(f"saved: {args.out_raw}, {args.out_inl}")

if __name__ == "__main__":
    main()
