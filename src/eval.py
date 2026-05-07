import argparse
import logging
import math
import pickle
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm

import wandb

from src.gaussians.gaussian_model import GaussianModel
from src.gaussians.gaussian_rasterizer import GaussianRasterizer
from src.training.export import export
from src.video.video_loader import VideoLoader

logging.basicConfig(level=logging.INFO)


def _load_state(path: str, device: str) -> dict:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict):
        for k in ("model_state", "state_dict"):
            if k in ckpt and isinstance(ckpt[k], dict):
                return ckpt[k]
    return ckpt


def _load_model(path: str, device: str) -> GaussianModel:
    state = _load_state(path, device)
    n = state["xyz"].shape[0]
    model = GaussianModel(n_gaussians=n, device=device)
    model.load_state_dict(state)
    return model.to(device)


def _projection_matrix(K, width, height, device, znear=0.01, zfar=100.0):
    P = torch.zeros(4, 4, device=device)
    P[0, 0] = 2 * K[0, 0] / width
    P[1, 1] = 2 * K[1, 1] / height
    P[0, 2] = 2 * K[0, 2] / width - 1
    P[1, 2] = 2 * K[1, 2] / height - 1
    P[2, 2] = zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    P[3, 2] = 1
    return P


def _ssim(img1: torch.Tensor, img2: torch.Tensor) -> float:
    C1, C2 = 0.01**2, 0.03**2
    mu1 = F.avg_pool2d(img1, 3, 1, padding=1)
    mu2 = F.avg_pool2d(img2, 3, 1, padding=1)
    mu1_sq, mu2_sq, mu1_mu2 = mu1.pow(2), mu2.pow(2), mu1 * mu2
    s1 = F.avg_pool2d(img1 * img1, 3, 1, padding=1) - mu1_sq
    s2 = F.avg_pool2d(img2 * img2, 3, 1, padding=1) - mu2_sq
    s12 = F.avg_pool2d(img1 * img2, 3, 1, padding=1) - mu1_mu2
    return float(
        (((2 * mu1_mu2 + C1) * (2 * s12 + C2)) / ((mu1_sq + mu2_sq + C1) * (s1 + s2 + C2)))
        .mean()
        .item()
    )


def _psnr(pred: torch.Tensor, gt: torch.Tensor) -> float:
    mse = F.mse_loss(pred, gt).item()
    if mse < 1e-10:
        return 100.0
    return float(-10.0 * math.log10(mse))


def _init_wandb(args) -> Optional["wandb.sdk.wandb_run.Run"]:
    if args.wandb_mode == "disabled":
        return None
    try:
        return wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            mode=args.wandb_mode,
            name=args.wandb_run_name,
            tags=["eval"],
            config=vars(args),
            job_type="eval",
            settings=wandb.Settings(start_method="thread"),
        )
    except Exception as e:
        logging.warning(f"wandb.init failed ({e}); skipping wandb logging")
        return None


def main():
    ap = argparse.ArgumentParser(description="Render views from a trained 3DGS model")
    ap.add_argument("--model", required=True, help="Path to final_model.pth")
    ap.add_argument("--sfm-cache", required=True, help="Path to cache/<stem>_sfm.pkl")
    ap.add_argument("--output", default="./output/eval", help="Where to save renders")
    ap.add_argument("--num-views", type=int, default=12)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--wandb_project", default="3d-gaussian-splatting")
    ap.add_argument("--wandb_entity", default=None)
    ap.add_argument(
        "--wandb_mode",
        default="online",
        choices=["online", "offline", "disabled"],
    )
    ap.add_argument("--wandb_run_name", default=None)
    args = ap.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    run = _init_wandb(args)

    # Re-export PLY next to the renders for convenience
    ply_path = out / "model.ply"
    export(args.model, str(ply_path))

    with open(args.sfm_cache, "rb") as f:
        sfm = pickle.load(f)

    poses = sfm["poses"]
    K_np = sfm["intrinsics"]
    K = torch.tensor(K_np, device=args.device, dtype=torch.float32)
    frame_indices = sfm["frame_indices"]
    video_path = sfm["video_path"]

    model = _load_model(args.model, args.device)
    rasterizer = GaussianRasterizer(K=K, device=args.device, backend="gsplat")

    loader = VideoLoader(video_path, cache_frames=False)
    width = int(loader.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(loader.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    n_total = len(poses)
    n = min(args.num_views, n_total)
    pick = np.linspace(0, n_total - 1, n).astype(int)

    xyz = model.xyz.detach()
    scene_min = xyz.min(0).values.cpu().numpy()
    scene_max = xyz.max(0).values.cpu().numpy()
    scene_extent = float(np.linalg.norm(scene_max - scene_min))

    print(f"[eval] gaussians: {xyz.shape[0]:,}")
    print(f"[eval] scene min: {scene_min}")
    print(f"[eval] scene max: {scene_max}")
    print(f"[eval] mean opacity: {model.get_opacity.mean().item():.4f}")
    print(f"[eval] rendering {n} of {n_total} training views to {out}")

    if run is not None:
        run.summary["model/n_gaussians"] = int(xyz.shape[0])
        run.summary["model/scene_extent"] = scene_extent
        run.summary["model/mean_opacity"] = float(model.get_opacity.mean().item())
        run.summary["model/mean_scale"] = float(model.get_scaling.mean().item())

    params = {
        "means3D": model.get_xyz,
        "scales": model.get_scaling,
        "rotations": model.get_rotation,
        "opacities": model.get_opacity,
        "shs": model.get_features,
    }
    proj = _projection_matrix(K, width, height, args.device)
    bg = torch.zeros(3, device=args.device)

    l1_sum = ssim_sum = psnr_sum = 0.0
    n_metric = 0
    gallery = []

    for j, idx in enumerate(tqdm(pick, desc="Rendering")):
        pose = torch.tensor(poses[idx], device=args.device, dtype=torch.float32)
        viewpoint = {
            "world_view_transform": pose,
            "projection_matrix": proj,
            "image_width": width,
            "image_height": height,
        }
        with torch.no_grad():
            res = rasterizer.backend.render_with_depth(
                params, viewpoint, bg_color=bg,
                render_mode="RGB", device=args.device,
            )
        rendered = res["render"]
        if rendered.shape[0] == 3:
            rendered = rendered.permute(1, 2, 0)
        rendered_t = rendered.clamp(0, 1)
        rendered_np = (rendered_t.cpu().numpy() * 255).astype(np.uint8)

        gt_bgr = loader.get_frame(int(frame_indices[idx]))
        if gt_bgr is None:
            cv2.imwrite(str(out / f"view_{j:03d}_render.png"), rendered_np)
            continue

        if gt_bgr.shape[:2] != rendered_np.shape[:2]:
            rendered_np = cv2.resize(rendered_np, (gt_bgr.shape[1], gt_bgr.shape[0]))

        # Metrics
        gt_t = torch.tensor(gt_bgr / 255.0, device=args.device, dtype=torch.float32)
        if gt_t.shape != rendered_t.shape:
            rendered_t_for_metric = torch.tensor(
                rendered_np / 255.0, device=args.device, dtype=torch.float32
            )
        else:
            rendered_t_for_metric = rendered_t
        l1 = F.l1_loss(rendered_t_for_metric, gt_t).item()
        ssim = _ssim(
            rendered_t_for_metric.permute(2, 0, 1).unsqueeze(0),
            gt_t.permute(2, 0, 1).unsqueeze(0),
        )
        psnr = _psnr(rendered_t_for_metric, gt_t)
        l1_sum += l1
        ssim_sum += ssim
        psnr_sum += psnr
        n_metric += 1

        side = np.hstack([gt_bgr, rendered_np])
        cv2.imwrite(str(out / f"view_{j:03d}.png"), side)

        if run is not None and len(gallery) < 16:
            gallery.append(
                wandb.Image(
                    side[..., ::-1],  # BGR -> RGB
                    caption=f"view {j} | l1={l1:.3f} ssim={ssim:.3f} psnr={psnr:.2f}",
                )
            )

    if n_metric > 0:
        mean_l1 = l1_sum / n_metric
        mean_ssim = ssim_sum / n_metric
        mean_psnr = psnr_sum / n_metric
        print(f"[eval] mean l1={mean_l1:.4f} ssim={mean_ssim:.4f} psnr={mean_psnr:.2f}")
        if run is not None:
            run.summary["eval/l1"] = mean_l1
            run.summary["eval/ssim"] = mean_ssim
            run.summary["eval/psnr"] = mean_psnr
            run.summary["eval/n_views"] = n_metric

    if run is not None:
        if gallery:
            run.log({"eval/gallery": gallery})
        try:
            art = wandb.Artifact(name=f"3dgs-eval-{run.id}", type="eval")
            art.add_file(str(ply_path))
            for p in sorted(out.glob("view_*.png"))[:16]:
                art.add_file(str(p))
            run.log_artifact(art)
        except Exception as e:
            logging.warning(f"wandb eval artifact failed: {e}")
        run.finish()

    print(f"[eval] saved {n} renders + model.ply to {out}")


if __name__ == "__main__":
    main()
