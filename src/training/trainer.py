import math
import time
from logging import log, WARNING, INFO, DEBUG
from pathlib import Path
from typing import List, Dict, Optional

import cv2
import numpy as np
import torch
from torch import GradScaler, autocast
from torch.optim import Adam
from tqdm import tqdm

from src.gaussians.gaussian_model import GaussianModel
from src.gaussians.gaussian_rasterizer import GaussianRasterizer
from src.gaussians.training_config import TrainingConfig
from src.training.export import export
from src.video.video_loader import VideoLoader

from sklearn.neighbors import NearestNeighbors
from torch.nn import functional as F
import matplotlib.pyplot as plt


class GaussianTrainer:
    def __init__(self, config: TrainingConfig, device, wandb_run=None):
        self.config = config
        self.device = device
        self.wandb_run = wandb_run

        self.scaler = GradScaler() if config.use_mixed_precision else None

        self.iteration = 0
        self.loss_history = []
        self.opacity_history = []
        self.gaussian_history = []

        self._step_start_time = None

        # Cumulative densify counters for W&B trend curves.
        self._cum_cloned = 0
        self._cum_split = 0
        self._cum_pruned = 0
        self._densify_event_idx = 0

        # Shared VideoLoader cache: populated in train() with pre-decoded frames
        # so train batches and validation reads are dict lookups, not H.264 seeks.
        self._video_loaders: Dict[str, "VideoLoader"] = {}

    # ----- W&B helpers ---------------------------------------------------

    def _wandb_log(self, data: Dict, step: Optional[int] = None):
        if self.wandb_run is None:
            return
        try:
            self.wandb_run.log(data, step=step if step is not None else self.iteration)
        except Exception as e:
            log(WARNING, f"wandb log failed: {e}")

    def _gpu_stats(self) -> Dict:
        if not torch.cuda.is_available() or str(self.device) == "cpu":
            return {}
        d = torch.device(self.device) if not isinstance(self.device, torch.device) else self.device
        return {
            "gpu/mem_alloc_mb": torch.cuda.memory_allocated(d) / 1e6,
            "gpu/mem_reserved_mb": torch.cuda.memory_reserved(d) / 1e6,
            "gpu/mem_max_alloc_mb": torch.cuda.max_memory_allocated(d) / 1e6,
        }

    @staticmethod
    def _psnr(pred: torch.Tensor, gt: torch.Tensor) -> float:
        mse = F.mse_loss(pred, gt).item()
        if mse < 1e-10:
            return 100.0
        return float(-10.0 * math.log10(mse))

    def _active_sh_degree(self) -> int:
        """
        Current SH degree under the warmup schedule. Starts at 0 (DC only) and
        increments by 1 every sh_increment_interval iterations until reaching
        sh_degree_max. Higher-order coefficients stay zero (their gradient is
        masked out by gsplat for degrees above the current cap) until enabled.
        """
        return min(
            int(self.iteration // self.config.sh_increment_interval),
            int(self.config.sh_degree_max),
        )

    @staticmethod
    def _quantile_safe(x: torch.Tensor, qs: torch.Tensor, max_n: int = 1_000_000) -> torch.Tensor:
        """
        torch.quantile has an undocumented hard limit of ~2^24 elements and
        crashes with `quantile() input tensor is too large` past it. With ~5.5M
        gaussians × 3 axes the scaling tensor blows past this. Subsample for
        logging — the percentile estimate is statistically indistinguishable.
        """
        if x.numel() > max_n:
            idx = torch.randint(0, x.numel(), (max_n,), device=x.device)
            x = x[idx]
        return torch.quantile(x, qs)

    @staticmethod
    def _stclamp(x: torch.Tensor) -> torch.Tensor:
        # Forward: clamp to [0,1] so L1/SSIM stay bounded when gsplat returns
        # un-clamped RGB. Backward: identity, so gaussians whose SH eval drifts
        # out of [0,1] still receive a corrective gradient.
        return x + (x.clamp(0.0, 1.0) - x).detach()

    # ----- Initialization -------------------------------------------------

    def _initialize_gaussians(self, merged_data):
        points_3d = merged_data["points_3d"]
        colors = merged_data["colors"]
        n = len(points_3d)

        if n == 0:
            log(WARNING, "No 3D points found! Initializing random gaussians")
            return GaussianModel(
                n_gaussians=int(self.config.initial_gaussians), device=str(self.device)
            )

        n_gaussians = min(
            max(n * 3, int(self.config.initial_gaussians)),
            int(self.config.max_gaussians // 2),
        )

        print(f"Creating {n_gaussians:,} initial Gaussians from {n:,} 3D points")

        gaussians = GaussianModel(n_gaussians=n_gaussians, device=str(self.device))

        with torch.no_grad():
            points_tensor = torch.tensor(
                points_3d, device=self.device, dtype=torch.float32
            )

            if n_gaussians <= n:
                indices = torch.randperm(n)[:n_gaussians]
            else:
                indices = torch.randint(0, n, (n_gaussians,))

            gaussians.xyz.data = (
                points_tensor[indices]
                + torch.randn(n_gaussians, 3, device=self.device) * 0.001
            )

            if len(colors) > 0:
                colors_tensor = torch.tensor(
                    colors, device=self.device, dtype=torch.float32
                )
                sh_colors = self._rgb_to_sh0(colors_tensor[indices])
                gaussians.features_dc.data[:, 0, :] = sh_colors

            self._initialize_scales_smart(gaussians, points_tensor)

            gaussians.opacity.data = torch.logit(
                torch.ones(n_gaussians, 1, device=self.device) * 0.005
            )

            log(
                DEBUG,
                "pts3d mean/std/min/max",
                points_3d.mean(0),
                points_3d.std(0),
                points_3d.min(0),
                points_3d.max(0),
            )
            log(DEBUG, "first pose translation", merged_data["all_poses"][0][:3, 3])

        return gaussians

    def _initialize_scales_smart(self, gaussians, points):
        positions = gaussians.xyz.data.cpu().numpy()
        all_points = points.cpu().numpy()

        nbrs = NearestNeighbors(
            n_neighbors=min(7, len(all_points)), algorithm="kd_tree"
        )
        nbrs.fit(all_points)
        distances, _ = nbrs.kneighbors(positions)

        avg_distances = distances[:, 1:].mean(axis=1) if distances.shape[1] > 1 else distances[:, 0]

        scales = torch.tensor(avg_distances, device=self.device, dtype=torch.float32)
        scales = scales.clamp(min=1e-9)
        scales = torch.log(scales.unsqueeze(1).expand(-1, 3))

        gaussians.scaling.data = scales

    def debug_reprojection(self, points_3d, poses, K, frame, out_path="debug_reproj.png"):
        pose0 = poses[0]
        K = np.array(K)

        # Skip non-finite/extreme points before matmul — outliers (z near 0 or
        # z >> scene scale) overflow the projection and pollute the image.
        X = np.asarray(points_3d, dtype=np.float64)
        finite = np.isfinite(X).all(axis=1)
        X = X[finite]
        if len(X) == 0:
            cv2.imwrite(out_path, frame)
            return frame

        X_h = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            X_c = (pose0 @ X_h.T).T
        Z = X_c[:, 2]
        good = (Z > 1e-3) & np.isfinite(Z) & np.isfinite(X_c).all(axis=1)
        X_c = X_c[good]
        if len(X_c) == 0:
            cv2.imwrite(out_path, frame)
            return frame

        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            proj = (K @ X_c[:, :3].T).T
            uv = proj[:, :2] / proj[:, 2:3]
        valid = np.isfinite(uv).all(axis=1)
        uv = uv[valid]

        img = frame.copy()
        for u, v in uv.astype(int):
            if 0 <= u < img.shape[1] and 0 <= v < img.shape[0]:
                img[v, u] = (0, 0, 255)

        cv2.imwrite(out_path, img)
        return img

    # ----- View bookkeeping ----------------------------------------------

    def _build_views(self, merged_data) -> List[Dict]:
        all_views = []
        for i, video_info in enumerate(merged_data["video_info"]):
            video_path = video_info["path"]
            poses = merged_data["all_poses"][i]
            K = merged_data["all_intrinsics"][i]

            loader = self._get_loader(video_path)
            width = int(loader.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(loader.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

            frame_indices = merged_data["frame_indices"][i]
            for j, frame_idx in enumerate(frame_indices[: len(poses)]):
                all_views.append(
                    {
                        "video_path": video_path,
                        "frame_idx": int(frame_idx),
                        "pose": poses[j],
                        "K": K,
                        "width": width,
                        "height": height,
                    }
                )
        return all_views

    def _split_views(self, all_views: List[Dict]):
        n_total = len(all_views)
        if n_total < 4:
            return all_views, []

        rng = np.random.RandomState(self.config.val_seed)
        n_val = min(
            max(1, int(round(n_total * self.config.val_fraction))),
            self.config.val_max_views,
        )
        permuted = rng.permutation(n_total)
        val_idx = set(int(i) for i in permuted[:n_val])
        train_views = [v for i, v in enumerate(all_views) if i not in val_idx]
        val_views = [all_views[i] for i in sorted(val_idx)]
        log(INFO, f"Train views: {len(train_views)}, Val views: {len(val_views)}")
        return train_views, val_views

    def _to_tensor_view(self, view: Dict, frame: np.ndarray) -> Dict:
        return {
            "image": torch.tensor(
                frame / 255.0, device=self.device, dtype=torch.float32
            ),
            "pose": torch.tensor(
                view["pose"], device=self.device, dtype=torch.float32
            ),
            "K": torch.tensor(view["K"], device=self.device, dtype=torch.float32),
            "width": view["width"],
            "height": view["height"],
        }

    def _prepare_video_loaders(self, all_views: List[Dict]):
        """
        Build one VideoLoader per source video and pre-decode every frame any
        view will touch. Random-access H.264 reads cost 30–80 ms each; a single
        sequential pass costs ~1 ms per frame. With ~100 frames at 1080×1920
        the whole video fits in ~600 MB of CPU RAM, so we just hold it.
        """
        per_video: Dict[str, set] = {}
        for v in all_views:
            per_video.setdefault(v["video_path"], set()).add(int(v["frame_idx"]))
        for video_path, idx_set in per_video.items():
            loader = VideoLoader(video_path, cache_frames=True)
            loader.preload(sorted(idx_set))
            self._video_loaders[video_path] = loader
            log(
                INFO,
                f"Pre-decoded {len(loader.frame_cache)} frames from {video_path} into RAM",
            )

    def _get_loader(self, video_path: str) -> VideoLoader:
        loader = self._video_loaders.get(video_path)
        if loader is None:
            loader = VideoLoader(video_path, cache_frames=True)
            self._video_loaders[video_path] = loader
        return loader

    def _create_train_loader(self, train_views: List[Dict]):
        if not train_views:
            raise ValueError("No training views available")

        def gen():
            while True:
                indices = np.random.choice(
                    len(train_views), self.config.batch_size, replace=True
                )
                batch = []
                for idx in indices:
                    view = train_views[idx]
                    frame = self._get_loader(view["video_path"]).get_frame(view["frame_idx"])
                    if frame is not None:
                        batch.append(self._to_tensor_view(view, frame))
                if batch:
                    yield batch

        return gen()

    # ----- Optimizer / projection ----------------------------------------

    def _setup_optimizer(self, gaussians):
        params = [
            {"params": [gaussians.xyz], "lr": self.config.position_lr_init, "name": "xyz"},
            {"params": [gaussians.features_dc], "lr": 0.0025, "name": "f_dc"},
            {"params": [gaussians.features_rest], "lr": 0.0025 / 20.0, "name": "f_rest"},
            {"params": [gaussians.opacity], "lr": 0.05, "name": "opacity"},
            {"params": [gaussians.scaling], "lr": 0.005, "name": "scaling"},
            {"params": [gaussians.rotation], "lr": 0.001, "name": "rotation"},
        ]
        return Adam(params, lr=0.0, eps=1e-15)

    def _get_projection_matrix(self, K, width, height) -> torch.Tensor:
        znear, zfar = 0.01, 100.0
        P = torch.zeros(4, 4, device=self.device)
        P[0, 0] = 2 * K[0, 0] / width
        P[1, 1] = 2 * K[1, 1] / height
        P[0, 2] = 2 * K[0, 2] / width - 1
        P[1, 2] = 2 * K[1, 2] / height - 1
        P[2, 2] = zfar / (zfar - znear)
        P[2, 3] = -(zfar * znear) / (zfar - znear)
        P[3, 2] = 1
        return P

    def _compute_ssim(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        C1, C2 = 0.01**2, 0.03**2

        mu1 = F.avg_pool2d(img1, 3, 1, padding=1)
        mu2 = F.avg_pool2d(img2, 3, 1, padding=1)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.avg_pool2d(img1 * img1, 3, 1, padding=1) - mu1_sq
        sigma2_sq = F.avg_pool2d(img2 * img2, 3, 1, padding=1) - mu2_sq
        sigma12 = F.avg_pool2d(img1 * img2, 3, 1, padding=1) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
            (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        )
        return ssim_map.mean()

    def _update_learning_rate(self, optimizer: Adam):
        progress = min(self.iteration / self.config.position_lr_max_steps, 1.0)
        lr = (
            self.config.position_lr_init
            * (self.config.position_lr_final / self.config.position_lr_init) ** progress
        )
        for group in optimizer.param_groups:
            if group.get("name") == "xyz":
                group["lr"] = lr
        return lr

    def _save_checkpoint(self, gaussians: GaussianModel, optimizer: Adam, output_path: Path):
        model_pth = output_path / f"checkpoint_{self.iteration}.pth"
        torch.save(
            {
                "iteration": self.iteration,
                "model_state": gaussians.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "n_gaussians": gaussians.xyz.shape[0],
            },
            model_pth,
        )
        export(str(model_pth), str(model_pth).replace("pth", "ply"))

    # ----- Render helpers -------------------------------------------------

    def _gaussian_params(self, gaussians):
        return {
            "means3D": gaussians.get_xyz,
            "scales": gaussians.get_scaling,
            "rotations": gaussians.get_rotation,
            "opacities": gaussians.get_opacity,
            "shs": gaussians.get_features,
        }

    def _render_view(self, gaussians, rasterizer, view_t):
        viewpoint = {
            "world_view_transform": view_t["pose"],
            "projection_matrix": self._get_projection_matrix(
                view_t["K"], view_t["width"], view_t["height"]
            ),
            "image_width": view_t["width"],
            "image_height": view_t["height"],
        }
        return rasterizer.backend.render_with_depth(
            self._gaussian_params(gaussians),
            viewpoint,
            bg_color=torch.zeros(3, device=self.device),
            render_mode="RGB",
            device=str(self.device),
            sh_degree=self._active_sh_degree(),
        )

    @staticmethod
    def _render_to_uint8(img: torch.Tensor) -> np.ndarray:
        if img.shape[0] == 3:
            img = img.permute(1, 2, 0)
        return (img.clamp(0, 1).detach().cpu().numpy() * 255).astype(np.uint8)

    # ----- Training step --------------------------------------------------

    def _scale_ratio_reg(self, gaussians) -> torch.Tensor:
        """
        Splatfacto/PhysGaussian-style hinge penalty on per-gaussian scale
        anisotropy. Zero when max/min scale ratio is at or below
        `config.scale_reg_max_ratio`; grows linearly above. Prevents the
        "needle" failure mode where one axis collapses to ~0 while another
        sits at the scale clamp ceiling.
        """
        scales = gaussians.get_scaling  # (N, 3), already exp-activated
        ratio = scales.amax(dim=-1) / scales.amin(dim=-1).clamp(min=1e-8)
        hinge = ratio.clamp(min=self.config.scale_reg_max_ratio) - self.config.scale_reg_max_ratio
        return self.config.scale_reg_weight * hinge.mean()

    def _training_step(self, gaussians, rasterizer, batch, optimizer):
        optimizer.zero_grad()

        gaussian_params = self._gaussian_params(gaussians)
        active_sh_degree = self._active_sh_degree()

        total_loss = torch.tensor(0.0, device=self.device)
        l1_total = 0.0
        ssim_total = 0.0
        psnr_total = 0.0

        for view_data in batch:
            viewpoint = {
                "world_view_transform": view_data["pose"],
                "projection_matrix": self._get_projection_matrix(
                    view_data["K"], view_data["width"], view_data["height"]
                ),
                "image_width": view_data["width"],
                "image_height": view_data["height"],
            }

            with autocast(
                enabled=self.config.use_mixed_precision, device_type=str(self.device)
            ):
                rendered = rasterizer.backend.render_with_depth(
                    gaussian_params,
                    viewpoint,
                    bg_color=torch.zeros(3, device=self.device),
                    render_mode="RGB",
                    device=str(self.device),
                    sh_degree=active_sh_degree,
                )

            rendered_img = rendered["render"]
            if rendered_img.shape[0] == 3:
                rendered_img = rendered_img.permute(1, 2, 0)
            gt_image = view_data["image"]

            rendered_for_loss = self._stclamp(rendered_img)
            l1 = F.l1_loss(rendered_for_loss, gt_image)
            ssim = self._compute_ssim(
                rendered_for_loss.permute(2, 0, 1).unsqueeze(0),
                gt_image.permute(2, 0, 1).unsqueeze(0),
            )
            ssim_loss = 1.0 - ssim
            loss = (1.0 - self.config.lambda_dssim) * l1 + self.config.lambda_dssim * ssim_loss
            total_loss = total_loss + loss

            l1_total += float(l1.detach().item())
            ssim_total += float(ssim.detach().item())
            psnr_total += self._psnr(rendered_img.detach().clamp(0.0, 1.0), gt_image.detach())

        n = len(batch)
        total_loss = total_loss / n

        scale_reg = self._scale_ratio_reg(gaussians)
        total_loss = total_loss + scale_reg

        if self.scaler:
            self.scaler.scale(total_loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            total_loss.backward()
            optimizer.step()

        return {
            "loss": float(total_loss.item()),
            "l1": l1_total / n,
            "ssim": ssim_total / n,
            "psnr": psnr_total / n,
            "scale_reg": float(scale_reg.detach().item()),
        }

    # ----- Validation -----------------------------------------------------

    @torch.no_grad()
    def _validate(self, gaussians, rasterizer, val_views: List[Dict]) -> Optional[Dict]:
        if not val_views:
            return None

        l1_sum = 0.0
        ssim_sum = 0.0
        psnr_sum = 0.0
        n = 0
        gallery = []  # (gt_uint8, render_uint8, l1, ssim, psnr) per view, capped

        max_gallery = 6

        for view in val_views:
            frame = self._get_loader(view["video_path"]).get_frame(view["frame_idx"])
            if frame is None:
                continue
            view_t = self._to_tensor_view(view, frame)
            res = self._render_view(gaussians, rasterizer, view_t)
            img = res["render"]
            if img.shape[0] == 3:
                img = img.permute(1, 2, 0)
            img = img.clamp(0.0, 1.0)
            gt = view_t["image"]

            l1 = F.l1_loss(img, gt).item()
            ssim = self._compute_ssim(
                img.permute(2, 0, 1).unsqueeze(0),
                gt.permute(2, 0, 1).unsqueeze(0),
            ).item()
            psnr = self._psnr(img, gt)
            l1_sum += l1
            ssim_sum += ssim
            psnr_sum += psnr
            n += 1

            if len(gallery) < max_gallery:
                gallery.append(
                    (self._render_to_uint8(gt), self._render_to_uint8(img), l1, ssim, psnr, view["frame_idx"])
                )

        if n == 0:
            return None

        out = {
            "val/l1": l1_sum / n,
            "val/ssim": ssim_sum / n,
            "val/psnr": psnr_sum / n,
            "val/n_views": n,
        }
        if gallery and self.wandb_run is not None:
            try:
                import wandb
                # First entry: large side-by-side for backward compat.
                gt0, ren0, l10, ss0, ps0, fi0 = gallery[0]
                out["val/sample"] = wandb.Image(
                    np.hstack([gt0, ren0])[..., ::-1],
                    caption=f"iter {self.iteration} f{fi0} | l1={l10:.3f} psnr={ps0:.2f} | left=GT right=render",
                )
                # Multi-view gallery
                gallery_imgs = [
                    wandb.Image(
                        np.hstack([gt, ren])[..., ::-1],
                        caption=f"f{fi} l1={l1:.3f} ssim={ss:.3f} psnr={ps:.2f}",
                    )
                    for gt, ren, l1, ss, ps, fi in gallery
                ]
                out["val/gallery"] = gallery_imgs
            except Exception as e:
                log(WARNING, f"wandb val image failed: {e}")
        return out

    # ----- Main training --------------------------------------------------

    def _compute_scene_geometry(self, merged_data):
        """
        Compute robust scene_extent (median radius from median centroid, capped
        by 2 × median camera depth), the bbox dimensions, and a list of point
        depths. Returns (scene_extent, scene_bbox, n_outliers_beyond_extent,
        median_depth, in_extent_mask).

        in_extent_mask is per-point: True for points within scene_extent of
        the median centroid. Caller can use this to filter the SfM cloud
        before initializing gaussians, so a stale cache with bad triangulations
        cannot seed gaussians at z=5000+.
        """
        points_3d = np.asarray(merged_data["points_3d"], dtype=np.float64)
        if len(points_3d) == 0:
            return 10.0, np.zeros(3), 0, 0.0, np.zeros(0, dtype=bool)

        centroid = np.median(points_3d, axis=0)
        radii = np.linalg.norm(points_3d - centroid, axis=1)
        scene_bbox = points_3d.max(0) - points_3d.min(0)

        if len(radii) >= 8 and np.isfinite(radii).any():
            med_radius = float(np.median(radii[np.isfinite(radii)]))
            depths = []
            Xh = np.hstack([points_3d, np.ones((len(points_3d), 1))])
            for p_arr in merged_data["all_poses"]:
                for pose in p_arr:
                    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                        z = (pose @ Xh.T).T[:, 2]
                    z = z[(z > 0) & np.isfinite(z)]
                    if len(z):
                        depths.append(float(np.median(z)))
            depth_cap = 2.0 * float(np.median(depths)) if depths else float("inf")
            scene_extent = float(min(2.0 * med_radius, depth_cap))
            median_depth = float(np.median(depths)) if depths else 0.0
        else:
            scene_extent = float(np.linalg.norm(scene_bbox))
            median_depth = 0.0

        in_extent = radii <= scene_extent
        n_outliers = int((~in_extent).sum())
        return scene_extent, scene_bbox, n_outliers, median_depth, in_extent

    def _filter_merged_by_extent(self, merged_data, in_extent_mask):
        """
        Drop SfM points (and their colors) that lie outside scene_extent of the
        median centroid. Returns a shallow copy of merged_data with the cloud
        filtered. Poses, intrinsics, and frame indices are untouched.
        """
        if in_extent_mask.all() or len(in_extent_mask) == 0:
            return merged_data
        filtered = dict(merged_data)
        filtered["points_3d"] = np.asarray(merged_data["points_3d"])[in_extent_mask]
        if "colors" in merged_data and len(merged_data["colors"]) == len(in_extent_mask):
            filtered["colors"] = np.asarray(merged_data["colors"])[in_extent_mask]
        return filtered

    def train(self, merged_data, output_dir) -> GaussianModel:
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)

        # Compute scene geometry FIRST so we can drop outlier SfM points before
        # initializing gaussians. Otherwise a single point at z=5000 spawns a
        # cluster of useless gaussians that pull every densify event downstream.
        scene_extent, scene_bbox, n_outliers, median_depth, in_extent = (
            self._compute_scene_geometry(merged_data)
        )
        log(
            INFO,
            f"scene_extent={scene_extent:.3f}  outliers_beyond_extent={n_outliers}/"
            f"{len(merged_data['points_3d'])}  median_depth={median_depth:.3f}",
        )
        merged_data = self._filter_merged_by_extent(merged_data, in_extent)

        print(f"Initializing with {len(merged_data['points_3d'])} 3D points (after outlier filter)")

        gaussians = self._initialize_gaussians(merged_data)
        print(f"Initialized {gaussians.xyz.shape[0]} Gaussians")

        # Debug reprojection of SfM points onto frame 0
        poses0 = merged_data["all_poses"][0]
        K0 = merged_data["all_intrinsics"][0]
        video0 = merged_data["video_info"][0]["path"]
        frame0 = self._get_loader(video0).get_frame(0)
        reproj_img = self.debug_reprojection(
            merged_data["points_3d"], poses0, K0, frame0,
            out_path="debug_reproj_points.png",
        )
        if self.wandb_run is not None:
            try:
                import wandb
                self._wandb_log(
                    {
                        "sfm/reprojection": wandb.Image(
                            reproj_img[..., ::-1],
                            caption="SfM points reprojected onto frame 0",
                        ),
                        "sfm/n_points": len(merged_data["points_3d"]),
                        "sfm/n_videos": len(merged_data["video_info"]),
                    },
                    step=0,
                )
            except Exception as e:
                log(WARNING, f"wandb sfm log failed: {e}")

        optimizer = self._setup_optimizer(gaussians)

        K = torch.tensor(
            merged_data["all_intrinsics"][0], device=self.device, dtype=torch.float32
        )
        rasterizer = GaussianRasterizer(
            K=K, device=str(self.device), enable_caching=True, backend="gsplat"
        )

        # Train/val split
        all_views = self._build_views(merged_data)
        train_views, val_views = self._split_views(all_views)
        # Pre-decode every frame the train/val loops will touch. With 102 views
        # at 1080×1920 this is ~600 MB held in CPU RAM; the per-step batch is
        # then a dict lookup instead of an H.264 random-access seek+decode.
        self._prepare_video_loaders(all_views)
        train_loader = self._create_train_loader(train_views)

        # Sanity render at iter 0 for debug PNGs
        dbg_batch = next(train_loader)
        dbg_view = dbg_batch[0]
        with torch.no_grad():
            out = self._render_view(gaussians, rasterizer, dbg_view)
            init_render = self._render_to_uint8(out["render"])
            init_gt = self._render_to_uint8(dbg_view["image"])
        cv2.imwrite("debug_init_render.png", init_render)
        cv2.imwrite("debug_init_gt.png", init_gt)
        if self.wandb_run is not None:
            try:
                import wandb
                self._wandb_log(
                    {
                        "init/render": wandb.Image(init_render[..., ::-1]),
                        "init/gt": wandb.Image(init_gt[..., ::-1]),
                    },
                    step=0,
                )
            except Exception as e:
                log(WARNING, f"wandb init image log failed: {e}")

        # SfM scene stats logged once at iter 0 (scene_extent already computed
        # at the top of train(); the merged cloud here is post-filter).
        if self.wandb_run is not None and len(merged_data["points_3d"]):
            try:
                # Pull the actual focal length used (in pixels) and 35mm
                # equivalent — this is the load-bearing intrinsic, and a wrong
                # value (heuristic vs. true iPhone 24mm) silently poisons every
                # downstream stage. Surfacing it in wandb means we can never
                # again train a 12h job and only realize after that focal was
                # wrong because an env var didn't propagate.
                K0 = np.asarray(merged_data["all_intrinsics"][0])
                f_px = float(K0[0, 0])
                # Recover image dimensions from the principal point — the
                # calibrator sets cx=W/2, cy=H/2 in identify_intrinsics.
                long_axis = max(2.0 * float(K0[0, 2]), 2.0 * float(K0[1, 2]))
                # 35mm-equivalent = (f_px / max(W,H)) * 36mm
                f_35mm_equiv = (f_px / long_axis) * 36.0 if long_axis > 0 else 0.0
                self._wandb_log(
                    {
                        "scene/extent": scene_extent,
                        "scene/n_outliers_dropped": n_outliers,
                        "scene/bbox_x": float(scene_bbox[0]),
                        "scene/bbox_y": float(scene_bbox[1]),
                        "scene/bbox_z": float(scene_bbox[2]),
                        "scene/median_depth": median_depth,
                        "scene/n_cameras": int(sum(len(p) for p in merged_data["all_poses"])),
                        "scene/n_points_kept": int(len(merged_data["points_3d"])),
                        "sfm/focal_px": f_px,
                        "sfm/focal_35mm_equiv": f_35mm_equiv,
                    },
                    step=0,
                )
                log(INFO, f"Active focal: {f_px:.1f} px (~{f_35mm_equiv:.1f}mm 35mm-equiv)")
            except Exception as e:
                log(WARNING, f"wandb scene stats log failed: {e}")

        # Training loop
        progress = tqdm(total=int(self.config.iterations_per_video), desc="Training")
        self._step_start_time = time.time()
        last_window_start = self._step_start_time

        while self.iteration < self.config.iterations_per_video:
            batch = next(train_loader)
            step_metrics = self._training_step(gaussians, rasterizer, batch, optimizer)

            with torch.no_grad():
                if gaussians.xyz.grad is not None:
                    gaussians.xyz_gradient_accum += torch.norm(
                        gaussians.xyz.grad, dim=-1, keepdim=True
                    )
                    gaussians.xyz_gradient_count += 1

            # Densify / prune
            if (
                self.iteration > 5000
                and self.iteration % self.config.densify_interval == 0
            ):
                stats = gaussians.densify_and_prune(
                    grads_threshold=self.config.densify_grads_threshold,
                    min_opacity=self.config.densify_min_opacity,
                    extent=scene_extent,
                    max_screen_size=self.config.densify_max_screen_size,
                    optimizer=optimizer,
                    clone_extent_ratio=self.config.densify_clone_extent_ratio,
                    prune_extent_ratio=self.config.densify_prune_extent_ratio,
                    max_gaussians=int(self.config.max_gaussians),
                )
                self._wandb_log(
                    {
                        "densify/cloned": stats["cloned"],
                        "densify/split": stats["split"],
                        "densify/pruned": stats["pruned"],
                        "densify/pruned_low_op": stats.get("pruned_low_op", 0),
                        "densify/pruned_too_big": stats.get("pruned_too_big", 0),
                        "densify/n_before": stats["n_before"],
                        "densify/n_after": stats["n_after"],
                        "densify/delta": stats["n_after"] - stats["n_before"],
                        "densify/cumulative_cloned": self._cum_cloned,
                        "densify/cumulative_split": self._cum_split,
                        "densify/cumulative_pruned": self._cum_pruned,
                        "densify/event_idx": self._densify_event_idx,
                        "densify/capped": int(stats.get("capped", False)),
                    }
                )
                self._cum_cloned += stats["cloned"]
                self._cum_split += stats["split"]
                self._cum_pruned += stats["pruned"]
                self._densify_event_idx += 1

            # Periodic opacity reset. Original 3DGS pattern: reset EVERY
            # gaussian's opacity to min(current, 0.01) so all of them have to
            # re-earn their place via gradient signal in the next ~1000 iters.
            # The ones that don't recover end up below `densify_min_opacity`
            # and get pruned at the next densify event.
            #
            # The previous version reset only opacity > 0.98 gaussians down to
            # 0.2 — which is 40× above the prune threshold of 0.005 — so the
            # mechanism never actually drove pruning. Result: population grew
            # unchecked because nothing was dying. See densify_and_prune for
            # the matching prune_mask fix.
            if self.iteration % self.config.opacity_reset_interval == 0 and self.iteration > 0:
                with torch.no_grad():
                    current = gaussians.get_opacity
                    target = torch.minimum(
                        current,
                        torch.full_like(current, 0.01),
                    )
                    # logit(0) is -inf; clamp away from 0 before inverse-sigmoid.
                    target = target.clamp(min=1e-6, max=1.0 - 1e-6)
                    gaussians.opacity.data = torch.logit(target)

            lr_xyz = self._update_learning_rate(optimizer)

            # Hard cap on gaussian scale: prevents the "blanket" failure mode
            # where a gaussian grows to scene_extent magnitude to cover error
            # it can't cover by repositioning (under-constrained scenes).
            with torch.no_grad():
                scale_ceiling = math.log(scene_extent * self.config.scale_clamp_ratio + 1e-9)
                gaussians.scaling.data.clamp_(max=scale_ceiling)

            # Per-step bookkeeping (cheap)
            if self.iteration % 10 == 0:
                progress.set_postfix(
                    {"loss": f"{step_metrics['loss']:.4f}", "n": f"{gaussians.xyz.shape[0]:,}"}
                )
                self.loss_history.append(step_metrics["loss"])
                self.gaussian_history.append(int(gaussians.xyz.shape[0]))
            if self.iteration % 100 == 0:
                self.opacity_history.append(float(gaussians.get_opacity.mean().item()))

            # W&B scalar logging
            if self.iteration % self.config.log_scalar_interval == 0:
                now = time.time()
                window = max(now - last_window_start, 1e-9)
                steps_per_sec = self.config.log_scalar_interval / window
                last_window_start = now

                # Per-group LRs (xyz is the only one updated by the schedule today,
                # but the others may evolve in the future).
                lrs = {f"train/lr_{g.get('name','grp')}": float(g["lr"]) for g in optimizer.param_groups}

                # Distribution percentiles of opacity & scale (snapshot — cheap)
                with torch.no_grad():
                    op = gaussians.get_opacity.detach().reshape(-1)
                    sc = gaussians.get_scaling.detach().reshape(-1)
                    qs = torch.tensor([0.5, 0.95, 0.99], device=op.device)
                    op_q = self._quantile_safe(op, qs)
                    sc_q = self._quantile_safe(sc, qs)
                    grad_norms = {}
                    for name, p in (
                        ("xyz", gaussians.xyz),
                        ("opacity", gaussians.opacity),
                        ("scaling", gaussians.scaling),
                        ("rotation", gaussians.rotation),
                        ("features_dc", gaussians.features_dc),
                    ):
                        if p.grad is not None:
                            grad_norms[f"grad/{name}_norm"] = float(p.grad.detach().norm().item())

                self._wandb_log(
                    {
                        "train/loss": step_metrics["loss"],
                        "train/l1": step_metrics["l1"],
                        "train/ssim": step_metrics["ssim"],
                        "train/psnr": step_metrics["psnr"],
                        "train/scale_reg": step_metrics["scale_reg"],
                        "train/lr_xyz": lr_xyz,
                        "train/sh_degree": self._active_sh_degree(),
                        **lrs,
                        "train/n_gaussians": int(gaussians.xyz.shape[0]),
                        "train/mean_opacity": float(op.mean().item()),
                        "train/mean_scale": float(sc.mean().item()),
                        "train/median_opacity": float(op_q[0].item()),
                        "train/p95_opacity": float(op_q[1].item()),
                        "train/p99_opacity": float(op_q[2].item()),
                        "train/median_scale": float(sc_q[0].item()),
                        "train/p95_scale": float(sc_q[1].item()),
                        "train/p99_scale": float(sc_q[2].item()),
                        "train/max_scale": float(sc.max().item()),
                        "train/steps_per_sec": steps_per_sec,
                        "train/wallclock_sec": time.time() - self._step_start_time,
                        **grad_norms,
                    }
                )

            if self.iteration % self.config.log_gpu_interval == 0:
                gpu = self._gpu_stats()
                if gpu:
                    self._wandb_log(gpu)

            if (
                self.config.log_hist_interval > 0
                and self.iteration > 0
                and self.iteration % self.config.log_hist_interval == 0
                and self.wandb_run is not None
            ):
                try:
                    import wandb
                    op = gaussians.get_opacity.detach().cpu().numpy().reshape(-1)
                    sc = gaussians.get_scaling.detach().cpu().numpy().reshape(-1)
                    self._wandb_log(
                        {
                            "dist/opacity": wandb.Histogram(op),
                            "dist/scale": wandb.Histogram(sc),
                        }
                    )
                except Exception as e:
                    log(WARNING, f"wandb histogram log failed: {e}")

            # Validation
            if (
                self.config.val_interval > 0
                and self.iteration > 0
                and self.iteration % self.config.val_interval == 0
            ):
                val_metrics = self._validate(gaussians, rasterizer, val_views)
                if val_metrics is not None:
                    log(
                        INFO,
                        f"[val @ {self.iteration}] l1={val_metrics['val/l1']:.4f} "
                        f"ssim={val_metrics['val/ssim']:.4f} psnr={val_metrics['val/psnr']:.2f}",
                    )
                    self._wandb_log(val_metrics)

            # Sample render to W&B
            if (
                self.config.log_image_interval > 0
                and self.iteration > 0
                and self.iteration % self.config.log_image_interval == 0
                and self.wandb_run is not None
            ):
                with torch.no_grad():
                    out = self._render_view(gaussians, rasterizer, dbg_view)
                    rendered = self._render_to_uint8(out["render"])
                    gt = self._render_to_uint8(dbg_view["image"])
                try:
                    import wandb
                    side = np.hstack([gt, rendered])
                    self._wandb_log(
                        {"render/train_sample": wandb.Image(side[..., ::-1])}
                    )
                except Exception as e:
                    log(WARNING, f"wandb image log failed: {e}")

            # Checkpoints
            if self.iteration > 0 and self.iteration % self.config.checkpoint_interval == 0:
                self._save_checkpoint(gaussians, optimizer, output_path)

            self.iteration += 1
            progress.update(1)

        progress.close()
        torch.save(gaussians.state_dict(), output_path / "final_model.pth")
        self.draw_graphs()

        # Final validation
        final_val = self._validate(gaussians, rasterizer, val_views)
        if final_val is not None:
            self._wandb_log({f"final/{k.replace('val/', '')}": v for k, v in final_val.items()})

        print(f"\nTraining complete! Model saved to {output_path}")
        return gaussians

    def draw_graphs(self):
        steps = range(0, len(self.loss_history) * 10, 10)
        opacity_steps = range(0, len(self.opacity_history) * 100, 100)

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        axes[0].plot(steps, self.loss_history)
        axes[0].set_xlabel("Iteration")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Training Loss")
        axes[0].grid(True)

        axes[1].plot(steps, self.gaussian_history)
        axes[1].set_xlabel("Iteration")
        axes[1].set_ylabel("Count")
        axes[1].set_title("Gaussian Count")
        axes[1].grid(True)

        opacity_values = [
            o.item() if torch.is_tensor(o) else o for o in self.opacity_history
        ]
        axes[2].plot(opacity_steps, opacity_values)
        axes[2].set_xlabel("Iteration")
        axes[2].set_ylabel("Mean Opacity")
        axes[2].set_title("Average Gaussian Opacity")
        axes[2].grid(True)

        plt.tight_layout()
        plt.savefig("training_metrics.png", dpi=150)
        print("Saved training graphs to training_metrics.png")
        plt.close()

    def _rgb_to_sh0(self, rgb):
        if rgb.max() > 1.01:
            rgb = rgb / 255.0
        C0 = 0.28209479177387814
        return (rgb - 0.5) / C0
