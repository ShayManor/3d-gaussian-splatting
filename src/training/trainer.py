from logging import log, WARNING, INFO
from pathlib import Path
from typing import List, Dict

import cv2
import numpy as np
import torch
from torch import GradScaler, autocast
from torch.optim import Adam
from tqdm import tqdm

from src.gaussians.gaussian_model import GaussianModel
from src.gaussians.gaussian_rasterizer import GaussianRasterizer
from src.gaussians.training_config import TrainingConfig
from src.video.video_loader import VideoLoader

from sklearn.neighbors import NearestNeighbors
from torch.nn import functional as F
import matplotlib.pyplot as plt


class GaussianTrainer:
    def __init__(self, config: TrainingConfig, device):
        self.config = config
        self.device = device

        self.scaler = GradScaler() if config.use_mixed_precision else None

        self.iteration = 0
        self.loss_history = []
        self.opacity_history = []
        self.gaussian_history = []

    def _initialize_gaussians(self, merged_data):
        """
        Initialize Gaussians from SFM point cloud
        :param merged_data: Point cloud
        :return: Gaussians
        """
        points_3d = merged_data["points_3d"]
        colors = merged_data["colors"]
        n = len(points_3d)

        if n == 0:
            log(WARNING, "No 3D points found! Initializing random gaussians")
            return GaussianModel(
                n_gaussians=int(self.config.initial_gaussians), device=str(self.device)
            )

        # Calculate initial number of Gaussians
        n_gaussians = min(
            max(n * 3, int(self.config.initial_gaussians)),  # tt least 3x points
            int(self.config.max_gaussians // 2),  # Leave room for densification
        )

        print(f"Creating {n_gaussians:,} initial Gaussians from {n:,} 3D points")

        gaussians = GaussianModel(n_gaussians=n_gaussians, device=str(self.device))

        with torch.no_grad():
            # convert to tensor
            points_tensor = torch.tensor(
                points_3d, device=self.device, dtype=torch.float32
            )

            if n_gaussians <= n:
                # subsample points
                indices = torch.randperm(n)[:n_gaussians]
            else:
                # duplicate points with noise
                indices = torch.randint(0, n, (n_gaussians,))

            # set positions with small random offset
            gaussians.xyz.data = (
                points_tensor[indices]
                + torch.randn(n_gaussians, 3, device=self.device) * 0.001
            )

            # Initialize colors if available
            if len(colors) > 0:
                colors_tensor = torch.tensor(
                    colors, device=self.device, dtype=torch.float32
                )
                # convert to SH
                sh_colors = self._rgb_to_sh0(colors_tensor[indices])
                gaussians.features_dc.data[:, 0, :] = sh_colors

            # Smart scale initialization based on nearest neighbors
            self._initialize_scales_smart(gaussians, points_tensor)

            # Initialize opacity to be slightly visible
            gaussians.opacity.data = torch.logit(
                torch.ones(n_gaussians, 1, device=self.device) * 0.01
            )

        return gaussians

    def _initialize_scales_smart(self, gaussians, points):
        """
        Initialize scales based on local point density
        """
        positions = gaussians.xyz.data.cpu().numpy()
        all_points = points.cpu().numpy()

        nbrs = NearestNeighbors(
            n_neighbors=min(7, len(all_points)), algorithm="kd_tree"
        )
        nbrs.fit(all_points)
        distances, _ = nbrs.kneighbors(positions)

        # Average distance as scale
        avg_distances = (
            distances[:, 1:].mean(axis=1) if distances.shape[1] > 1 else distances[:, 0]
        )

        scales = torch.tensor(avg_distances, device=self.device, dtype=torch.float32)
        scales = scales.clamp(min=1e-7)
        scales = torch.log(scales.unsqueeze(1).expand(-1, 3))

        gaussians.scaling.data = scales

    def _setup_optimizer(self, gaussians):
        params = [
            {
                "params": [gaussians.xyz],
                "lr": self.config.position_lr_init,
                "name": "xyz",
            },
            {"params": [gaussians.features_dc], "lr": 0.0025, "name": "f_dc"},
            {
                "params": [gaussians.features_rest],
                "lr": 0.0025 / 20.0,
                "name": "f_rest",
            },
            {"params": [gaussians.opacity], "lr": 0.05, "name": "opacity"},
            {"params": [gaussians.scaling], "lr": 0.005, "name": "scaling"},
            {"params": [gaussians.rotation], "lr": 0.001, "name": "rotation"},
        ]
        return Adam(params, lr=0.0, eps=1e-15)

    def _create_data_loader(self, merged_data):
        """
        Efficient data loader
        """
        all_views = []
        for i, video_info in enumerate(merged_data["video_info"]):
            video_path = video_info["path"]
            poses = merged_data["all_poses"][i]
            K = merged_data["all_intrinsics"][i]

            loader = VideoLoader(video_path, cache_frames=False)
            width = int(loader.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(loader.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Sample frames with stride
            total_frames = min(loader.total_frames, len(poses))
            frame_indices = list(range(0, total_frames, self.config.frame_stride))

            for j, frame_idx in enumerate(frame_indices[: len(poses)]):
                all_views.append(
                    {
                        "video_path": video_path,
                        "frame_idx": frame_idx,
                        "pose": poses[j],
                        "K": K,
                        "width": width,
                        "height": height,
                    }
                )

        log(INFO, f"Total training views: {len(all_views)}")

        def batch_generator():
            loaders = {}
            while True:
                # random sample
                indices = np.random.choice(
                    len(all_views), self.config.batch_size, replace=True
                )
                batch = []

                for idx in indices:
                    view = all_views[idx]
                    # get or create loader
                    if view["video_path"] not in loaders:
                        loaders[view["video_path"]] = VideoLoader(
                            view["video_path"], cache_frames=False
                        )

                    frame = loaders[view["video_path"]].get_frame(view["frame_idx"])
                    if frame is not None:
                        batch.append(
                            {
                                "image": torch.tensor(
                                    frame / 255.0,
                                    device=self.device,
                                    dtype=torch.float32,
                                ),
                                "pose": torch.tensor(
                                    view["pose"],
                                    device=self.device,
                                    dtype=torch.float32,
                                ),
                                "K": torch.tensor(
                                    view["K"], device=self.device, dtype=torch.float32
                                ),
                                "width": view["width"],
                                "height": view["height"],
                            }
                        )

                if batch:
                    yield batch

        return batch_generator()

    def _get_projection_matrix(
        self, K: torch.Tensor, width: int, height: int
    ) -> torch.Tensor:
        """
        Simple projection matrix from intrinsics
        """
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
        """
        Simple SSIM computation
        """
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
        """
        Update learning rate schedule
        """
        progress = min(self.iteration / self.config.position_lr_max_steps, 1.0)
        lr = (
            self.config.position_lr_init
            * (self.config.position_lr_final / self.config.position_lr_init) ** progress
        )

        for group in optimizer.param_groups:
            if group.get("name") == "xyz":
                group["lr"] = lr

    def _save_checkpoint(
        self, gaussians: GaussianModel, optimizer: Adam, output_path: Path
    ):
        """
        Save training checkpoint
        """
        torch.save(
            {
                "iteration": self.iteration,
                "model_state": gaussians.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "n_gaussians": gaussians.xyz.shape[0],
            },
            output_path / f"checkpoint_{self.iteration}.pth",
        )

    def _training_step(
        self,
        gaussians: GaussianModel,
        rasterizer: GaussianRasterizer,
        batch: List[Dict],
        optimizer: Adam,
    ) -> float:
        """
        Single training step. Returns loss
        """
        optimizer.zero_grad()

        gaussian_params = {
            "means3D": gaussians.get_xyz,
            "scales": gaussians.get_scaling,
            "rotations": gaussians.get_rotation,
            "opacities": gaussians.get_opacity,
            "shs": gaussians.get_features,
        }

        total_loss = torch.tensor(0.0, device=self.device)

        for view_data in batch:
            # Create viewpoint
            viewpoint = {
                "world_view_transform": torch.inverse(view_data["pose"]),
                "projection_matrix": self._get_projection_matrix(
                    view_data["K"], view_data["width"], view_data["height"]
                ),
                "image_width": view_data["width"],
                "image_height": view_data["height"],
            }

            # Render with backend
            with autocast(
                enabled=self.config.use_mixed_precision, device_type=self.device
            ):
                rendered = rasterizer.backend.render_with_depth(
                    gaussian_params,
                    viewpoint,
                    bg_color=torch.zeros(3, device=self.device),
                    render_mode="RGB",
                    device=str(self.device),
                )

            rendered_img = rendered["render"]  # [H, W, 3]
            if rendered_img.shape[0] == 3:  # If [C, H, W]
                rendered_img = rendered_img.permute(1, 2, 0)  # Convert to [H, W, C]
            gt_image = view_data["image"]  # [H, W, 3]
            # L1 + SSIM loss
            l1_loss = F.l1_loss(rendered_img, gt_image)
            ssim_loss = 1.0 - self._compute_ssim(
                rendered_img.permute(2, 0, 1).unsqueeze(0),  # Convert [H, W, 3] to [1, 3, H, W]
                gt_image.permute(2, 0, 1).unsqueeze(0),  # Convert [H, W, 3] to [1, 3, H, W]
            )

            loss = (
                1.0 - self.config.lambda_dssim
            ) * l1_loss + self.config.lambda_dssim * ssim_loss

            total_loss += loss

        total_loss = total_loss / len(batch)

        # Backward
        if self.scaler:
            self.scaler.scale(total_loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            total_loss.backward()
            optimizer.step()

        return total_loss.item()

    def train(self, merged_data, output_dir) -> GaussianModel:
        """
        Main training loop
        :param merged_data: Dict with total data
        :param output_dir: output location for model
        :return: GaussianModel: trained model
        """

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)

        print(f"Initializing with {len(merged_data['points_3d'])} 3D points")

        # initialize gaussians from point cloud
        gaussians = self._initialize_gaussians(merged_data)
        print(f"Initialized {gaussians.xyz.shape[0]} Gaussians")

        optimizer = self._setup_optimizer(gaussians)

        # Setup rasterizer with first camera's intrinsics
        K = torch.tensor(
            merged_data["all_intrinsics"][0], device=self.device, dtype=torch.float32
        )
        rasterizer = GaussianRasterizer(
            K=K, device=str(self.device), enable_caching=True, backend="gsplat"
        )
        train_loader = self._create_data_loader(merged_data)

        # training loop
        progress = tqdm(total=int(self.config.iterations_per_video), desc="Training")

        while self.iteration < self.config.iterations_per_video:
            batch = next(train_loader)

            # compute loss and backward
            loss = self._training_step(gaussians, rasterizer, batch, optimizer)

            # Accumulate gradients for densification
            with torch.no_grad():
                if gaussians.xyz.grad is not None:
                    gaussians.xyz_gradient_accum += gaussians.xyz.grad.data
                    gaussians.xyz_gradient_count += 1

            if (
                self.iteration > 500
                and self.iteration % self.config.densify_interval == 0
            ):
                gaussians.densify_and_prune(
                    grads_threshold=0.0002,
                    min_opacity=0.005,
                    extent=4.0,
                    max_screen_size=20.0,
                )
                gaussians.xyz_gradient_accum.zero_()
                gaussians.xyz_gradient_count.zero_()
                optimizer = self._setup_optimizer(gaussians)

            # Reset opacity periodically
            if self.iteration % self.config.opacity_reset_interval == 0:
                with torch.no_grad():
                    mask = gaussians.get_opacity < 0.01
                    gaussians.opacity.data[mask] = torch.logit(
                        torch.ones_like(gaussians.opacity.data[mask]) * 0.01
                    )

            self._update_learning_rate(optimizer)

            if self.iteration % 10 == 0:
                progress.set_postfix(
                    {"loss": f"{loss:.4f}", "n_gauss": f"{gaussians.xyz.shape[0]:,}"}
                )
                self.loss_history.append(loss)
                self.gaussian_history.append(gaussians.xyz.shape[0])

                # Checkpoint
            if self.iteration % 1000 == 0:
                self._save_checkpoint(gaussians, optimizer, output_path)
                self.opacity_history.append(gaussians.opacity.mean().item())

            self.iteration += 1
            progress.update(1)

        progress.close()
        torch.save(gaussians.state_dict(), output_path / "final_model.pth")
        self.draw_graphs()
        print(f"\nTraining complete! Model saved to {output_path}")

        return gaussians

    def draw_graphs(self):
        steps = range(0, len(self.loss_history) * 10, 10)
        opacity_steps = range(0, len(self.opacity_history) * 1000, 1000)

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # Loss
        axes[0].plot(steps, self.loss_history)
        axes[0].set_xlabel('Iteration')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training Loss')
        axes[0].grid(True)

        # Gaussian count
        axes[1].plot(steps, self.gaussian_history)
        axes[1].set_xlabel('Iteration')
        axes[1].set_ylabel('Count')
        axes[1].set_title('Gaussian Count')
        axes[1].grid(True)

        # Opacity mean
        opacity_values = [o.item() if torch.is_tensor(o) else o for o in self.opacity_history]
        axes[2].plot(opacity_steps, opacity_values)
        axes[2].set_xlabel('Iteration')
        axes[2].set_ylabel('Mean Opacity')
        axes[2].set_title('Average Gaussian Opacity')
        axes[2].grid(True)

        plt.tight_layout()
        plt.savefig('training_metrics.png', dpi=150)
        print(f"Saved training graphs to training_metrics.png")
        plt.close()

    def _rgb_to_sh0(self, rgb):
        # Normalize RGB
        if rgb.max() > 1.01:
            rgb = rgb / 255.0

        # Calculated C0 coefficient = sqrt(1/4pi)
        C0 = 0.28209479177387814
        return (rgb - 0.5) / C0
