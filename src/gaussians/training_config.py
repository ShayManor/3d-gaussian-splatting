from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class TrainingConfig:
    """
    Config for 3DGS training
    """

    # Video Processing
    frame_stride: int = 10  # sampling
    image_scale: float = 1.0
    cache_dir: str = "./cache"
    matcher: str = "opencv"  # opencv or loftr

    # Gaussians
    initial_gaussians: int = 1e5
    max_gaussians: int = 5e6
    densify_interval: int = 100
    prune_interval: int = 2000
    opacity_reset_interval: int = 5000

    # Densify thresholds (multipliers against scene_extent)
    densify_grads_threshold: float = 5e-4
    densify_min_opacity: float = 0.005
    densify_max_screen_size: float = 12.0
    # max_scale <= extent * clone_extent_ratio -> clone candidate
    densify_clone_extent_ratio: float = 0.1
    # max_scale  > extent * prune_extent_ratio -> "too big" prune candidate
    densify_prune_extent_ratio: float = 2.0
    # Per-step hard cap on gaussian scale, expressed as a fraction of
    # scene_extent. After every optimizer.step() the scaling param is
    # clamped to log(scene_extent * scale_clamp_ratio) so individual
    # gaussians can't grow huge to "blanket" reconstruction error.
    scale_clamp_ratio: float = 0.2
    # Splatfacto-style anisotropy penalty: hinge loss on max/min scale ratio.
    # Loss = scale_reg_weight * mean(max(ratio, scale_reg_max_ratio) - max_ratio).
    # Zero contribution when all gaussians are within ratio; grows linearly above.
    scale_reg_max_ratio: float = 10.0
    scale_reg_weight: float = 0.1

    # Training config
    iterations_per_video: int = 3e5
    batch_size: int = 4
    lr: float = 0.001
    position_lr_init: float = 1.6e-4
    position_lr_final: float = 1.6e-7
    position_lr_delay_mult: float = 0.01
    position_lr_max_steps: int = 3e5

    # GPU Optimizations
    use_mixed_precision: bool = True
    gradient_accumulation: int = 4
    num_workers: int = 8
    prefetch_frames: int = 100

    # Multi-GPU
    distributed: bool = False
    world_size: int = 1

    # Losses
    lambda_dssim: float = 0.2
    lambda_depth: float = 0.1
    lambda_normal: float = 0.05

    # Logging / validation
    log_scalar_interval: int = 10       # log loss/lr/n_gauss every N iters
    log_gpu_interval: int = 200         # log gpu memory every N iters
    log_image_interval: int = 2000      # log a sample render every N iters
    log_hist_interval: int = 5000       # log opacity/scale histograms every N iters
    val_interval: int = 1000            # run validation every N iters
    val_fraction: float = 0.1           # held-out view fraction (capped by val_max_views)
    val_max_views: int = 16
    val_seed: int = 42                  # deterministic train/val split
    checkpoint_interval: int = 10000

    # W&B
    wandb_project: str = "3d-gaussian-splatting"
    wandb_entity: Optional[str] = None
    wandb_mode: str = "online"          # online | offline | disabled
    wandb_run_name: Optional[str] = None
    wandb_tags: Optional[List[str]] = None
