from dataclasses import dataclass


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
    prune_interval: int = 1000
    opacity_reset_interval: int = 3000

    # Training config
    iterations_per_video: int = 3e4
    batch_size: int = 4
    lr: float = 0.001
    position_lr_init: float = 0.00016
    position_lr_final: float = 0.0000016
    position_lr_delay_mult: float = 0.01
    position_lr_max_steps: int = 3e4

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
