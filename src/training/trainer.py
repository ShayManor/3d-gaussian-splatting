from pathlib import Path

import torch
from torch import GradScaler
from tqdm import tqdm

from src.gaussians.gaussian_model import GaussianModel
from src.gaussians.gaussian_rasterizer import GaussianRasterizer
from src.gaussians.training_config import TrainingConfig


class GaussianTrainer:
    def __init__(self, config: TrainingConfig, device):
        self.config = config
        self.device = device

        self.scaler = GradScaler() if config.use_mixed_precision else None

        self.iteration = 0
        self.loss_history = []

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
            merged_data['all_intrinsics'][0],
            device=self.device,
            dtype=torch.float32
        )
        rasterizer = GaussianRasterizer(
            K=K,
            device=str(self.device),
            enable_caching=True,
            backend="gsplat"
        )
        train_loader = self._create_data_loader(merged_data)

        # training loop
        progress = tqdm(total=int(self.config.iterations_per_video), desc="Training")

        while self.iteration < self.config.iterations_per_video:
            self.iteration += 1

        return GaussianModel()

    def _initialize_gaussians(self, merged_data):
        pass

    def _setup_optimizer(self, gaussians):
        pass

    def _create_data_loader(self, merged_data):
        pass
