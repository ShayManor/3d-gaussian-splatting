from logging import log, WARNING
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
        """
        Initialize Gaussians from SFM point cloud
        :param merged_data: Point cloud
        :return: Gaussians
        """
        points_3d = merged_data['points_3d']
        colors = merged_data['colors']
        n = len(points_3d)

        if n == 0:
            log(WARNING, "No 3D points found! Initializing random gaussians")
            return GaussianModel(
                n_gaussians=int(self.config.initial_gaussians),
                device=str(self.device)
            )

        # Calculate initial number of Gaussians
        n_gaussians = min(
            max(n * 3, int(self.config.initial_gaussians)),  # tt least 3x points
            int(self.config.max_gaussians // 2)  # Leave room for densification
        )

        print(f"Creating {n_gaussians:,} initial Gaussians from {n:,} 3D points")

        gaussians = GaussianModel(
            n_gaussians=n_gaussians,
            device=str(self.device)
        )

        with torch.no_grad():
            # convert to tensor
            points_tensor = torch.tensor(points_3d, device=self.device, dtype=torch.float32)

            if n_gaussians <= n:
                # subsample points
                indices = torch.randperm(n)[:n_gaussians]
            else:
                # duplicate points with noise
                indices = torch.randint(0, n, (n_gaussians,))

            # set positions with small random offset
            gaussians.xyz.data = points_tensor[indices] + torch.randn(n_gaussians, 3, device=self.device) * 0.001

            # Initialize colors if available
            if len(colors) > 0:
                colors_tensor = torch.tensor(colors, device=self.device, dtype=torch.float32)
                gaussians.features_dc.data[:, 0, :] = colors_tensor[indices]

            # Smart scale initialization based on nearest neighbors
            self._initialize_scales_smart(gaussians, points_tensor)

            # Initialize opacity to be slightly visible
            gaussians.opacity.data = torch.logit(
                torch.ones(n_gaussians, 1, device=self.device) * 0.1
            )

        return gaussians

    def _initialize_scales_smart(self, gaussians, points):
        """
        Initialize scales based on local point density
        """


    def _setup_optimizer(self, gaussians):
        pass

    def _create_data_loader(self, merged_data):
        pass
