from src.gaussians.training_config import TrainingConfig


class GaussianTrainer:
    def __init__(self, config: TrainingConfig, device):
        self.config = config
        self.device = device

    def train(self, merged_data, output_dir):
        pass