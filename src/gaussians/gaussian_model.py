import torch
from torch import nn
from torch.nn import functional as F


class GaussianModel(nn.Module):
    """
    Gaussian model optimized for GPUs
    Initializes each gaussian as a quaternion
    """

    def __init__(self, n_gaussians: int = 1e5, device_s: str = "cuda"):
        super().__init__()

        device = torch.device(device_s)

        # Params - using half precision when possible
        self.xyz = nn.Parameter(torch.randn(n_gaussians, 3, device=device) * 10)
        self.features_dc = nn.Parameter(torch.zeros(n_gaussians, 1, 3, device=device))
        self.features_rest = nn.Parameter(
            torch.zeros(n_gaussians, 15, 3, device=device)
        )
        self.scaling = nn.Parameter(
            torch.ones(n_gaussians, 3, device=device) * -3
        )  # log scale
        self.rotation = nn.Parameter(
            torch.zeros(n_gaussians, 4, device=device)
        )  # quaternions
        self.opacity = nn.Parameter(torch.zeros(n_gaussians, 1, device=device))

        self.xyz_gradient_accum = torch.zeros_like(self.xyz)
        self.xyz_gradient_counts = torch.zeros(n_gaussians, 1, device=device)
        self.max_radii_2D = torch.zeros(n_gaussians, device=device)

        # Initialize quaternions
        with torch.no_grad():
            self.rotation[:, 0] = 1  # w component

        self.scaling_activation = torch.exp
        self.opacity_activation = torch.sigmoid
        self.rotation_activation = F.normalize

    @property
    def get_xyz(self):
        return self.xyz

    @property
    def get_scaling(self):
        return self.scaling_activation(self.scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self.rotation, dim=-1)

    @property
    def get_opacity(self):
        return self.opacity_activation(self.opacity)

    @property
    def get_features(self):
        return torch.cat([self.features_dc, self.features_rest], dim=1)

    def densify_and_prune(
        self,
        grads_threshold: float,
        min_opacity: float,
        extent: float,
        max_screen_size: float,
    ):
        """
        Densifies and prunes low-opacity gaussians
        """
        with torch.no_grad():
            grads = self.xyz_gradient_accum / (self.xyz_gradient_counts + 1e-8)
            grads_norm = torch.norm(grads, dim=-1)

            # Clone small gaussians with high gradients
            scales = self.get_scaling
            max_scale = scales.amax(dim=-1)

            clone_mask = (grads_norm >= grads_threshold) & (max_scale <= extent * 0.1)

            # Split large gaussians (high gradients and large)
            split_mask = (grads_norm >= grads_threshold) & (max_scale > extent * 0.1)

            # Prune low opacity gaussians
            prune_mask = (self.get_opacity < min_opacity).squeeze()
            big_points_mask = max_scale > extent / 2
            prune_mask = prune_mask | big_points_mask

            if clone_mask.sum() > 0:
                self._clone_gaussians(clone_mask)

            if split_mask.sum() > 0:
                self._split_gaussians(split_mask)

            if prune_mask.sum() > 0:
                self._prune_gaussians(~prune_mask)  # not operator because prune removes

            # Reset gradients
            self.xyz_gradient_counts.zero_()
            self.xyz_gradient_accum.zero_()
