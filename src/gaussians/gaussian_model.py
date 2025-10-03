import torch
from torch import nn
from torch.nn import functional as F


class GaussianModel(nn.Module):
    """
    Gaussian model optimized for GPUs
    Initializes each gaussian as a quaternion
    """

    def __init__(self, n_gaussians: int = 1e5, device: str = "cuda"):
        super().__init__()

        device_c = torch.device(device)

        # Params - using half precision when possible
        self.xyz = nn.Parameter(torch.randn(n_gaussians, 3, device=device_c) * 10)
        self.features_dc = nn.Parameter(torch.zeros(n_gaussians, 1, 3, device=device_c))
        self.features_rest = nn.Parameter(
            torch.zeros(n_gaussians, 15, 3, device=device_c)
        )
        self.scaling = nn.Parameter(
            torch.ones(n_gaussians, 3, device=device_c) * -3
        )  # log scale
        self.rotation = nn.Parameter(
            torch.zeros(n_gaussians, 4, device=device_c)
        )  # quaternions
        self.opacity = nn.Parameter(torch.zeros(n_gaussians, 1, device=device_c))

        self.xyz_gradient_accum = torch.zeros_like(self.xyz)
        self.xyz_gradient_count = torch.zeros(n_gaussians, 1, device=device_c)
        self.max_radii_2D = torch.zeros(n_gaussians, device=device_c)

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
            grads = self.xyz_gradient_accum / (self.xyz_gradient_count + 1e-8)
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
            self.xyz_gradient_count.zero_()
            self.xyz_gradient_accum.zero_()

    def _clone_gaussians(self, mask):
        """
        Clones the masked gaussians
        :param mask: The given mask
        """
        new_xyz = self.xyz[mask]
        new_features_dc = self.features_dc[mask]
        new_features_rest = self.features_rest[mask]
        new_opacity = self.opacity[mask]
        new_scaling = self.scaling[mask]
        new_rotation = self.rotation[mask]

        self.densify(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

    def _split_gaussians(self, mask):
        """
        Splits large gaussians / high gradients into smaller ones
        :param mask: Mask for which to split
        """
        n_splits = 2

        # Get Gaussians to split
        xyz = self.xyz[mask]
        features_dc = self.features_dc[mask].repeat(n_splits, 1, 1)
        features_rest = self.features_rest[mask].repeat(n_splits, 1, 1)
        opacity = self.opacity[mask].repeat(n_splits, 1)
        scaling = self.scaling[mask].repeat(n_splits, 1)
        rotation = self.rotation[mask].repeat(n_splits, 1)

        # Reduces scaling and adds noise to positions to remove. 1.6 was used by others (plagiarized)
        scaling = scaling - torch.log(torch.tensor(1.6, device=scaling.device))

        scales = self.get_scaling[mask].repeat(n_splits, 1)
        # rotations = self.get_rotation[mask].repeat(n_splits, 1)

        # Samples by creating a new random normal tensor in the same shape repeated
        samples = torch.randn_like(xyz.repeat(n_splits, 1)) * scales[:, :3]
        xyz = xyz.repeat(n_splits, 1) + samples

        # Remove original
        self._prune_gaussians(~mask)
        self.densify(xyz, features_dc, features_rest, opacity, scaling, rotation)

    def _prune_gaussians(self, keep_mask):
        """
        Efficiently removes gaussians
        :param keep_mask: Mask for gaussians to keep
        """
        self.xyz = nn.Parameter(self.xyz[keep_mask])
        self.features_dc = nn.Parameter(self.features_dc[keep_mask])
        self.features_rest = nn.Parameter(self.features_rest[keep_mask])
        self.opacity = nn.Parameter(self.opacity[keep_mask])
        self.scaling = nn.Parameter(self.scaling[keep_mask])
        self.rotation = nn.Parameter(self.rotation[keep_mask])

        self.xyz_gradient_accum = self.xyz_gradient_accum[keep_mask]
        self.xyz_gradient_count = self.xyz_gradient_count[keep_mask]
        self.max_radii_2D = self.max_radii_2D[keep_mask]

    def densify(self, new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation):
        """
        Adds new gaussians efficiently from given values
        """
        # Concatenate parameters
        self.xyz = nn.Parameter(torch.cat([self.xyz, new_xyz]))
        self.features_dc = nn.Parameter(torch.cat([self.features_dc, new_features_dc]))
        self.features_rest = nn.Parameter(torch.cat([self.features_rest, new_features_rest]))
        self.opacity = nn.Parameter(torch.cat([self.opacity, new_opacity]))
        self.scaling = nn.Parameter(torch.cat([self.scaling, new_scaling]))
        self.rotation = nn.Parameter(torch.cat([self.rotation, new_rotation]))

        # update stats
        n_new = new_xyz.shape[0]
        device = new_xyz.device
        self.xyz_gradient_accum = torch.cat([
            self.xyz_gradient_accum,
            torch.zeros(n_new, 3, device=device)
        ])

        self.xyz_gradient_count = torch.cat([
            self.xyz_gradient_count,
            torch.zeros(n_new, 1, device=device)
        ])
        self.max_radii_2D = torch.cat([
            self.max_radii_2D,
            torch.zeros(n_new, device=device)
        ])
