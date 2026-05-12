import torch
from torch import nn
from torch.nn import functional as F


_PARAM_ATTRS = ("xyz", "features_dc", "features_rest", "opacity", "scaling", "rotation")


def _replace_param_in_optimizer(model, attr, new_data, optimizer, *, keep_mask=None, append_data_for_state=None):
    """
    Replace `model.<attr>` (an nn.Parameter) with a new Parameter wrapping `new_data`,
    and — if `optimizer` is provided — carry the Adam moment buffers across to the
    new parameter so per-parameter momentum is preserved through densify/prune.

    Exactly one of `keep_mask` or `append_data_for_state` must be supplied, and
    `new_data` must already match that operation:
      * keep_mask           : `new_data == old[keep_mask]`        (for prune)
      * append_data_for_state: `new_data == cat([old, append])`   (for densify/clone)
    """
    old = getattr(model, attr)
    new_param = nn.Parameter(new_data)
    setattr(model, attr, new_param)

    if optimizer is None:
        return new_param

    for group in optimizer.param_groups:
        for i, p in enumerate(group["params"]):
            if p is old:
                group["params"][i] = new_param
                state = optimizer.state.pop(old, None)
                if state is not None:
                    new_state = {k: v for k, v in state.items()}
                    for sk in ("exp_avg", "exp_avg_sq"):
                        buf = state.get(sk)
                        if buf is None:
                            continue
                        if keep_mask is not None:
                            new_state[sk] = buf[keep_mask]
                        elif append_data_for_state is not None:
                            n_new = append_data_for_state.shape[0]
                            zeros = torch.zeros(
                                (n_new, *buf.shape[1:]),
                                dtype=buf.dtype,
                                device=buf.device,
                            )
                            new_state[sk] = torch.cat([buf, zeros], dim=0)
                    optimizer.state[new_param] = new_state
                return new_param
    return new_param


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
        optimizer=None,
        clone_extent_ratio: float = 0.1,
        prune_extent_ratio: float = 2.0,
        max_gaussians: int = None,
    ):
        """
        Densify (clone + split) and prune gaussians in a single sweep.

        If `optimizer` is provided, Adam moment buffers (`exp_avg`, `exp_avg_sq`)
        are spliced through every parameter-tensor change, so per-parameter
        momentum is preserved across densify events.

        `clone_extent_ratio` and `prune_extent_ratio` scale the densify
        thresholds against the scene extent:
          * clone candidates have `max_scale <= extent * clone_extent_ratio`
          * "too large" prune candidates have `max_scale > extent * prune_extent_ratio`

        `max_gaussians` is a hard population cap. If the current count is at or
        above the cap, no clone/split happens (prune still does). If the
        projected post-densify count exceeds the cap, only the highest-gradient
        clone+split candidates are kept — those benefit most from densification.
        """
        stats = {
            "cloned": 0,
            "split": 0,
            "pruned": 0,
            "n_before": int(self.xyz.shape[0]),
            "capped": False,
        }
        n_before = stats["n_before"]

        with torch.no_grad():
            grads = self.xyz_gradient_accum / (self.xyz_gradient_count + 1e-8)
            grads_norm = torch.norm(grads, dim=-1)
            scales = self.get_scaling
            max_scale = scales.amax(dim=-1)
            opacity = self.get_opacity.squeeze()

            clone_mask = (
                (grads_norm >= grads_threshold)
                & (max_scale <= extent * clone_extent_ratio)
                & (opacity > min_opacity)
            )
            split_mask = (
                (grads_norm >= grads_threshold)
                & (max_scale > extent * clone_extent_ratio)
                & (opacity > min_opacity)
            )

            # Enforce population cap. Each clone adds 1; each split removes the
            # original and adds 2 (net +1). So projected delta = n_clone + n_split.
            if max_gaussians is not None and n_before >= int(max_gaussians):
                clone_mask = torch.zeros_like(clone_mask)
                split_mask = torch.zeros_like(split_mask)
                stats["capped"] = True
            elif max_gaussians is not None:
                n_clone = int(clone_mask.sum().item())
                n_split = int(split_mask.sum().item())
                n_allowed = int(max_gaussians) - n_before
                if n_clone + n_split > n_allowed:
                    stats["capped"] = True
                    combined = clone_mask | split_mask
                    keep = self._top_k_mask_by_grad(combined, grads_norm, n_allowed)
                    clone_mask = clone_mask & keep
                    split_mask = split_mask & keep

            if clone_mask.sum() > 0:
                stats["cloned"] = int(clone_mask.sum().item())
                self._clone_gaussians(clone_mask, optimizer=optimizer)

            # Recompute scale/opacity for split (clone grew the tensors). The
            # split_mask was built against the pre-clone shape, so we need to
            # zero-pad it to the new size.
            new_n = self.xyz.shape[0]
            if split_mask.shape[0] < new_n:
                pad = torch.zeros(
                    new_n - split_mask.shape[0], dtype=split_mask.dtype, device=split_mask.device
                )
                split_mask = torch.cat([split_mask, pad])

            if split_mask.sum() > 0:
                stats["split"] = int(split_mask.sum().item())
                self._split_gaussians(split_mask, optimizer=optimizer)

            grads = self.xyz_gradient_accum / (self.xyz_gradient_count + 1e-8)
            grads_norm = torch.norm(grads, dim=-1)
            scales = self.get_scaling
            max_scale = scales.amax(dim=-1)
            opacity = self.get_opacity.squeeze()

            # Prune if EITHER weak (low opacity) OR oversized. Original 3DGS
            # pattern. The previous `(low_op & low_grad) | too_big` was wrong:
            # a high-grad + low-opacity gaussian is exactly what should die
            # (rasterizer keeps using it but it isn't contributing), but the
            # AND clause kept it alive. Combined with opacity-reset that didn't
            # actually drive anyone toward the prune threshold, pruning was
            # firing ~125× less often than cloning and population exploded.
            low_op = opacity < min_opacity
            too_big = max_scale > extent * prune_extent_ratio
            prune_mask = low_op | too_big

            if prune_mask.sum() > 0:
                stats["pruned"] = int(prune_mask.sum().item())
                stats["pruned_low_op"] = int(low_op.sum().item())
                stats["pruned_too_big"] = int(too_big.sum().item())
                self._prune_gaussians(~prune_mask, optimizer=optimizer)
            else:
                stats["pruned_low_op"] = 0
                stats["pruned_too_big"] = 0

            # Reset gradient accumulators (sized to current N).
            self.xyz_gradient_count = torch.zeros_like(self.xyz_gradient_count)
            self.xyz_gradient_accum = torch.zeros_like(self.xyz_gradient_accum)

        stats["n_after"] = int(self.xyz.shape[0])
        return stats

    @staticmethod
    def _top_k_mask_by_grad(mask: torch.Tensor, grads_norm: torch.Tensor, k: int) -> torch.Tensor:
        """
        Reduce a boolean mask to the k highest-gradient entries among those
        currently True. Used to enforce max_gaussians by keeping only the most
        impactful densify candidates when the population cap would be exceeded.
        """
        if k <= 0:
            return torch.zeros_like(mask)
        if int(mask.sum().item()) <= k:
            return mask
        idx = mask.nonzero(as_tuple=False).view(-1)
        top_k = torch.topk(grads_norm[idx], k=k).indices
        chosen = idx[top_k]
        out = torch.zeros_like(mask)
        out[chosen] = True
        return out

    def _clone_gaussians(self, mask, optimizer=None):
        """
        Clone gaussians selected by `mask` (bool or long-index tensor) — append
        copies of their parameter rows. Adam state is spliced if `optimizer` is
        passed.
        """
        if mask.dtype == torch.bool:
            idx = mask.nonzero(as_tuple=False).view(-1)
        else:
            idx = mask
        if idx.numel() == 0:
            return

        new_xyz = self.xyz[idx]
        new_features_dc = self.features_dc[idx]
        new_features_rest = self.features_rest[idx]
        new_opacity = self.opacity[idx]
        new_scaling = self.scaling[idx]
        new_rotation = self.rotation[idx]

        self.densify(
            new_xyz, new_features_dc, new_features_rest,
            new_opacity, new_scaling, new_rotation,
            optimizer=optimizer,
        )

    def _split_gaussians(self, mask, optimizer=None):
        """
        Split large/high-gradient gaussians into `n_splits` (=2) smaller ones,
        positioned with isotropic Gaussian noise around the original. The
        original is pruned first, then the children are appended. Adam state is
        spliced if `optimizer` is passed.
        """
        if mask.dtype == torch.bool:
            idx = mask.nonzero(as_tuple=False).view(-1)
        else:
            idx = mask
        if idx.numel() == 0:
            return

        n_splits = 2
        xyz = self.xyz[idx]
        features_dc = self.features_dc[idx].repeat(n_splits, 1, 1)
        features_rest = self.features_rest[idx].repeat(n_splits, 1, 1)
        opacity = self.opacity[idx].repeat(n_splits, 1)
        scaling = self.scaling[idx].repeat(n_splits, 1)
        rotation = self.rotation[idx].repeat(n_splits, 1)

        # 1.6 is from the original 3DGS paper.
        scaling = scaling - torch.log(torch.tensor(1.6, device=scaling.device))
        scales = torch.exp(scaling)
        samples = torch.randn_like(xyz.repeat(n_splits, 1)) * scales[:, :3]
        new_xyz = xyz.repeat(n_splits, 1) + samples

        keep_mask = torch.ones(self.xyz.shape[0], dtype=torch.bool, device=self.xyz.device)
        keep_mask[idx] = False
        self._prune_gaussians(keep_mask, optimizer=optimizer)
        self.densify(
            new_xyz, features_dc, features_rest,
            opacity, scaling, rotation,
            optimizer=optimizer,
        )

    def _prune_gaussians(self, keep_mask, optimizer=None):
        """
        Subset every parameter tensor by `keep_mask`. Adam state is spliced if
        `optimizer` is passed. Buffers (gradient accumulators, max radii) are
        always sliced — they are not optimizer parameters.
        """
        for attr in _PARAM_ATTRS:
            old = getattr(self, attr)
            _replace_param_in_optimizer(
                self, attr, old.detach()[keep_mask], optimizer, keep_mask=keep_mask
            )

        self.xyz_gradient_accum = self.xyz_gradient_accum[keep_mask]
        self.xyz_gradient_count = self.xyz_gradient_count[keep_mask]
        self.max_radii_2D = self.max_radii_2D[keep_mask]

    def densify(
        self,
        new_xyz, new_features_dc, new_features_rest,
        new_opacity, new_scaling, new_rotation,
        optimizer=None,
    ):
        """
        Append new gaussians. Adam state is spliced (zero-initialized rows) if
        `optimizer` is passed.
        """
        appends = {
            "xyz": new_xyz,
            "features_dc": new_features_dc,
            "features_rest": new_features_rest,
            "opacity": new_opacity,
            "scaling": new_scaling,
            "rotation": new_rotation,
        }
        for attr, append_data in appends.items():
            old = getattr(self, attr)
            new_data = torch.cat([old.detach(), append_data], dim=0)
            _replace_param_in_optimizer(
                self, attr, new_data, optimizer,
                append_data_for_state=append_data,
            )

        n_new = new_xyz.shape[0]
        device = new_xyz.device
        self.xyz_gradient_accum = torch.cat(
            [self.xyz_gradient_accum, torch.zeros(n_new, 3, device=device)]
        )
        self.xyz_gradient_count = torch.cat(
            [self.xyz_gradient_count, torch.zeros(n_new, 1, device=device)]
        )
        self.max_radii_2D = torch.cat(
            [self.max_radii_2D, torch.zeros(n_new, device=device)]
        )
