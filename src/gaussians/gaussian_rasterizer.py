from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from logging import log, INFO
from typing import List, Dict, Optional

import torch

from src.backends.gsplat_backend import GSplatBackend
from src.backends.pytorch_backend import PyTorchBackend


@dataclass
class RenderCache:
    means2d: torch.Tensor
    depths: torch.Tensor
    radii: torch.Tensor
    conics: torch.Tensor
    tiles_touched: torch.Tensor
    frame_idx: int
    view_matrix: torch.Tensor
    rendered_image: torch.Tensor
    alpha: torch.Tensor
    depth: torch.Tensor


class GaussianRasterizer:
    """
    Fast GPU accelerated rasterizer for gaussians
    """

    def __init__(
        self,
        K,
        device: str = "cuda",
        tile_size: int = 16,
        enable_caching: bool = True,
        cache_size: int = 10,
        num_workers: int = 4,
        backend: str = "auto",
    ):
        """
        Create fast rasterizer
        :param K: camera intrinsics
        :param device: Device for backend, normally cuda
        :param tile_size: Very important for memory optimization
        :param enable_caching: For caching, for many gaussians should be enabled
        :param cache_size: Size for cache - generally don't change this
        :param num_workers: Increase for many CPU cores
        :param backend: Backend for rasterizer. Options: 'auto', 'gplat', 'pytorch'.
        """
        self._cached_gaussian_params = None
        self._project_gaussians = None
        self._cached_gaussian_version = None
        self.device = torch.device(device)
        self.tile_size = tile_size
        self.num_workers = num_workers
        self.K = K

        self.enable_caching = enable_caching
        self.cache_size = cache_size
        self.render_cache = deque(maxlen=cache_size)

        self.cache_hits = 0
        self.cache_misses = 0

        self.buffer_pool = {}

        self.preprocessing_executor = ThreadPoolExecutor(max_workers=num_workers)
        self.preprocessing_queue = deque(maxlen=5)

        self.backend = self._initialize_backend(backend, self.K)
        self.render_stream = torch.cuda.Stream()
        self.preprocess_stream = torch.cuda.Stream()

        self._project_gaussians_compiled = torch.compile(
            self._project_gaussians, mode="max-autotune"
        )

    def _initialize_backend(self, backend, K):
        if backend == "auto":
            backends_to_try = ["gplat", "pytorch"]
        else:
            backends_to_try = [backend]

        for backend_name in backends_to_try:
            try:
                if backend_name == "gsplat":
                    from gsplat import rasterization

                    log(INFO, "Using gsplat backend (fastest)")
                    return GSplatBackend(K)

                elif backend_name == "pytorch":
                    log(INFO, "Using PyTorch backend (fallback)")
                    return PyTorchBackend(K)

            except ImportError:
                continue

        raise RuntimeError("No rasterization backend available!")

    @torch.cuda.nvtx.range("render_sequential")
    def render_sequential_batch(
        self,
        viewpoints: List[Dict],
        gaussians: "GaussianModel",
        bg_color: Optional[torch.Tensor] = None,
        scaling_modifier: float = 1.0,
    ) -> List[Dict]:
        """
        Optimized rendering for sequential frames with temporal coherence
        """

        if not bg_color:
            bg_color = torch.zeros(3, device=self.device)

        rendered_frames = []

        # Get Gaussian parameters once for the entire batch
        with torch.cuda.nvtx.range("get_gaussian_params"):
            gaussian_params = self._get_gaussian_params_cached(gaussians)

        view_groups = self._group_similar_viewpoints(viewpoints)

        for group in view_groups:
            if len(group) > 1:
                # Batch render similar views
                rendered = self._render_batch(group, gaussian_params, bg_color)
                rendered_frames.extend(rendered)
            else:
                rendered = self._render_single_cached(
                    group[0], gaussian_params, bg_color
                )
                rendered_frames.append(rendered)

        return rendered_frames

    def _get_gaussian_params_cached(self, gaussians):
        """
        Cache gaussians and avoids recomputation
        :param gaussians: The gaussians to cache
        :return: gaussian params
        """

        if not hasattr(self, "_cached_gaussian_version"):
            # No change
            self._cached_gaussian_version = -1

        current_version = gaussians.version if hasattr(gaussians, "version") else 0

        if current_version != self._cached_gaussian_version:
            self._cached_gaussian_params = {
                "means3D": gaussians.get_xyz,
                "scales": gaussians.get_scaling,
                "rotations": gaussians.get_rotation,
                "opacities": gaussians.get_opacity,
                "shs": gaussians.get_features,
            }
            self._cached_gaussian_version = current_version

        return self._cached_gaussian_params

    def _group_similar_viewpoints(
        self, viewpoints: List[Dict], similarity_threshold: float = 0.95
    ) -> List[List[Dict]]:
        """
        Groups similar viewpoints with threshold and batch processing
        """

        if len(viewpoints) <= 1:
            return [viewpoints]

        groups = []
        current_group = [viewpoints[0]]

        for i in range(1, len(viewpoints)):
            # Check similarity with last group
            if self._viewpoints_similar(
                current_group[-1], viewpoints[i], similarity_threshold
            ):
                current_group.append(viewpoints[i])
            else:
                groups.append(current_group)
                current_group = [viewpoints[i]]

        if current_group:
            groups.append(current_group)

        return groups

    def _viewpoints_similar(self, vp1: Dict, vp2: Dict, threshold: float) -> bool:
        """
        Check if two viewpoints are similar enough for batching
        """
        # Compare view matrices
        view_diff = torch.norm(
            vp1["world_view_transform"] - vp2["world_view_transform"]
        )
        return view_diff < (1 - threshold)

    @torch.cuda.nvtx.range("render_batch")
    def _render_batch(
        self, viewpoints: List[Dict], gaussian_params: Dict, bg_color: torch.Tensor
    ) -> List[Dict]:
        """
        Renders multiple viewpoints efficiently
        """

        batch_size = len(viewpoints)

        # Stack view matrices for batch processing
        view_matrices = torch.stack([vp["world_view_transform"] for vp in viewpoints])
        proj_matrices = torch.stack([vp["projection_matrix"] for vp in viewpoints])

        # Use the optimized backend
        with self.render_stream:
            rendered_batch = self.backend.render_batch(
                gaussian_params,
                view_matrices,
                proj_matrices,
                viewpoints[0]["image_width"],
                viewpoints[0]["image_height"],
                bg_color,
            )

            # Split batch results
            results = []
            for i in range(batch_size):
                results.append(
                    {
                        "render": rendered_batch["images"][i],
                        "alpha": rendered_batch["alphas"][i],
                        "depth": rendered_batch["depths"][i]
                        if "depths" in rendered_batch
                        else None,
                    }
                )

            return results

    def _render_single_cached(
        self, viewpoint: Dict, gaussian_params: Dict, bg_color: torch.Tensor
    ) -> Dict:
        """
        Renders a single viewpoint and caches
        """

        # Checks cache for similar
        if self.enable_caching:
            cached = self._check_cache(viewpoint)
            if cached:
                self.cache_hits += 1
                return cached
            self.cache_misses += 1

        with self.render_stream:
            result = self.backend.render_single(gaussian_params, viewpoint, bg_color)

        if self.enable_caching:
            self._update_cache(viewpoint, result)

        return result

    def _check_cache(self, viewpoint: Dict) -> Optional[Dict]:
        view_matrix = viewpoint["world_view_transform"]

        for entry in self.render_cache:
            diff = torch.norm(entry.view_matrix - view_matrix)
            if diff < 0.01:
                return {
                    "render": entry.rendered_image,
                    "alpha": entry.alpha,
                    "depth": entry.depth,
                }
        return None

    def _update_cache(self, viewpoint: Dict, result: Dict):
        """
        Update render cache with new result
        """
        cache_entry = RenderCache(
            means2d=result.get("means2d"),
            depths=result.get("depth"),
            radii=result.get("radii"),
            conics=result.get("conics"),
            tiles_touched=result.get("tiles_touched"),
            frame_idx=viewpoint.get("frame_idx", 0),
            view_matrix=viewpoint["world_view_transform"],
            rendered_image=result["render"],
            alpha=result.get("alpha"),
            depth=result.get("depth"),
        )
        self.render_cache.append(cache_entry)

    def get_stats(self) -> Dict:
        """Get performance stats"""
        total = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total if total > 0 else 0

        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": hit_rate,
            "total_renders": total,
        }
