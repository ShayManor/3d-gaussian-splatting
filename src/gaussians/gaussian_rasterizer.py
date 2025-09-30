from collections import deque
from concurrent.futures import ThreadPoolExecutor
from logging import log, INFO
from typing import List, Dict, Optional

import torch


class GaussianRasterizer:
    """
    Fast GPU accelerated rasterizer for gaussians
    """

    def __init__(
            self,
            device: str = 'cuda',
            tile_size: int = 16,
            enable_caching: bool = True,
            cache_size: int = 10,
            num_workers: int = 4,
            backend: str = 'auto'
    ):
        """
        Create fast rasterizer
        :param device: Device for backend, normally cuda
        :param tile_size: Very important for memory optimization
        :param enable_caching: For caching, for many gaussians should be enabled
        :param cache_size: Size for cache - generally don't change this
        :param num_workers: Increase for many CPU cores
        :param backend: Backend for rasterizer. Options: 'auto', 'gplat', 'pytorch'.
        """
        self.device = torch.device(device)
        self.tile_size = tile_size
        self.num_workers = num_workers

        self.enable_caching = enable_caching
        self.cache_size = cache_size
        self.render_cache = deque(maxlen=cache_size)

        self.cache_hits = 0
        self.cache_misses = 0

        self.buffer_pool = {}

        self.preprocessing_executor = ThreadPoolExecutor(max_workers=num_workers)
        self.preprocessing_queue = deque(maxlen=5)

        self.backend = self._initialize_backend(backend)
        self.render_stream = torch.cuda.Stream()
        self.preprocess_stream = torch.cuda.Stream()

        self._project_gaussians_compiled = torch.compile(
            self._project_gaussians,
            mode="max-autotune"
        )

    def _initialize_backend(self, backend):
        if backend == 'auto':
            backends_to_try = ['gplat', 'pytorch']
        else:
            backends_to_try = [backend]

        for backend_name in backends_to_try:
            try:
                if backend_name == 'gsplat':
                    from gsplat import rasterization, project_gaussians
                    log(INFO, "Using gsplat backend (fastest)")
                    return GSplatBackend()

                elif backend_name == 'pytorch':
                    log(INFO, "Using PyTorch backend (fallback)")
                    return PyTorchBackend()

            except ImportError:
                continue

        raise RuntimeError("No rasterization backend available!")

    @torch.cuda.nvtx.range("render_sequential")
    def render_sequential_batch(self,
        viewpoints: List[Dict],
        gaussians: 'GaussianModel',
        bg_color: Optional[torch.Tensor] = None,
        scaling_modifier: float = 1.0
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