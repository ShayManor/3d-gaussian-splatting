from typing import Dict

import torch
from gsplat import rasterization


class GSplatBackend:
    def __init__(self):
        self.rasterization = rasterization

    def render_batch(
        self,
        gaussian_params: Dict,
        view_matrices: torch.Tensor,
        proj_matrices: torch.Tensor,
        width: int,
        height: int,
        bg_color: torch.Tensor,
        device: str,
    ):
        """
        Uses GSplat for batch rendering images
        :param gaussian_params:  Dict with means3D, scales, rotations, opacities, colors/shs
        :param view_matrices: Batch of view matrices [B, 4, 4]
        :param proj_matrices: Batch of projection matrices [B, 4, 4]
        :param width: Image width
        :param height:Image height
        :param bg_color: background color
        :param device: Device for training
        """
        batch_size = view_matrices.shape[0]

        rendered_images = torch.zeros(batch_size, 3, height, width, device=device)
        alphas = torch.zeros(batch_size, 1, height, width, device=device)

        for i in range(batch_size):
            # TODO: real batch processing
            result = self.render_single(
                gaussian_params,
                {
                    'world_view_transform': view_matrices[i],
                    'projection_matrix': proj_matrices[i],
                    'image_width': width,
                    'image_height': height
                },
                bg_color
            )
            rendered_images[i] = result['render']
            alphas[i] = result['alpha']

        return {
            'images': rendered_images,
            'alphas': alphas
        }

    def render_single(self,
            gaussian_params: Dict,
            viewpoint: Dict,
            bg_color: torch.Tensor
        ):
        """
        Single view rendering with gsplat
        """

        means3D = gaussian_params['means3D']
        scales = gaussian_params['scales']
        quats = gaussian_params['rotations']
        opacities = gaussian_params['opacities']

        # Get colors
        if 'shs' in gaussian_params and gaussian_params['shs'] is not None:
            colors = gaussian_params['shs']
            sh_degree = 3
        else:
            colors = gaussian_params.get('colors', gaussian_params.get('features', None))
            sh_degree = None

        # Camera parameters
        viewmat = viewpoint['world_view_transform']
        K = self._projection_to_intrinsic(
            viewpoint['projection_matrix'],
            viewpoint['image_width'],
            viewpoint['image_height']
        )
        if 'K' in viewpoint:
            K = viewpoint['K']

        renders, alphas, meta = self.rasterization(
            means=means3D,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=viewmat[None, ...],  # Add batch dimension [1, 4, 4]
            Ks=K[None, ...],  # Add batch dimension [1, 3, 3]
            width=viewpoint['image_width'],
            height=viewpoint['image_height'],
            packed=False,
            render_mode="RGB+ED",  # Get both color and depth
            sh_degree=sh_degree,
            backgrounds=bg_color[None, ...] if bg_color.dim() == 1 else bg_color,
        )

        rendered = renders[0]  # [H, W, 3] for RGB or [H, W, 4] for RGB+ED
        alpha = alphas[0]  # [H, W]

        if rendered.shape[-1] == 4:
            # RGB+ED
            rgb = rendered[..., :3]
            depth = rendered[..., 3]
        else:
            # RGB
            rgb = rendered
            depth = meta.get('depths', [None])[0] if 'depths' in meta else None

        result = {
            'render': rgb,  # [H, W, 3]
            'alpha': alpha,  # [H, W]
            'depth': depth,  # [H, W] or None
        }

        if meta:
            # Intermediate results
            if 'means2d' in meta:
                result['means2d'] = meta['means2d'][0]  # [N, 2]
            if 'radii' in meta:
                result['radii'] = meta['radii'][0]  # [N]
                result['visibility_filter'] = result['radii'] > 0
            if 'tiles_touched' in meta:
                result['tiles_touched'] = meta['tiles_touched'][0]

        return result
