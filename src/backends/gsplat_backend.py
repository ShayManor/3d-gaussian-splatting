from typing import Dict

import torch
from gsplat import rasterization


class GSplatBackend:
    def __init__(self, K):
        self.rasterization = rasterization
        self.K = K

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

        renders, alphas, meta = self.rasterization(
            means=means3D,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=viewmat[None, ...],  # Add batch dimension [1, 4, 4]
            Ks=self.K[None, ...],  # Add batch dimension [1, 3, 3]
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

    def render_with_depth(
            self,
            gaussian_params: Dict,
            viewpoint: Dict,
            bg_color: torch.Tensor = None,
            render_mode: str = "RGB+ED",
    ) -> Dict:
        """
        Render with explicit depth output
        render_mode options:
        - "RGB": Color only
        - "D": Accumulated depth only
        - "ED": Expected depth only
        - "RGB+D": Color with accumulated depth
        - "RGB+ED": Color with expected depth
        """

        if bg_color is None:
            bg_color = torch.zeros(3, device='cuda')

        # Extract parameters (same as render_single)
        means3D = gaussian_params['means3D']
        scales = gaussian_params['scales']
        quats = gaussian_params['rotations']
        opacities = gaussian_params['opacities']

        if opacities.dim() == 2:
            opacities = opacities.squeeze(-1)

        colors = gaussian_params.get('colors', gaussian_params.get('shs'))
        sh_degree = 3 if 'shs' in gaussian_params else None

        # Camera parameters
        viewmat = viewpoint['world_view_transform']

        # Render with specified mode
        renders, alphas, meta = self.rasterization(
            means=means3D,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=viewmat[None, ...],
            Ks=self.K[None, ...],
            width=viewpoint['image_width'],
            height=viewpoint['image_height'],
            packed=False,
            render_mode=render_mode,
            sh_degree=sh_degree,
            backgrounds=bg_color[None, ...],
        )

        # Parse output based on render mode
        output = renders[0]
        alpha = alphas[0]

        result = {'alpha': alpha}

        if render_mode == "RGB":
            result['render'] = output  # [H, W, 3]
        elif render_mode == "D":
            result['accumulated_depth'] = output.squeeze(-1)  # [H, W]
        elif render_mode == "ED":
            result['expected_depth'] = output.squeeze(-1)  # [H, W]
        elif render_mode == "RGB+D":
            result['render'] = output[..., :3]  # [H, W, 3]
            result['accumulated_depth'] = output[..., 3]  # [H, W]
        elif render_mode == "RGB+ED":
            result['render'] = output[..., :3]  # [H, W, 3]
            result['expected_depth'] = output[..., 3]  # [H, W]

        # Add metadata
        if 'means2d' in meta:
            result['means2d'] = meta['means2d'][0]
        if 'radii' in meta:
            result['radii'] = meta['radii'][0]
            result['visibility_filter'] = result['radii'] > 0

        return result
