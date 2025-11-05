import numpy as np
import torch
from plyfile import PlyData, PlyElement

def _as_np(x):
    if x is None: return None
    if torch.is_tensor(x): return x.detach().cpu().numpy()
    return np.asarray(x)

def _get_from(d, *names):
    for n in names:
        if n in d:
            return d[n]
    return None

def _load_state(path):
    ckpt = torch.load(path, map_location="cpu")
    # Accept: raw state_dict or wrapped in {"model_state": ...} or {"state_dict": ...}
    if isinstance(ckpt, dict):
        for k in ("model_state", "state_dict"):
            if k in ckpt and isinstance(ckpt[k], dict):
                return ckpt[k]
    return ckpt

def _reshape_dc(fdc, N):
    fdc = _as_np(fdc)
    if fdc is None:
        raise ValueError("Missing features_dc / f_dc / rgb_dc")
    fdc = fdc.reshape(N, -1)
    if fdc.shape[1] < 3:
        raise ValueError(f"features_dc has wrong shape: {fdc.shape}")
    return fdc[:, :3].astype(np.float32)

def _reshape_rest(fr, N):
    if fr is None:
        return None
    fr = _as_np(fr).reshape(N, -1).astype(np.float32)
    # Clip or zero-pad to 45 to match degree-3 SH layout
    if fr.shape[1] >= 45:
        return fr[:, :45]
    out = np.zeros((N, 45), dtype=np.float32)
    out[:, :fr.shape[1]] = fr
    return out

def _reshape_scales(sc, N):
    sc = _as_np(sc).reshape(N, -1).astype(np.float32)
    if sc.shape[1] == 1:
        sc = np.repeat(sc, 3, axis=1)
    if sc.shape[1] != 3:
        raise ValueError(f"scales must have 3 comps after broadcast, got {sc.shape}")
    return sc

def _to_structured(xyz, nrm, fdc, opacity, scales, rots, frest):
    # Build dtype in the canonical order
    props = [
        ("x","f4"),("y","f4"),("z","f4"),
        ("nx","f4"),("ny","f4"),("nz","f4"),
        ("f_dc_0","f4"),("f_dc_1","f4"),("f_dc_2","f4"),
        ("opacity","f4"),
        ("scale_0","f4"),("scale_1","f4"),("scale_2","f4"),
        ("rot_0","f4"),("rot_1","f4"),("rot_2","f4"),("rot_3","f4"),
    ]
    if frest is not None:
        props += [(f"f_rest_{i}","f4") for i in range(45)]
    dtype = np.dtype(props)

    N = xyz.shape[0]
    out = np.empty(N, dtype=dtype)
    # Flatten into the same order
    cols = [
        xyz, nrm, fdc, opacity.reshape(N,1), scales, rots
    ]
    if frest is not None:
        cols.append(frest)
    mat = np.concatenate([c if c.ndim==2 else c.reshape(N,-1) for c in cols], axis=1)
    out[:] = list(map(tuple, mat))
    return out

def _write_ply(path, vertex_struct):
    el = PlyElement.describe(vertex_struct, "vertex")
    PlyData([el], text=False).write(path)  # binary_little_endian

def export(input_pth: str, output_ply: str):
    sd = _load_state(input_pth)
    if not isinstance(sd, dict):
        raise ValueError("Unexpected checkpoint structure. Expected dict-like state_dict().")

    # Positions
    xyz = _as_np(_get_from(sd, "xyz", "means3D", "positions", "points", "points_xyz"))
    if xyz is None or xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"Missing or malformed positions, got {None if xyz is None else xyz.shape}")
    xyz = xyz.astype(np.float32)
    N = xyz.shape[0]

    # Normals: zeros
    nrm = np.zeros_like(xyz, dtype=np.float32)

    # Features DC
    fdc = _reshape_dc(_get_from(sd, "features_dc", "f_dc", "rgb_dc", "dc"), N)

    # Features rest (optional)
    frest = _reshape_rest(_get_from(sd, "features_rest", "f_rest", "sh_rest"), N)

    # Opacity logits
    opacity = _as_np(_get_from(sd, "opacity", "opacities", "logit_opacity", "logit_opacities"))
    if opacity is None:
        raise ValueError("Missing opacity/logit_opacity")
    opacity = opacity.reshape(N, 1).astype(np.float32)

    # Log-scales
    scales = _reshape_scales(_get_from(sd, "scaling", "scales", "log_scales", "scale"), N)

    # Rotations (quaternion)
    rots = _as_np(_get_from(sd, "rotation", "rotations", "quat", "quaternion"))
    if rots is None:
        raise ValueError("Missing rotation/rotations")
    rots = rots.reshape(N, 4).astype(np.float32)

    vertex = _to_structured(xyz, nrm, fdc, opacity, scales, rots, frest)
    _write_ply(output_ply, vertex)
    print(f"Wrote {output_ply} with {N} gaussians.")
