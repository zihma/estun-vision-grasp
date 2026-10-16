"""Explicit-unit RGB-D projection for Gemini color optical frame."""
import json
from pathlib import Path

import numpy as np


def load_capture(folder):
    """Load RGB PNG, aligned depth NPY (millimetres), and intrinsic JSON."""
    import cv2

    folder = Path(folder)
    rgb = cv2.cvtColor(cv2.imread(str(folder / "rgb.png"), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    depth_mm = np.load(folder / "depth_mm.npy", allow_pickle=False)
    meta = json.loads((folder / "intrinsics.json").read_text())
    if meta.get("depth_unit") != "mm" or meta.get("aligned_to") != "rgb":
        raise ValueError("Capture must explicitly declare aligned RGB depth in mm")
    if rgb.shape[:2] != depth_mm.shape:
        raise ValueError("RGB and aligned depth shapes differ")
    return rgb, depth_mm, meta


def project(rgb, depth_mm, intrinsics, min_m=0.10, max_m=2.0):
    """Return valid Nx3 metres, Nx3 RGB [0,1], and full-image validity mask."""
    if rgb.shape[:2] != depth_mm.shape or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("RGB and depth dimensions must align")
    if not np.issubdtype(depth_mm.dtype, np.number):
        raise ValueError("Depth array must be numeric")
    fx, fy, cx, cy = (float(intrinsics[k]) for k in ("fx", "fy", "cx", "cy"))
    if not np.all(np.isfinite([fx, fy, cx, cy])) or fx <= 0 or fy <= 0:
        raise ValueError("Invalid RGB camera intrinsics")
    z = np.asarray(depth_mm, np.float32) * 0.001
    valid = np.isfinite(z) & (z >= min_m) & (z <= max_m)
    v, u = np.indices(z.shape)
    xyz = np.stack(((u - cx) * z / fx, (v - cy) * z / fy, z), axis=-1)
    return xyz[valid].astype(np.float32), rgb[valid].astype(np.float32) / 255.0, valid


def save_ply(path, points, colors):
    """Portable point cloud output without an Open3D dependency."""
    path = Path(path)
    colors = np.clip(np.asarray(colors) * 255, 0, 255).astype(np.uint8)
    with path.open("wb") as f:
        f.write(("ply\nformat binary_little_endian 1.0\n"
                 f"element vertex {len(points)}\n"
                 "property float x\nproperty float y\nproperty float z\n"
                 "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n").encode())
        data = np.empty(len(points), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                           ("r", "u1"), ("g", "u1"), ("b", "u1")])
        for i, key in enumerate(("x", "y", "z")):
            data[key] = points[:, i]
        for i, key in enumerate(("r", "g", "b")):
            data[key] = colors[:, i]
        data.tofile(f)
