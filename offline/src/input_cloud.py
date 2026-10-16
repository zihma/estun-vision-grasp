"""Bound inference input while preserving the full scene for collision checks."""
import numpy as np


def bounded_voxel_input(points, target_mask, voxel_m, max_points=35000, max_voxel_m=0.008):
    if not np.isfinite(voxel_m) or voxel_m <= 0 or max_points < 30:
        raise ValueError("Invalid inference input budget")
    target_mask = np.asarray(target_mask, dtype=bool)
    if len(points) != len(target_mask) or not target_mask.any():
        raise ValueError("Nonempty target mask required")
    ceiling = max(voxel_m, max_voxel_m)
    effective = voxel_m
    while True:
        cells = np.floor(points / effective).astype(np.int64)
        _, first, inverse = np.unique(cells, axis=0, return_index=True, return_inverse=True)
        model_mask = np.zeros(len(first), dtype=bool)
        np.logical_or.at(model_mask, inverse, target_mask)
        if len(first) <= max_points:
            return points[first], model_mask, effective
        if effective >= ceiling:
            raise RuntimeError(f"Input has {len(first)} points at {effective*1000:.1f} mm; limit {max_points}. Reselect a tighter object region.")
        effective = min(ceiling, effective * 1.25)
