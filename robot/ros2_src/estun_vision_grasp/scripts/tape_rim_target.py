"""Choose a physical point on the left side of a flat tape roll.

The detector's box center looks through the roll's hole. This module uses the
calibrated depth to require a raised, coherent patch on the left annulus.
"""

import numpy as np


class InvalidTapeTarget(ValueError):
    pass


def select_left_rim(depth_mm, bbox, intrinsics, transform):
    h, w = depth_mm.shape
    x1, y1, x2, y2 = (int(v) for v in bbox)
    bw, bh = x2 - x1, y2 - y1
    if bw < 40 or bh < 40 or x1 < 0 or y1 < 0 or x2 > w or y2 > h:
        raise InvalidTapeTarget('胶带检测框过小或超出图像')
    fx, fy, cx, cy = (float(intrinsics[k]) for k in ('fx', 'fy', 'cx', 'cy'))
    if not np.isfinite([fx, fy, cx, cy]).all() or min(fx, fy) <= 0:
        raise InvalidTapeTarget('相机内参无效')
    transform = np.asarray(transform, dtype=float)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise InvalidTapeTarget('手眼矩阵无效')

    yy, xx = np.mgrid[y1:y2, x1:x2]
    z = np.asarray(depth_mm[y1:y2, x1:x2], dtype=float) / 1000.0
    valid = np.isfinite(z) & (z > 0.1) & (z < 3.0)
    camera = np.stack(((xx - cx) * z / fx, (yy - cy) * z / fy, z), axis=-1)
    base_z = camera @ transform[2, :3] + transform[2, 3]

    # A flat roll exposes the tabletop in its hole. Require that the chosen
    # annulus patch rises well above this measured hole depth.
    hole = ((xx - (x1 + x2) / 2) ** 2 / (0.10 * bw) ** 2 +
            (yy - (y1 + y2) / 2) ** 2 / (0.10 * bh) ** 2 <= 1)
    hole_values = base_z[valid & hole]
    if len(hole_values) < 12:
        raise InvalidTapeTarget('中孔深度不足，无法区分外圈与桌面')
    hole_z = float(np.median(hole_values))

    # Midpoint of the visible left band, away from both the outer edge and
    # the hole. A small disk limits contamination by surrounding tabletop.
    preferred_x = x1 + 0.17 * bw
    preferred_y = y1 + 0.50 * bh
    radius = max(3.0, min(bw, bh) * 0.065)
    patch = ((xx - preferred_x) ** 2 + (yy - preferred_y) ** 2 <= radius ** 2)
    raised = valid & patch & (base_z > hole_z + 0.020)
    values = base_z[raised]
    if len(values) < 8:
        raise InvalidTapeTarget('左侧外圈没有足够的实体顶面深度')
    top_z = float(np.median(values))
    surface = raised & (np.abs(base_z - top_z) <= 0.006)
    if np.count_nonzero(surface) < 8 or np.median(np.abs(base_z[surface] - top_z)) > 0.004:
        raise InvalidTapeTarget('左侧外圈深度不稳定')
    camera_point = np.median(camera[surface], axis=0)
    pixel = (int(round(float(np.median(xx[surface])))),
             int(round(float(np.median(yy[surface])))))
    center_ray = np.array([((x1 + x2) / 2 - cx) / fx,
                           ((y1 + y2) / 2 - cy) / fy, 1.0])
    base_ray = transform[:3, :3] @ center_ray
    if abs(base_ray[2]) < 0.1:
        raise InvalidTapeTarget('相机视线与胶带顶面过于平行')
    center_distance = (top_z - transform[2, 3]) / base_ray[2]
    if center_distance <= 0:
        raise InvalidTapeTarget('胶带中心投影无效')
    center_base = transform[:3, 3] + center_distance * base_ray
    point_base = transform[:3, :3] @ camera_point + transform[:3, 3]
    radial = point_base[:2] - center_base[:2]
    radius_m = float(np.linalg.norm(radial))
    if not 0.008 <= radius_m <= 0.15:
        raise InvalidTapeTarget('胶带外圈方向或半径无效')
    radial /= radius_m
    return camera_point.tolist(), pixel, {'hole_base_z_m': hole_z,
                                          'rim_base_z_m': top_z,
                                          'height_m': top_z - hole_z,
                                          'surface_pixels': int(np.count_nonzero(surface)),
                                          'radial_axis_base_xy': radial.tolist(),
                                          'radius_m': radius_m}
