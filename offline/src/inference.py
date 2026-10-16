"""Official licensed SDK adapter. Nothing here performs robot execution."""
from argparse import Namespace

import numpy as np


def predict(points, target_mask, checkpoint, max_width_m, finger_height_m, limit=20, approach_direction=None, approach_thresh_deg=25.0):
    if points.dtype != np.float32 or points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("AnyGrasp input must be Nx3 float32 metres")
    if target_mask.shape != (len(points),) or not target_mask.any():
        raise ValueError("Target mask must match nonempty point cloud")
    if not (0 < max_width_m <= 0.1) or finger_height_m <= 0:
        raise ValueError("Explicit physical gripper width/height required; SDK width <= 0.1 m")
    if not np.isfinite(points).all() or not np.isfinite(finger_height_m):
        raise ValueError("Nonfinite point cloud or finger height")
    from gsnet import create_detector, check_license
    if check_license('license') is not True:
        raise RuntimeError("Official AnyGrasp license verification failed")

    detector = create_detector(Namespace(checkpoint_path=str(checkpoint),
                                         max_gripper_width=max_width_m,
                                         gripper_height=finger_height_m))
    if detector is None:
        raise RuntimeError("Official AnyGrasp detector failed to load; verify license and checkpoint")
    params = {"region_steering": target_mask.astype(bool),
              "collision_detection": True, "dense_grasp": False}
    if approach_direction is not None:
        direction = np.asarray(approach_direction, dtype=float)
        if direction.shape != (3,) or not np.isfinite(direction).all() or np.linalg.norm(direction) < 1e-8:
            raise ValueError("Approach direction must be a nonzero finite camera-frame vector")
        if not 0 <= approach_thresh_deg <= 180:
            raise ValueError("Approach angle must be between 0 and 180 degrees")
        params.update(approach_steering=direction.tolist(),
                      approach_thresh=float(np.deg2rad(approach_thresh_deg)))
    group = detector.get_grasp(points, params)
    if group is None or len(group) == 0:
        return []
    group = group.nms().sort_by_score()
    from scipy.spatial import cKDTree
    target_tree = cKDTree(points[target_mask.astype(bool)])
    results = []
    for grasp in group:
        p = np.asarray(grasp.translation, dtype=float)
        r = np.asarray(grasp.rotation_matrix, dtype=float)
        if not np.all(np.isfinite(p)) or not np.all(np.isfinite(r)):
            continue
        if target_tree.query(p + float(grasp.depth) * r[:, 0])[0] > 0.06:
            continue
        results.append({"frame": "gemini336_color_optical_frame", "unit": "m",
                        "position_xyz_m": p.tolist(), "rotation_matrix": r.tolist(),
                        "approach_xyz": r[:, 0].tolist(), "closing_axis_xyz": r[:, 1].tolist(),
                        "tip_xyz_m": (p + float(grasp.depth) * r[:, 0]).tolist(),
                        "opening_width_m": float(grasp.width), "finger_height_m": float(grasp.height),
                        "insertion_depth_m": float(grasp.depth), "model_score": float(grasp.score),
                        "object_id": int(grasp.object_id)})
        if len(results) >= limit:
            break
    return results
