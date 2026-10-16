"""Offline AnyGrasp inference from a saved Gemini snapshot; no ROS or motor APIs."""
import argparse
import json
import hashlib
import time
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rgbd import load_capture, project, save_ply
from selection import choose_mask, mask_from_click
from input_cloud import bounded_voxel_input
from visualize import save_html, save_lines, show


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("capture", type=Path)
    ap.add_argument("--name", required=True, help="Unique output name")
    ap.add_argument("--select", choices=["polygon", "rectangle", "click"], default="polygon")
    ap.add_argument("--mask", type=Path, help="Optional saved binary PNG mask for headless runs")
    ap.add_argument("--click-pixel", nargs=2, type=int, metavar=("X", "Y"),
                    help="Recorded manual pixel selection for reproducible headless runs")
    ap.add_argument("--checkpoint", type=Path)
    ap.add_argument("--max-width-m", type=float, help="Measured four-finger opening; <=0.1 m")
    ap.add_argument("--finger-height-m", type=float, help="Collision-model finger height in metres")
    ap.add_argument("--geometry-assumed", action="store_true", help="Mark finger height as an unmeasured offline assumption")
    ap.add_argument("--approach-camera", nargs=3, type=float)
    ap.add_argument("--approach-thresh-deg", type=float, default=25.0)
    ap.add_argument("--context-margin-m", type=float, default=0.15, help="Scene margin around target for GPU inference")
    ap.add_argument("--voxel-m", type=float, default=0.005, help="SDK voxel size; full-resolution clouds remain saved")
    ap.add_argument("--max-model-points", type=int, default=35000, help="Inference point budget; adapt voxel size up to 8 mm or reject")
    ap.add_argument("--preview-only", action="store_true", help="Save cloud and target without SDK")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()
    out = ROOT / "outputs" / args.name
    out.mkdir(parents=True, exist_ok=False)
    rgb, depth_mm, meta = load_capture(args.capture)
    points, colors, valid = project(rgb, depth_mm, meta)
    if args.mask and args.click_pixel:
        raise ValueError("Choose either --mask or --click-pixel")
    if args.mask:
        import cv2
        mask_image = cv2.imread(str(args.mask), cv2.IMREAD_GRAYSCALE)
        if mask_image is None or mask_image.shape != depth_mm.shape:
            raise ValueError("Saved target mask missing or wrong shape")
        target_image = mask_image > 0
    elif args.click_pixel:
        target_image = mask_from_click(depth_mm, *args.click_pixel)
    else:
        target_image = choose_mask(rgb, depth_mm, args.select)
    target_mask = target_image[valid]
    if target_mask.sum() < 30:
        raise ValueError("Too few valid target points")
    import cv2
    overlay = cv2.cvtColor(rgb.copy(), cv2.COLOR_RGB2BGR)
    overlay[target_image] = (0.5 * overlay[target_image] +
                             0.5 * np.array([0, 210, 255])).astype(np.uint8)
    cv2.imwrite(str(out / "selection_overlay.png"), overlay)
    save_ply(out / "scene.ply", points, colors)
    save_ply(out / "target.ply", points[target_mask], colors[target_mask])
    np.save(out / "target_mask.npy", target_image.astype(bool), allow_pickle=False)
    status = {"capture": str(args.capture), "frame": meta["frame"], "unit": "m",
              "scene_points": len(points), "target_points": int(target_mask.sum()),
              "anygrasp_ran": False, "candidates": []}
    if not args.preview_only:
        if not args.checkpoint or not args.checkpoint.is_file():
            raise FileNotFoundError("Official checkpoint required for AnyGrasp inference")
        if args.max_width_m is None or args.finger_height_m is None:
            raise ValueError("Measure and supply gripper opening and finger height")
        from inference import predict
        import torch
        torch.cuda.reset_peak_memory_stats()
        if not np.isfinite(args.voxel_m) or args.voxel_m <= 0:
            raise ValueError("Voxel size must be positive and finite")
        if not np.isfinite(args.context_margin_m) or args.context_margin_m < 0.10:
            raise ValueError("Context margin must be at least 0.10 m")
        target_points = points[target_mask]
        context = ((points >= target_points.min(0) - args.context_margin_m) &
                   (points <= target_points.max(0) + args.context_margin_m)).all(1)
        context_points, context_mask = points[context], target_mask[context]
        model_points, model_mask, effective_voxel = bounded_voxel_input(
            context_points, context_mask, args.voxel_m, args.max_model_points)
        status.update(model_points=len(model_points), model_target_points=int(model_mask.sum()),
                      voxel_m=effective_voxel, requested_voxel_m=args.voxel_m,
                      max_model_points=args.max_model_points, context_points=len(context_points),
                      context_margin_m=args.context_margin_m,
                      collision_cloud='SDK: target context; CPU postfilter: full scene')
        (out / "results.json").write_text(json.dumps(status, indent=2))
        print(f"Target: {status['target_points']} points; context: {len(context_points)}; GPU input: {len(model_points)}; voxel: {effective_voxel*1000:.2f} mm", flush=True)
        started = time.perf_counter()
        try:
            with torch.no_grad():
                candidates = predict(model_points, model_mask, args.checkpoint,
                             args.max_width_m, args.finger_height_m, limit=100,
                             approach_direction=args.approach_camera,
                             approach_thresh_deg=args.approach_thresh_deg)
        except torch.cuda.OutOfMemoryError as error:
            status.update(inference_error='CUDA out of memory', error_detail=str(error), robot_execution=False)
            (out / "results.json").write_text(json.dumps(status, indent=2))
            raise RuntimeError("GPU inference ran out of memory. No motion requested. Inspect selection_overlay.png and reselect a tighter target.") from error
        torch.cuda.synchronize()
        status.update(inference_seconds=time.perf_counter() - started,
                      peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                      checkpoint=str(args.checkpoint.resolve()),
                      checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                      license_passed=True,
                      max_width_m=args.max_width_m,
                      finger_height_m=args.finger_height_m,
                      geometry_assumed=args.geometry_assumed,
                      real_four_finger_collision_verified=False,
                      approach_camera=args.approach_camera,
                      approach_thresh_deg=args.approach_thresh_deg,
                      robot_execution=False)
        # SDK region steering is primary. Confirm candidate tip is near selected object.
        from scipy.spatial import cKDTree
        target_tree = cKDTree(points[target_mask])
        candidates = [g for g in candidates if target_tree.query(g["tip_xyz_m"])[0] <= 0.06]
        status['candidates_before_full_scene_filter'] = len(candidates)
        if candidates:
            from graspnetAPI import GraspGroup
            from collision_detector import ModelFreeCollisionDetector
            rows = [[g['model_score'], g['opening_width_m'], g['finger_height_m'],
                     g['insertion_depth_m'], *np.asarray(g['rotation_matrix']).reshape(-1),
                     *g['position_xyz_m'], -1] for g in candidates]
            group = GraspGroup(np.asarray(rows))
            collision = ModelFreeCollisionDetector(points, voxel_size=0.005)
            bad = []
            for i in range(0, len(group), 10):
                bad.extend(collision.detect(group[i:i+10], approach_dist=0.15, collision_thresh=0.01))
            candidates = [g for g, rejected in zip(candidates, bad) if not rejected]
        status['candidates_after_full_scene_filter'] = len(candidates)
        status['full_scene_collision_model'] = {'type': 'ideal_parallel_jaw', 'finger_width_m': 0.01,
            'finger_length_m': 0.06, 'voxel_m': 0.005, 'approach_dist_m': 0.15, 'collision_thresh': 0.01}
        candidates = candidates[:20]
        status["anygrasp_ran"] = True
        status["candidates"] = candidates
    (out / "results.json").write_text(json.dumps(status, indent=2))
    save_lines(out / "grasp_axes.ply", points, colors, target_mask, status["candidates"])
    save_html(out / "scene.html", points, colors, target_mask, status["candidates"])
    print(json.dumps({"output": str(out), "scene_points": len(points),
                      "target_points": int(target_mask.sum()),
                      "candidates": len(status["candidates"]),
                      "anygrasp_ran": status["anygrasp_ran"]}, indent=2))
    if args.show:
        show(points, colors, target_mask, status["candidates"])


if __name__ == "__main__":
    main()
