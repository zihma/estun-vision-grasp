"""Reopen saved scene, highlighted target, and candidate axes in Open3D."""
import argparse
import json
from pathlib import Path

import open3d as o3d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("output", type=Path)
    args = ap.parse_args()
    report = json.loads((args.output / "results.json").read_text())
    scene = o3d.io.read_point_cloud(str(args.output / "scene.ply"))
    target = o3d.io.read_point_cloud(str(args.output / "target.ply"))
    target.paint_uniform_color([1.0, 0.75, 0.0])
    things = [scene, target]
    if report["candidates"]:
        things.append(o3d.io.read_line_set(str(args.output / "grasp_axes.ply")))
    print("Candidate scores:", [round(g["model_score"], 3) for g in report["candidates"]])
    o3d.visualization.draw_geometries(things, window_name="Gold target; red approach; blue closing")


if __name__ == "__main__":
    main()
