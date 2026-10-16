"""Render actual saved AnyGrasp predictions as a static 3D figure."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, default=Path(__file__).resolve().parents[1] / "outputs/pringles_anygrasp_topdown_20261011")
    p.add_argument("--count", type=int, default=5)
    a = p.parse_args()
    report = json.loads((a.results / "results.json").read_text())
    candidates = report["candidates"][:a.count]
    scene = np.asarray(o3d.io.read_point_cloud(str(a.results / "scene.ply")).points)
    target = np.asarray(o3d.io.read_point_cloud(str(a.results / "target.ply")).points)
    if not len(target):
        raise ValueError("No target cloud in inference result")
    lo, hi = target.min(axis=0) - 0.12, target.max(axis=0) + 0.12
    local = scene[((scene >= lo) & (scene <= hi)).all(axis=1)][::10]
    target = target[::2]
    fig = plt.figure(figsize=(14, 7), layout="constrained")
    ax = fig.add_subplot(121, projection="3d")
    ax.scatter(local[:, 0], local[:, 1], local[:, 2], s=0.3, c="lightgray", alpha=0.35, rasterized=True)
    ax.scatter(target[:, 0], target[:, 1], target[:, 2], s=1.0, c="goldenrod", alpha=0.8, rasterized=True)
    for rank, c in enumerate(candidates, 1):
        xyz = np.asarray(c["position_xyz_m"])
        rot = np.asarray(c["rotation_matrix"])
        width = c["opening_width_m"]
        ax.scatter(*xyz, color="black", s=35)
        ax.quiver(*xyz, *(rot[:, 0] * 0.06), color="red", linewidth=2, arrow_length_ratio=0.25)
        ax.plot(*np.vstack((xyz - width/2 * rot[:, 1], xyz + width/2 * rot[:, 1])).T, color="blue", lw=2)
        ax.text(*xyz, f"{rank}:{c['model_score']:.2f}", fontsize=9)
    ax.set(xlabel="X right (m)", ylabel="Y down (m)", zlabel="Z forward (m)",
           title="Gemini 336 target and AnyGrasp predictions")
    ax.set_box_aspect((1, 1, 0.7))
    ax.view_init(elev=26, azim=-55)
    ax2 = fig.add_subplot(122)
    rgb = a.results / "selection_overlay.png"
    ax2.imshow(plt.imread(rgb))
    ax2.set_title("Saved RGB target selection")
    ax2.axis("off")
    fig.text(0.5, 0.02, "Red: approach (+grasp X)    Blue: jaw closing axis (+grasp Y)    Labels: rank:score\n"
             "Camera optical frame, metres. Scores are model scores, not physical success probabilities.",
             ha="center", fontsize=10)
    out = a.results / "grasp_preview.png"
    fig.savefig(out, dpi=180)
    print(out)


if __name__ == "__main__":
    main()
