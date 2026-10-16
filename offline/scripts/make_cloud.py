"""Project a recorded Gemini RGB-D pair without opening the camera or SDK."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rgbd import load_capture, project, save_ply


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("capture", type=Path)
    ap.add_argument("--name", required=True)
    args = ap.parse_args()
    out = ROOT / "outputs" / args.name
    out.mkdir(parents=True, exist_ok=False)
    rgb, depth_mm, meta = load_capture(args.capture)
    points, colors, _ = project(rgb, depth_mm, meta)
    save_ply(out / "scene.ply", points, colors)
    report = {"capture": str(args.capture), "frame": meta["frame"],
              "unit": "m", "point_count": int(len(points)),
              "depth_min_m": float(points[:, 2].min()),
              "depth_max_m": float(points[:, 2].max()),
              "anygrasp_ran": False}
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
