"""One-shot Gemini 336 RGB-D capture. Run only after other camera clients stop."""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gemini336_camera import GeminiCamera


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default=datetime.now().strftime("capture_%Y%m%d_%H%M%S"))
    args = ap.parse_args()
    out = ROOT / "data" / args.name
    out.mkdir(parents=True, exist_ok=False)
    camera = GeminiCamera()
    try:
        camera.start()
        frame = None
        for _ in range(20):
            frame = camera.get_frames()
            if frame is not None and frame["rgb"] is not None and frame["depth"] is not None:
                break
        if frame is None:
            raise RuntimeError("No complete RGB-D frame received")
        rgb_bgr = frame["rgb"]
        depth_mm = np.asarray(frame["depth"], np.float32)
        if rgb_bgr.shape[:2] != depth_mm.shape:
            raise ValueError("Hardware D2C returned mismatched RGB/depth sizes")
        skew_ms = abs(frame["timestamps"]["rgb"] - frame["timestamps"]["depth"]) / 1000.0
        if skew_ms > 50:
            raise ValueError(f"RGB/depth timestamp difference too large: {skew_ms:.1f} ms")
        intrinsics = {**camera.intrinsics, "depth_unit": "mm", "aligned_to": "rgb",
                      "frame": "gemini336_color_optical_frame",
                      "source": "GeminiCamera._depth_to_mm: raw uint16 * frame.get_depth_scale()",
                      "timestamps_us": frame["timestamps"]}
        cv2.imwrite(str(out / "rgb.png"), rgb_bgr)
        np.save(out / "depth_mm.npy", depth_mm, allow_pickle=False)
        (out / "intrinsics.json").write_text(json.dumps(intrinsics, indent=2))
        valid = depth_mm[np.isfinite(depth_mm) & (depth_mm > 0)]
        print(json.dumps({"saved": str(out), "shape": list(depth_mm.shape),
                          "rgb_depth_skew_ms": skew_ms,
                          "valid_fraction": float(len(valid) / depth_mm.size),
                          "depth_mm_min_median_max": [float(valid.min()), float(np.median(valid)), float(valid.max())]
                          if len(valid) else None}, indent=2))
    finally:
        camera.stop()


if __name__ == "__main__":
    main()
