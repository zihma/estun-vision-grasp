"""Render archived, object-only real grasp candidates without hardware or inference."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'offline/src'))
from visualize import save_html


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'demo_output')
    args = parser.parse_args()
    example = ROOT / 'examples/bowl'
    with np.load(example / 'target_points.npz', allow_pickle=False) as sample:
        points = sample['points_m']
    result = json.loads((example / 'results.json').read_text())
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError('Invalid object point cloud')
    if len(points) != result['target_points'] or result['unit'] != 'm':
        raise ValueError('Sample count or unit mismatch')
    for grasp in result['candidates']:
        rotation = np.asarray(grasp['rotation_matrix'])
        if rotation.shape != (3, 3) or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4):
            raise ValueError('Invalid saved candidate rotation')
    args.output.mkdir(parents=True, exist_ok=True)
    save_html(args.output / 'scene.html', points, np.ones_like(points) * .7,
              np.ones(len(points), dtype=bool), result['candidates'])
    report = dict(mode='archived_candidate_replay', point_count=len(points),
                  candidate_count=len(result['candidates']), frame=result['frame'], unit='m',
                  neural_inference_ran=False, robot_connected=False,
                  note='Object-only historical data; no full-scene collision revalidation.')
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    print('Open:', args.output / 'scene.html')


if __name__ == '__main__':
    main()
