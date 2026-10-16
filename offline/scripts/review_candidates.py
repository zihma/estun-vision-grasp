"""Offline coordinate review only; never generates executable robot commands."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import yaml
from scipy.spatial.transform import Rotation


# Operator-requested effective grasp reference; physical grasp remains unverified.
FLANGE_TO_GRASP_REFERENCE_M = .150
URDF_FLANGE_TO_TOOL0_M = .150
PHYSICAL_FLANGE_TO_TIP_M = .180

def geometry_assumptions(grasp_reference_m=FLANGE_TO_GRASP_REFERENCE_M):
    return {"flange_to_grasp_reference_m": grasp_reference_m,
            "urdf_flange_to_tool0_m": URDF_FLANGE_TO_TOOL0_M,
            "physical_flange_to_tip_m": PHYSICAL_FLANGE_TO_TIP_M,
            "tool0_to_grasp_reference_m": grasp_reference_m - URDF_FLANGE_TO_TOOL0_M,
            "status": "Grasp reference is configurable; physical fingertip remains 180 mm; physical grasp unverified"}

def review(candidate, transform, grasp_reference_m=FLANGE_TO_GRASP_REFERENCE_M):
    if not np.isfinite(grasp_reference_m) or not .150 <= grasp_reference_m <= .180:
        raise ValueError("Grasp reference must be between 150 and 180 mm")
    r = np.asarray(candidate['rotation_matrix'], float)
    if r.shape != (3, 3) or not np.isfinite(r).all() or not np.allclose(r.T @ r, np.eye(3), atol=1e-4) or not np.isclose(np.linalg.det(r), 1, atol=1e-4):
        raise ValueError('Invalid candidate rotation')
    r_bg = transform[:3, :3] @ r
    p = transform[:3, :3] @ np.asarray(candidate['position_xyz_m']) + transform[:3, 3]
    approach, closing = r_bg[:, 0], r_bg[:, 1]
    reference = p + candidate['insertion_depth_m'] * approach
    tool = reference - (grasp_reference_m - URDF_FLANGE_TO_TOOL0_M) * approach
    tip = tool + (PHYSICAL_FLANGE_TO_TIP_M - URDF_FLANGE_TO_TOOL0_M) * approach
    c = np.array([np.sqrt(.5), np.sqrt(.5), 0.])
    z = np.array([0., 0., 1.])
    tool_basis = np.column_stack((c, np.cross(z, c), z))
    poses = []
    for sign in (1., -1.):
        r_bt = np.column_stack((sign * closing, sign * r_bg[:, 2], approach)) @ tool_basis.T
        poses.append(Rotation.from_matrix(r_bt).as_quat().tolist())
    tilt = float(np.degrees(np.arccos(np.clip(-approach[2], -1, 1))))
    return {'model_score': candidate['model_score'], 'opening_width_m': candidate['opening_width_m'],
            'grasp_reference_base_m': reference.tolist(), 'geometry': geometry_assumptions(grasp_reference_m),
            'grasp_origin_base_m': p.tolist(), 'estimated_tip_base_m': tip.tolist(),
            'estimated_tool0_base_m': tool.tolist(),
            'estimated_pregrasp_tool0_base_m': (tool - .150 * approach).tolist(),
            'equivalent_tool0_quaternions_xyzw': poses,
            'approach_base': approach.tolist(), 'closing_base': closing.tolist(),
            'tilt_from_down_deg': tilt, 'within_25deg_down_cone': tilt <= 25.,
            'opening_plus_10mm_within_measured_120mm': candidate['opening_width_m'] + .010 <= .120}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('results', type=Path)
    ap.add_argument('--calibration', type=Path, required=True)
    args = ap.parse_args()
    result = json.loads(args.results.read_text())
    if not result.get('anygrasp_ran') or result.get('unit') != 'm':
        raise ValueError('Requires real AnyGrasp results in metres')
    cal = yaml.safe_load(args.calibration.read_text())
    if not cal.get('calibrated') or cal['parent_frame'] != 'base_link' or cal['child_frame'] != result['frame']:
        raise ValueError('Calibration/frame mismatch')
    t = np.asarray(cal['matrix_T_base_camera'], float)
    if t.shape != (4, 4) or not np.isfinite(t).all() or not np.allclose(t[3], [0, 0, 0, 1]) or not np.allclose(t[:3,:3].T @ t[:3,:3], np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(t[:3,:3]),1):
        raise ValueError('Invalid calibration matrix')
    out = {'mode': 'OFFLINE_REVIEW_ONLY', 'robot_execution_allowed': False,
           'source_results': str(args.results.resolve()), 'calibration': str(args.calibration.resolve()),
           'calibration_sha256': hashlib.sha256(args.calibration.read_bytes()).hexdigest(),
           'assumptions': {'measured_max_opening_m': .120, **geometry_assumptions(),
                           'urdf_flange_to_tool0_m': .150, 'closing_axis_tool0': [2**-.5, 2**-.5, 0],
                           'pregrasp_retreat_m': .150},
           'limitations': ['Saved capture is stale for motion', 'Calibration must match current camera placement',
                           'Physical contact offset remains approximate', 'Four-finger collisions and IK are not checked'],
           'candidates': [dict(rank=i+1, **review(g, t)) for i, g in enumerate(result['candidates'])]}
    path = args.results.parent / 'robot_frame_review.json'
    path.write_text(json.dumps(out, indent=2))
    print(path)

if __name__ == '__main__': main()
