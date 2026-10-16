"""Export AnyGrasp candidates for plan-only MoveIt; never authorize motion."""
import argparse, hashlib, json, re, sys, time
from pathlib import Path
import numpy as np, yaml
sys.path.insert(0,str(Path(__file__).resolve().parent))
from review_candidates import review, geometry_assumptions

def export(results,calibration):
 result=json.loads(results.read_text());cal=yaml.safe_load(calibration.read_text())
 if not result.get('anygrasp_ran') or not result.get('license_passed') or result.get('unit')!='m':raise ValueError('Real licensed AnyGrasp metre results required')
 if not cal.get('calibrated') or cal.get('parent_frame')!='base_link' or cal.get('child_frame')!=result['frame']:raise ValueError('Calibration mismatch')
 t=np.asarray(cal['matrix_T_base_camera'],float)
 if t.shape!=(4,4) or not np.isfinite(t).all() or not np.allclose(t[3],[0,0,0,1]) or not np.allclose(t[:3,:3].T@t[:3,:3],np.eye(3),atol=1e-5) or not np.isclose(np.linalg.det(t[:3,:3]),1):raise ValueError('Invalid transform')
 weight=Path(result['checkpoint'])
 if hashlib.sha256(weight.read_bytes()).hexdigest()!=result['checkpoint_sha256']:raise ValueError('Checkpoint hash mismatch')
 header=Path('/opt/robot_ws/src/estun_vision_grasp/src/gripper_ready_target.hpp')
 match=re.search(r'kGripperReady\s*=\s*\{([^}]+)\}',header.read_text())
 seed=[float(x.strip()) for x in match.group(1).split(',') if x.strip()]
 if len(seed)!=6:raise ValueError('Ready reference must contain six joints')
 capture=Path(result['capture'])/'rgb.png'
 bundle={'geometry':geometry_assumptions(),'schema_version':1,'mode':'PLAN_ONLY','robot_execution_allowed':False,'frame':'base_link','unit':'m',
  'results_file':str(results.resolve()),'results_sha256':hashlib.sha256(results.read_bytes()).hexdigest(),
  'calibration_file':str(calibration.resolve()),'calibration_sha256':hashlib.sha256(calibration.read_bytes()).hexdigest(),
  'checkpoint_sha256':result['checkpoint_sha256'],'export_unix_s':time.time(),
  'capture_unix_s':capture.stat().st_mtime,'capture_time_source':'saved RGB file mtime, not refreshed at export',
  'geometry_assumed':result.get('geometry_assumed',True),'real_four_finger_collision_verified':False,
  'offline_seed_joints':seed,'offline_seed_source':str(header),'candidates':[]}
 for rank,g in enumerate(result['candidates'],1):
  scalar=[g['model_score'],g['opening_width_m'],g['insertion_depth_m'],*g['position_xyz_m']]
  if not np.isfinite(scalar).all() or not 0<=g['model_score']<=1 or not 0<g['opening_width_m']<=.120 or not 0<=g['insertion_depth_m']<=.1:raise ValueError('Invalid candidate numbers')
  r=review(g,t)
  if not r['within_25deg_down_cone'] or not r['opening_plus_10mm_within_measured_120mm']:continue
  bundle['candidates'].append({'rank':rank,'score':g['model_score'],'opening_width_m':g['opening_width_m'],
   'pregrasp_xyz_m':r['estimated_pregrasp_tool0_base_m'],'tool0_quaternions_xyzw':r['equivalent_tool0_quaternions_xyzw'],
   'tilt_from_down_deg':r['tilt_from_down_deg']})
 if not bundle['candidates']:raise ValueError('No candidates pass offline width/tilt checks')
 return bundle

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('results',type=Path);ap.add_argument('--calibration',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--purpose',choices=['plan','positioning','cycle'],default='plan');ap.add_argument('--reference-positioning',type=Path);a=ap.parse_args()
 if a.output.exists():raise FileExistsError('Use a new output path to retain provenance')
 bundle=export(a.results,a.calibration)
 if a.purpose=='positioning':bundle.update(mode='PREGRASP_POSITIONING_ONLY',robot_execution_allowed=True)
 if a.purpose=='cycle':
  bundle.update(mode='FULL_GRASP_CYCLE',robot_execution_allowed=True,max_approach_tilt_deg=15.0)
  bundle['candidates']=[c for c in bundle['candidates'] if c['tilt_from_down_deg']<=15.0]
  if not bundle['candidates']:raise ValueError('No candidates within 15 degrees of vertical down; recapture/review, do not widen the limit')
  import open3d as o
  from scipy.spatial import cKDTree
  result=json.loads(a.results.read_text());cal=yaml.safe_load(a.calibration.read_text());t=np.asarray(cal['matrix_T_base_camera'],float)
  scene=np.asarray(o.io.read_point_cloud(str(a.results.parent/'scene.ply')).points);target=np.asarray(o.io.read_point_cloud(str(a.results.parent/'target.ply')).points)
  if len(target)<30:raise ValueError('Missing target cloud')
  near=scene[((scene>=target.min(0)-.12)&(scene<=target.max(0)+.12)).all(1)]
  bg=near[cKDTree(target).query(near)[0]>.015]
  if len(bg)<300:raise ValueError('Insufficient table background')
  table_z=float(np.quantile((bg@t[:3,:3].T+t[:3,3])[:,2],.25));bundle['estimated_table_z_m']=table_z
  for candidate in bundle['candidates']:
   r=review(result['candidates'][candidate['rank']-1],t)
   candidate['grasp_tool0_xyz_m']=r['estimated_tool0_base_m'];candidate['estimated_tip_table_clearance_m']=r['estimated_tip_base_m'][2]-table_z
  fallback=[]
  for candidate in bundle['candidates']:
   r=review(result['candidates'][candidate['rank']-1],t,grasp_reference_m=.160)
   alt=dict(candidate)
   alt.update(pregrasp_xyz_m=r['estimated_pregrasp_tool0_base_m'],grasp_tool0_xyz_m=r['estimated_tool0_base_m'],estimated_tip_table_clearance_m=r['estimated_tip_base_m'][2]-table_z)
   fallback.append(alt)
  bundle['fallback_geometry']=geometry_assumptions(.160)
  bundle['fallback_candidates']=fallback
  bundle['reference_fallback_policy']='150 mm first; 160 mm only after all candidates fail full preflight; never retry execution'
 a.output.write_text(yaml.safe_dump(bundle,sort_keys=False));print(f'{len(bundle["candidates"])} {bundle["mode"]} candidates -> {a.output}')
if __name__=='__main__':main()
