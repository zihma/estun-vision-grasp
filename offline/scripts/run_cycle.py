#!/usr/bin/python3
"""Run full grasp cycle directly, preserving terminal stdin for the GRASP prompt."""
import argparse, os, tempfile
from pathlib import Path
import yaml
from ament_index_python.packages import get_package_prefix
from moveit_configs_utils import MoveItConfigsBuilder

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--candidate-file',type=Path,required=True)
 ap.add_argument('--report-file',type=Path,required=True)
 ap.add_argument('--min-tip-table-clearance-m',type=float,default=0.0)
 ap.add_argument('--execute',action='store_true')
 ap.add_argument('--operator-reviewed',action='store_true')
 ap.add_argument('--write-params',type=Path,help='Only write ROS parameters; does not start a node')
 a=ap.parse_args()
 if not 0.0<=a.min_tip_table_clearance_m<=.100:raise ValueError('Clearance must be between 0.000 and 0.100 m')
 if a.execute and (not a.operator_reviewed or not os.isatty(0)):
  raise ValueError('Execution requires an interactive terminal and --operator-reviewed')
 if not a.candidate_file.is_file():raise FileNotFoundError(a.candidate_file)
 config=(MoveItConfigsBuilder('S3-60 Eco 2024',package_name='estun_s3_60_eco_moveit_config')
  .robot_description(file_path='/opt/robot_ws/src/estun_s3_60_eco_description/urdf/s3_60_eco.urdf').to_moveit_configs())
 params=config.to_dict();params.update(min_tip_table_clearance_m=a.min_tip_table_clearance_m,candidate_file=str(a.candidate_file.resolve()),report_file=str(a.report_file.resolve()),state_source='live',execute=a.execute,operator_reviewed=a.operator_reviewed)
 if a.write_params:
  a.write_params.write_text(yaml.safe_dump({'/**':{'ros__parameters':params}},sort_keys=False));print(a.write_params);return
 fd,path=tempfile.mkstemp(prefix='anygrasp_cycle_',suffix='.yaml')
 with os.fdopen(fd,'w') as f:yaml.safe_dump({'/**':{'ros__parameters':params}},f,sort_keys=False)
 exe=Path(get_package_prefix('estun_anygrasp_motion'))/'lib/estun_anygrasp_motion/anygrasp_cycle'
 os.execv(str(exe),[str(exe),'--ros-args','--params-file',path])
if __name__=='__main__':main()
