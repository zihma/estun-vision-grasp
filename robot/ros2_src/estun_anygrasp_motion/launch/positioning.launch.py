"""Explicit live high pregrasp positioning; defaults to no motion."""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument,OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder

def nodes(context):
 def value(name):return LaunchConfiguration(name).perform(context)
 if value('execute').lower()=='true':raise RuntimeError('Use scripts/run_positioning.py directly for the interactive POSITION confirmation')
 config=(MoveItConfigsBuilder('S3-60 Eco 2024',package_name='estun_s3_60_eco_moveit_config')
  .robot_description(file_path='/opt/robot_ws/src/estun_s3_60_eco_description/urdf/s3_60_eco.urdf').to_moveit_configs())
 return [Node(package='estun_anygrasp_motion',executable='anygrasp_positioning',output='screen',emulate_tty=True,parameters=[config.to_dict(),{
  'candidate_file':value('candidate_file'),'report_file':value('report_file'),
  'state_source':'live','execute':value('execute').lower()=='true','operator_reviewed':value('operator_reviewed').lower()=='true'}])]

def generate_launch_description():
 return LaunchDescription([DeclareLaunchArgument('candidate_file'),DeclareLaunchArgument('report_file',default_value='/tmp/anygrasp_positioning_report.yaml'),
  DeclareLaunchArgument('execute',default_value='false',choices=['true','false']),DeclareLaunchArgument('operator_reviewed',default_value='false',choices=['true','false']),OpaqueFunction(function=nodes)])
