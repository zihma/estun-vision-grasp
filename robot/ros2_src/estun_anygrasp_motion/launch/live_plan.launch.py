"""Planner only, for an already running MoveIt; does not start a robot bridge."""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument,OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder

def nodes(context):
 config=(MoveItConfigsBuilder('S3-60 Eco 2024',package_name='estun_s3_60_eco_moveit_config')
  .robot_description(file_path='/opt/robot_ws/src/estun_s3_60_eco_description/urdf/s3_60_eco.urdf').to_moveit_configs())
 return [Node(package='estun_anygrasp_motion',executable='anygrasp_pregrasp',output='screen',parameters=[config.to_dict(),{
  'candidate_file':LaunchConfiguration('candidate_file').perform(context),
  'report_file':LaunchConfiguration('report_file').perform(context),'state_source':'live','execute':False}])]

def generate_launch_description():
 return LaunchDescription([DeclareLaunchArgument('candidate_file'),DeclareLaunchArgument('report_file',default_value='/tmp/anygrasp_moveit_live_plan.yaml'),OpaqueFunction(function=nodes)])
