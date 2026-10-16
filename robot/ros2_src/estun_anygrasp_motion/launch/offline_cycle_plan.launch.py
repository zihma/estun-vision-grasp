from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, RegisterEventHandler, EmitEvent
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder
import yaml

def launch_nodes(context):
 candidates=LaunchConfiguration('candidate_file').perform(context)
 report=LaunchConfiguration('report_file').perform(context)
 with open(candidates) as f:bundle=yaml.safe_load(f)
 config=(MoveItConfigsBuilder('S3-60 Eco 2024', package_name='estun_s3_60_eco_moveit_config')
         .robot_description(file_path='/opt/robot_ws/src/estun_s3_60_eco_description/urdf/s3_60_eco.urdf')
         .planning_pipelines(pipelines=['ompl']).to_moveit_configs())
 remaps=[('/joint_states','/anygrasp_offline/joint_states')]
 planner=Node(package='estun_anygrasp_motion',executable='anygrasp_cycle',output='screen',parameters=[config.to_dict(),{'candidate_file':candidates,'report_file':report,'state_source':'offline','execute':False}],remappings=remaps)
 return [
  Node(package='robot_state_publisher',executable='robot_state_publisher',parameters=[config.robot_description],remappings=remaps),
  Node(package='estun_anygrasp_motion',executable='offline_state.py',parameters=[{'initial_joints':bundle['offline_seed_joints']}]),
  Node(package='moveit_ros_move_group',executable='move_group',output='screen',parameters=[config.to_dict(),{'allow_trajectory_execution':False,'publish_robot_description':True,'publish_robot_description_semantic':True}],remappings=remaps),
  planner,
  RegisterEventHandler(OnProcessExit(target_action=planner,on_exit=[EmitEvent(event=Shutdown(reason='Plan-only node finished'))])),
 ]

def generate_launch_description():
 return LaunchDescription([DeclareLaunchArgument('candidate_file'),DeclareLaunchArgument('report_file',default_value='/tmp/anygrasp_moveit_plan.yaml'),OpaqueFunction(function=launch_nodes)])
