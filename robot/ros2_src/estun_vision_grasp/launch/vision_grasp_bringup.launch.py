import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


PACKAGE = 'estun_vision_grasp'


def enabled(context, name):
    return LaunchConfiguration(name).perform(context).lower() in ('1', 'true', 'yes', 'on')


def launch_setup(context):
    mode = LaunchConfiguration('mode').perform(context)
    package_share = get_package_share_directory(PACKAGE)
    workspace_root = os.path.abspath(os.path.expanduser(
        LaunchConfiguration('workspace_root').perform(context)))
    source_root = os.path.join(workspace_root, 'src', PACKAGE)
    calibration_file = LaunchConfiguration('calibration_file').perform(context)
    samples_file = LaunchConfiguration('samples_file').perform(context)
    actions = []

    if enabled(context, 'start_robot'):
        robot_launch = os.path.join(
            get_package_share_directory('estun_s3_60_eco_moveit_config'),
            'launch', 'bringup.launch.py')
        actions.append(IncludeLaunchDescription(
            PythonLaunchDescriptionSource(robot_launch),
            launch_arguments={
                'mode': 'real',
                'use_rviz': LaunchConfiguration('use_rviz'),
                'workspace_root': workspace_root,
            }.items()))

    if enabled(context, 'start_gripper'):
        actions.append(Node(
            package='step_motor',
            executable='motor_node',
            name='gripper_motor',
            output='screen',
            parameters=[{
                'usart_port_name': LaunchConfiguration('gripper_port'),
                'serial_baud_rate': 115200,
            }]))

    conda_python = LaunchConfiguration('gemini_python').perform(context)
    show_flag = ['--show'] if enabled(context, 'show_camera') else []
    script_root = os.path.join(package_share, 'scripts')

    if mode == 'calibration':
        if enabled(context, 'start_camera'):
            actions.append(ExecuteProcess(
                cmd=[conda_python, '-u', os.path.join(script_root, 'gemini_aruco_tracker.py'),
                     '--output', '/tmp/estun_handeye_marker.yaml',
                     '--marker-id', LaunchConfiguration('marker_id'),
                     '--marker-length', LaunchConfiguration('marker_length')] + show_flag,
                name='gemini_aruco_tracker', output='screen',
                additional_env={'YOLO_CONFIG_DIR': '/tmp/estun_yolo_config'}))
        actions.append(Node(
            package=PACKAGE,
            executable='handeye_collector.py',
            output='screen',
            parameters=[os.path.join(package_share, 'config', 'system.yaml'), {
                'samples_file': samples_file,
            }]))
    elif mode == 'grasp':
        if enabled(context, 'start_camera'):
            actions.append(ExecuteProcess(
                cmd=[conda_python, '-u', os.path.join(script_root, 'gemini_foam_detector.py'),
                     '--model', os.path.join(package_share, 'models', 'best.pt'),
                     '--output', '/tmp/estun_foam_target_camera.yaml',
                     '--confidence', LaunchConfiguration('detector_confidence')] + show_flag,
                name='gemini_foam_detector', output='screen',
                additional_env={'YOLO_CONFIG_DIR': '/tmp/estun_yolo_config'}))
        actions.extend([
            Node(
                package=PACKAGE,
                executable='calibration_tf_publisher.py',
                output='screen',
                parameters=[{'calibration_file': calibration_file}]),
            Node(
                package=PACKAGE,
                executable='target_bridge.py',
                output='screen',
                parameters=[os.path.join(package_share, 'config', 'system.yaml'), {
                    'calibration_file': calibration_file,
                }]),
        ])
        if enabled(context, 'start_grasp_executor'):
            actions.append(Node(
                package=PACKAGE,
                executable='foam_brick_grasp',
                output='screen'))
    else:
        raise RuntimeError(f'Unsupported mode: {mode}')

    return actions


def generate_launch_description():
    default_workspace = os.path.expanduser('~/estun_ws')
    default_source = os.path.join(default_workspace, 'src', PACKAGE)
    return LaunchDescription([
        DeclareLaunchArgument('mode', default_value='calibration', choices=['calibration', 'grasp']),
        DeclareLaunchArgument('workspace_root', default_value=default_workspace),
        DeclareLaunchArgument('start_robot', default_value='true', choices=['true', 'false']),
        DeclareLaunchArgument('start_gripper', default_value='true', choices=['true', 'false']),
        DeclareLaunchArgument('start_camera', default_value='true', choices=['true', 'false']),
        DeclareLaunchArgument('start_grasp_executor', default_value='false', choices=['true', 'false']),
        DeclareLaunchArgument('use_rviz', default_value='true', choices=['true', 'false']),
        DeclareLaunchArgument('show_camera', default_value='true', choices=['true', 'false']),
        DeclareLaunchArgument('gripper_port', default_value='/dev/ttyACM0'),
        DeclareLaunchArgument(
            'gemini_python',
            default_value='/opt/conda/envs/vision_env/bin/python3'),
        DeclareLaunchArgument('marker_id', default_value='0'),
        DeclareLaunchArgument('marker_length', default_value='0.10'),
        DeclareLaunchArgument('detector_confidence', default_value='0.50'),
        DeclareLaunchArgument(
            'samples_file',
            default_value=os.path.join(default_source, 'data', 'handeye_samples.yaml')),
        DeclareLaunchArgument(
            'calibration_file',
            default_value=os.path.join(default_source, 'config', 'handeye_result.yaml')),
        OpaqueFunction(function=launch_setup),
    ])
