cmake_minimum_required(VERSION 3.8)

# The interactive shell uses the Python 3.11 vision_env Conda environment,
# while ROS 2 Jazzy on this machine is built for system Python 3.12.
set(Python3_EXECUTABLE "/usr/bin/python3" CACHE FILEPATH "ROS 2 system Python")

project(estun_vision_grasp)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(codroid_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(moveit_msgs REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED)
find_package(step_motor REQUIRED)
find_package(trajectory_msgs REQUIRED)

add_executable(foam_brick_grasp src/foam_brick_grasp.cpp)
ament_target_dependencies(
  foam_brick_grasp
  rclcpp
  codroid_msgs
  geometry_msgs
  moveit_msgs
  moveit_ros_planning_interface
  step_motor
)

add_executable(
  return_to_gripper_ready
  src/return_to_gripper_ready.cpp
)
ament_target_dependencies(
  return_to_gripper_ready
  rclcpp
  moveit_ros_planning_interface
  trajectory_msgs
)

install(
  TARGETS
    foam_brick_grasp
    return_to_gripper_ready
  DESTINATION lib/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/calibration_tf_publisher.py
  scripts/handeye_collector.py
  scripts/solve_eye_to_hand.py
  scripts/target_bridge.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch config models markers DESTINATION share/${PROJECT_NAME})

# These two programs run with the vision_env Conda interpreter selected by launch.
install(FILES
  scripts/gemini336_camera.py
  scripts/gemini_aruco_tracker.py
  scripts/gemini_foam_detector.py
  DESTINATION share/${PROJECT_NAME}/scripts
)

ament_package()
