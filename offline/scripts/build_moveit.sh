#!/bin/bash
set -eo pipefail
source /opt/ros/jazzy/setup.bash
source /opt/robot_ws/install/setup.bash
LAB="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
colcon --log-base "$LAB/log_ros" build --base-paths "$LAB/src/estun_anygrasp_motion" --build-base "$LAB/build_ros" --install-base "$LAB/install_ros" --packages-select estun_anygrasp_motion --cmake-args -DCMAKE_BUILD_TYPE=Release
ctest --test-dir "$LAB/build_ros/estun_anygrasp_motion" --output-on-failure
