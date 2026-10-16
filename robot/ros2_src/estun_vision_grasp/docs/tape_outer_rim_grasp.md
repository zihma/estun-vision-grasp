# 胶带卷左侧外圈夹取

2026-10-09 现场验证：单卷 `Scotch_Duct_Tape` 平放时，左侧外圈定位、径向闭合轴高位复核及完整夹取成功；胶带被夹起并带回 Gripper Ready。这里只记录一次成功，不代表其他位置、尺寸或摆放方式已验证。

## 目标选择

第 2 组模型识别胶带。检测框中心位于胶带中孔，深度可能来自桌面，因此不能直接用作抓取点。`--target-point tape-left-rim` 在左侧外圈的实体顶面选点；要求该处高于中孔底部至少 20 mm、深度区域连续有效。方向轴取胶带中心到该实体点的径向线。夹爪固定安装方向为 tool0 局部 +X 朝 +Y 旋转 45°；执行器只规划闭合轴沿径向线的两个等价姿态。

现场快照 `/tmp/estun_tape_snapshot.npz` 上，左侧目标像素约 `(285,196)`，胶带外圈顶面比中孔底部高约 45 mm。快照在 `/tmp`，不作为长期归档。换场景时仍需重新看实时画面。

## 运行

机器人停止后启动链路；已有同参数链路时不要重复启动。终端 A：

```bash
source /opt/ros/jazzy/setup.bash
source /opt/robot_ws/install/setup.bash
ros2 launch estun_vision_grasp vision_grasp_bringup.launch.py mode:=grasp start_camera:=false start_gripper:=true start_grasp_executor:=false
```

终端 B 使用源码脚本：

```bash
/opt/conda/envs/vision_env/bin/python3 \
  /opt/robot_ws/src/estun_vision_grasp/scripts/gemini_foam_detector.py \
  --model /opt/robot_ws/src/estun_vision_grasp/models/competition/group2/best.pt \
  --class-name Scotch_Duct_Tape \
  --target-point tape-left-rim \
  --output /tmp/estun_foam_target_camera.yaml \
  --confidence 0.50 --show
```

画面中红点必须持续落在左侧外圈实体上，紫线必须穿过胶带中心。检测无效或退出时，专用方向目标 `/tmp/estun_tape_oriented_target_base.txt` 会写成 `INVALID`。不要同时运行另一套相机程序。

终端 C 使用**已编译本版本**的 `estun_vision_grasp`。先将机器人置 AutoReady；不在 Gripper Ready 时，检查路径后执行 `return_to_gripper_ready` 并输入 `GRIPPER_READY`。第一次测试先到外圈上方 150 mm：

```bash
ros2 run estun_vision_grasp foam_brick_grasp --ros-args \
  -p use_tape_outer_orientation:=true \
  -p positioning_only:=true \
  -p positioning_clearance:=0.15 \
  -p positioning_max_joint_delta:=2.00
```

核对规划路径后输入 `POSITION`。到位时，日志中的闭合轴误差应不超过 2°，现场检查 TCP 是否位于外圈点上方、夹指是否能跨住该段实体。此步骤不下降或操作夹爪。

检查通过后先回 Gripper Ready，再运行一次完整夹取，核对路径后输入 `GRASP`：

```bash
ros2 run estun_vision_grasp foam_brick_grasp --ros-args \
  -p use_tape_outer_orientation:=true \
  -p execute_visual_grasp:=true
```

程序沿用砖块版本的 150 mm 下降/抬升及夹爪参数。实际是否夹稳仍需现场观察；停止或目标失效时不要强行继续。胶带在规划确认和执行期间保持不动。夹着胶带回 Ready 后，托住胶带且确认运动程序结束，才可单独发送张开夹爪命令。

若当前工作空间尚未重新编译，可临时在终端 C 先 `source /tmp/estun_tape_build.ordB8A/install/setup.bash` 使用本次隔离编译结果；该 `/tmp` 路径不是长期安装位置。后续在工作空间运行 `colcon build --symlink-install --packages-select estun_vision_grasp` 并重新加载 `install/setup.bash`。
