# ESTUN + Gemini 336 手眼标定与泡沫砖抓取

> 本包用于第三届“埃斯顿杯”创新型赛道一（智能抓取应用）。比赛任务背景、现场标定约束和标定后工作流见 [COMPETITION_CONTEXT.md](COMPETITION_CONTEXT.md)。

这个包把 Gemini 336、MoveIt `base_link` 手眼标定、四指夹爪和泡沫砖抓取放在同一目录中。旧的 ESTUN `coord0` 标定矩阵不会被读取。

## 坐标链

新增：[泡沫砖方向预览：固定 +45° 夹爪安装角](docs/brick_orientation_preview.md)。

现场方向验证后：[按砖块短轴进行视觉方向抓取](docs/brick_visual_orientation_grasp.md)。
先验证长短轴识别，不连接运动执行器。

```text
Gemini optical frame
  -- T_base_camera（本包重新标定） --> base_link
  -- MoveIt 规划 --> tool0
```

抓取程序只接受如下格式的目标文件：

```text
timestamp base_link x y z confidence
```

只要 frame 不是精确的 `base_link`，程序就会拒绝运动。

机器人模型中的 `joint6` 轴已经按真机与相机实测修正为 URDF `+Z`。
如果更换机器人模型，标定前必须再次做“只转动 J6”的方向一致性检查。

## 1. 编译

```bash
cd ~/estun_ws
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install --packages-select estun_vision_grasp
source install/setup.bash
```

包的 CMake 会强制使用 `/usr/bin/python3` 构建 ROS 2 部分；Gemini 336 相机进程仍由 launch 单独使用 `vision_env` Conda Python，避免 Python 3.11/3.12 ABI 混用。

## 2. 眼在手外标定

将 `markers/aruco_marker_0.png` 按原比例打印，把 ID 0、黑色方框的真实边长准确测量后固定在 `tool0` 上。示例命令中的 `0.10` 表示黑色方框边长 100 mm，必须改成实际测量值：

```bash
ros2 launch estun_vision_grasp vision_grasp_bringup.launch.py \
  mode:=calibration marker_id:=0 marker_length:=0.10
```

每次把机械臂移动到一个新的、安全且能清楚看到 Marker 的姿态，等待画面稳定后采样：

```bash
ros2 service call /handeye/capture_sample std_srvs/srv/Trigger '{}'
```

建议采集 15–25 组，平移和三个旋转轴都要有明显变化。采样读取的是 MoveIt TF：`base_link -> tool0`，不是 ESTUN 网站中的 `coord0/end_positions`。

求解：

```bash
ros2 run estun_vision_grasp solve_eye_to_hand.py \
  --input ~/estun_ws/src/estun_vision_grasp/data/handeye_samples.yaml \
  --output ~/estun_ws/src/estun_vision_grasp/config/handeye_result.yaml
```

结果必须检查 `translation_rms_m`、`rotation_rms_deg`，并用已知点做独立验证后才能抓取。建议目标：平移 RMS < 10 mm、旋转 RMS < 2°；这只是初始门槛，不替代现场验证。

## 3. 同时启动机械臂、相机、夹爪和目标转换

```bash
ros2 launch estun_vision_grasp vision_grasp_bringup.launch.py mode:=grasp
```

该命令默认启动：

- ESTUN 真机 MoveIt bringup；
- Gemini 336 + YOLO；
- 四指夹爪串口节点；
- `base_link -> gemini336_color_optical_frame` 静态 TF；
- camera → `base_link` 目标转换。

它默认不会自动运动。先检查：

```bash
ros2 topic echo /foam_block/pose_base
ros2 run tf2_ros tf2_echo base_link gemini336_color_optical_frame
```

确认目标位置、TCP、桌面高度和安全空间后，第一次只运行安全定位模式：

```bash
ros2 run estun_vision_grasp foam_brick_grasp --ros-args \
  -p positioning_only:=true \
  -p positioning_clearance:=0.10
```

程序完成规划后仍需准确输入 `POSITION` 才会开始真机动作。它只把虚拟
`tool0` 移动到泡沫砖上方 100 mm，同时约束 `tool0 +Z` 对准
`base_link -Z`（朝向砖块）；到达后立即退出，不下降、不发送夹爪命令。
`positioning_clearance` 允许范围为 0.10～0.58 m；通过范围检查不代表目标必然可达或无碰撞，仍以 IK、MoveIt 规划及现场安全检查为准。
仅定位目标的最大单关节变化默认不得超过 1.0 rad，超出时也会拒绝执行。

### 无夹爪手眼定位验证：裸法兰上方 150 mm

当前 URDF 的 `tool0` 位于实体法兰 `link6` 前方 150 mm。无夹爪时使用
`bare_flange_positioning:=true`，程序根据模型变换把目标换算给 `tool0`，
并在运动后直接核对 `link6` 是否位于泡沫砖检测点正上方 150 mm。
检测点由目标框中心附近的深度中值生成，通常对应可见表面；实际高度仍需现场测量。

先关闭 calibration launch，再启动识别和真机状态，不启动夹爪或自动抓取执行器：

```bash
ros2 launch estun_vision_grasp vision_grasp_bringup.launch.py \
  mode:=grasp start_gripper:=false start_grasp_executor:=false
```

新终端加载环境后，先只规划：

```bash
ros2 run estun_vision_grasp foam_brick_grasp --ros-args \
  -p positioning_only:=true \
  -p bare_flange_positioning:=true \
  -p positioning_clearance:=0.15 \
  -p positioning_max_joint_delta:=2.00
```

确认视觉目标和每一段 MoveIt 规划结果、RViz 与现场路径后，再显式允许执行：

```bash
ros2 run estun_vision_grasp foam_brick_grasp --ros-args \
  -p positioning_only:=true \
  -p bare_flange_positioning:=true \
  -p positioning_clearance:=0.15 \
  -p positioning_max_joint_delta:=2.00 \
  -p execute_positioning:=true
```

执行命令仍需人工输入 `POSITION`。程序从当前真实关节状态出发，最多允许
2.00 rad 总单关节变化，每 0.10 rad 单独通过 MoveIt 规划检查，合并逻辑点后
只执行一次，并核对真实关节终点、`tool0` 和裸法兰 `link6` 的位置。
任一检查失败就停止，不会自动改用其他路径。任意示教姿态不等于任意姿态都可达；
超过限值、IK 失败或路径有碰撞风险时，应改用安全起点，不能扩大限值强行执行。

只有在定位、TCP、标定矩阵和安全空间都独立验证完成，并且重新装好夹爪后，
才能运行完整抓取；完整抓取会要求输入 `GRASP`。也可以显式让 bringup 启动执行节点：

```bash
ros2 launch estun_vision_grasp vision_grasp_bringup.launch.py \
  mode:=grasp start_grasp_executor:=true
```

完整流程依次为：以统一的 0.10 rad 逻辑分段合并轨迹到砖块上方 150 mm，
张开夹爪，沿 `base_link -Z` 每段最多 20 mm 下降 150 mm，闭合夹爪，
沿 `base_link +Z` 每段最多 20 mm 上升 150 mm，最后再次使用 0.10 rad
逻辑分段合并轨迹返回 `Gripper Ready`。任何抬升后或返回过程的错误都会保持
夹爪闭合并停止，不会自动释放砖块。

真机速度方面，到砖块上方及返回 Ready 继续使用桥接层默认的 `0.10`；只有
垂直下降和抬升显式请求 `0.20` 速度与加速度。夹爪电机仍使用原先已经验证的
`speed=200`，没有随机械臂提速。

## 返回 Gripper Ready

抓取流程使用同一组 `Gripper Ready` 关节目标。先只规划检查：

```bash
ros2 run estun_vision_grasp return_to_gripper_ready
```

规划成功并检查路径安全后，才允许低速执行：

```bash
ros2 run estun_vision_grasp return_to_gripper_ready --ros-args \
  -p execute:=true
```

程序默认规划速度和加速度缩放均为 3%，最大单关节变化保护值为 1.50 rad，
执行前必须准确输入 `GRIPPER_READY`。程序沿用已经真机验证的 blended 方法：
按照最大单关节变化每 0.10 rad 生成一个逻辑关键点，逐段进行 MoveIt 规划检查，
然后把 Current 和全部逻辑关键点组合成一次连续 RobotMove。

以后由自动抓取状态机调用时，可以关闭人工口令，但只能在状态机已经完成环境、
故障状态和运动互锁检查后使用：

```bash
ros2 run estun_vision_grasp return_to_gripper_ready --ros-args \
  -p execute:=true \
  -p require_confirmation:=false
```

## 4. 抓取高度修正

YOLO + 深度得到的是泡沫砖可见表面中心，不一定等于夹爪 TCP 应到达的位置。先在 `config/system.yaml` 的 `target_offset_xyz` 中设置安全的 `base_link` 偏移；第一轮必须从较高的 Z 偏移开始，只做 TCP 定位验证，再逐步接近。

## 安全边界

- 总启动不会自动执行抓取。
- `handeye_result.yaml` 中 `calibrated: false` 时，抓取坐标链会拒绝启动。
- 相机目标超过 0.5 秒会被目标桥丢弃；抓取程序再次限制为 1 秒。
- 抓取大范围运动沿用已验证的 0.10 rad 分段检查和一次连续 RobotMove。
- 当前抓取执行程序是“准备文件”，必须经过空载定位、桌面高度、TCP 方向和急停测试后才能用于真实砖块。
- 标定结果若通过人工筛选样本得到，必须先使用 `positioning_only` 做高位验证，不能直接完整抓取。
