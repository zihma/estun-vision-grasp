# 泡沫砖视觉方向抓取（2026-10-09）

使用 group1 模型识别单块平放的 `Foam_Brick`，通过顶面深度估计中心及短轴。
夹爪闭合方向固定为 tool0 的 +X 朝 +Y 旋转 45°，沿用已测量的 TCP。
目标中心是画面中的白点，闭合方向是紫色短轴。砖块周围需露出桌面。

## 1. 启动机器人与夹爪

在机械臂停止、上一轮执行程序结束后，退出旧相机程序（预览窗口按 Q）。
如果原 bringup 已使用下面的参数运行，可继续使用，勿重复启动。
否则关闭旧 bringup，再在终端 A 执行：

```bash
source /opt/ros/jazzy/setup.bash
source /opt/robot_ws/install/setup.bash
ros2 launch estun_vision_grasp vision_grasp_bringup.launch.py mode:=grasp start_camera:=false start_gripper:=true start_grasp_executor:=false
```

## 2. 启动方向目标

终端 B：

```bash
/opt/conda/envs/vision_env/bin/python3 /opt/robot_ws/src/estun_vision_grasp/scripts/brick_orientation_preview.py --publish-grasp-target
```

保持窗口运行，确认 `TARGET_READY`、黄色长轴及紫色短轴与实物一致。
此模式使用当前帧有效几何，不要求连续 8 帧稳定。几何不合格时仍拒绝目标。
只运行一套相机程序。默认模型为 `models/competition/group1/best.pt`。

## 3. 抓取

控制器切换到 AutoReady，故障码为 0。终端 C：

```bash
source /opt/ros/jazzy/setup.bash
source /opt/robot_ws/install/setup.bash
```

如需回到已验证的 Gripper Ready，在确认回程路径可用后执行：

```bash
ros2 run estun_vision_grasp return_to_gripper_ready --ros-args -p execute:=true -p max_joint_delta:=2.00 -p velocity_scaling:=0.03 -p acceleration_scaling:=0.03
```

按提示输入 `GRIPPER_READY`。已在该准备位可跳过。

放稳砖块，保持相机视野无遮挡，执行完整抓取：

```bash
ros2 run estun_vision_grasp foam_brick_grasp --ros-args -p use_visual_orientation:=true -p execute_visual_grasp:=true
```

检查 RViz 路径及现场空间后，按提示输入 `GRASP`。执行期间不要移动砖块。
程序到达砖块上方 150 mm 后，沿用现有下降、闭合、抬升和返回 Gripper Ready 流程，最后保持夹紧。
如只需规划，省略 `-p execute_visual_grasp:=true`。

## 行为与检查

- 仅在闭合轴对准短轴的两个 180° 等价姿态中选择关节变化较小的解。
  无解时退出，不退回任意旋转角。因 tool0 朝下，tool0 X 的基坐标水平角为短轴角加 45°；不是直接给 joint6 加角度。
- 使用专用文件 `/tmp/estun_brick_oriented_target_base.txt`，不读取旧的无方向目标作为后备。
  无检测、几何无效、无图像或退出都会使该目标失效；目标超过 1 秒拒绝执行。
- 确认后重新检查目标，相对规划时移动超过 10 mm 或方向变化超过 5°，旧规划作废，重新运行抓取命令。
- 到达上方及闭合前，根据实际关节状态对应的 tool0 姿态检查闭合轴，误差超过 2°停止后续动作。
  该检查基于机器人模型，夹爪实物仍须按约定的 +45° 固定安装。
- 接近运动保持真机统一指导中的真实起点、0.10 rad 分段规划检查、合并关键点、一次 Execute、执行前后复核。
  下降和抬升沿用现有竖直运动实现。

验证记录：8 项 Python 几何测试、C++ 目标解析/时效/方向测试及 7 组现场快照接口回放通过；
已通过隔离编译，并完成工作空间编译安装。2026-10-09 用户反馈新方向模式实际成功抓取砖块，
本次验证了当前 +45°安装方向和补偿有效；尚未统计多位置、多角度的重复抓取成功率。
