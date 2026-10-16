# 泡沫砖方向预览（2026-10-09）

用途：单块泡沫砖平放桌面，先验证方向估计。保留已测量的 TCP，安装闭合轴按
tool0 XY 平面内从 +X 朝 +Y 的 +45°。本程序不连接 ROS、不控制夹爪或机械臂，
不更新 `/tmp/estun_foam_target_camera.yaml` 或 `/tmp/estun_foam_target_base.txt`。

## 启动

1. 机械臂完全停止、抓取执行程序已结束后，关闭占用 Gemini 的旧 detector 或
   整个 grasp launch（原 launch 默认包含 detector）。不要同时运行两套相机程序。
2. 保持相机固定，视野中只放一块平放砖块；砖块四周留出可见桌面，夹爪移出遮挡区域。
3. 直接执行，不需要 ROS 环境、不需要 colcon build：

```bash
/opt/conda/envs/vision_env/bin/python3 /opt/robot_ws/src/estun_vision_grasp/scripts/brick_orientation_preview.py
```

默认加载源码目录中的 `models/competition/group1/best.pt` 和
`config/handeye_result.yaml`，用 CPU 推理。可以通过 `--model` 和 `--calibration`
指定文件。此预览固定识别 `Foam_Brick`，不是所有物品的通用朝向估计。

## 看什么

- 绿色检测框：YOLO 找到的砖块范围。
- 蓝色覆盖区域：实际用于计算的砖块顶面，应覆盖顶面，不应覆盖桌面或夹爪。
- 黄色线：砖块长轴，应沿长边。
- 紫色线：砖块短轴，也是希望夹爪闭合的方向，应跨短边。
- 白点：拟合顶面中心。
- `WAIT_STABLE n/8`：当前几何估计可用，正在等待连续稳定。
- 绿色 `STABLE`：连续 8 帧方向、中心和尺寸变化通过初步门槛；只表示预览稳定，
  不能代替观察轴线与实物是否相符。
- `base XY long/short` 是机器人基坐标系的水平角度，不是图像角度或 joint6 角度。

先在桌面同一位置摆约 0°、30°、60°、90°，每次停稳后查看轴线；再换桌面位置，
重复检查。180°翻转表示同一条长轴，这是正常的；长短轴突然互换 90°不正常。
显示的长宽和顶面高度也应接近实物，用于识别“分割到了背景但角度稳定”的情况。

程序利用检测框外一圈深度估计桌面高度，再从框内寻找高于桌面的连通顶面，
检查平面程度、矩形形状、长宽比和覆盖率，最后在 base_link XY 平面拟合矩形。
第一版假设桌面和砖块顶面近似水平，不适用于倾斜、立放、堆叠、遮挡或多个相邻砖块。
要求顶面与桌面可分离（默认至少 8 mm），不能只根据普通检测框长边推断朝向。

常见拒绝原因：

| 显示 | 检查 |
|---|---|
| NEED_EXACTLY_ONE_BRICK | 只摆一块，确认模型确实检测到 Foam_Brick |
| BRICK_AT_IMAGE_EDGE | 将砖块完整放入画面 |
| RGB_DEPTH_RESOLUTION_MISMATCH | 彩色与深度尺寸不同，需检查实际 D2C 对齐配置 |
| NEED_VISIBLE_TABLE_AROUND_BRICK | 框外需要足够的有效桌面深度 |
| TABLE_NOT_FLAT_OR_BACKGROUND_OCCLUDED | 框外被手、工具、其他物品遮挡，或桌面倾斜 |
| TOP_NOT_SEPARATED_FROM_TABLE | 顶面与桌面深度未分开，或框落到桌面 |
| SIZE_OR_LONG_AXIS_AMBIGUOUS | 提取尺寸不合理或接近正方形，无法可靠区分长短轴 |
| TOP_NOT_HORIZONTAL_OR_DEPTH_NOISY | 顶面倾斜或深度噪声过大 |
| INCOMPLETE_RECTANGULAR_TOP / TOP_HAS_LARGE_HOLES_OR_OCCLUSION | 顶面不完整，需要检查分割、遮挡和反光 |

## 保存现场样本

选中图像窗口后按 **S** 保存当前诊断快照，按 **Q** 退出。
保存路径为 `/tmp/estun_brick_orientation_snapshots/`，每次包含：

- `.png`：带轴线与状态的画面；
- `.npz`：原彩色图、深度、检测框、内参和使用的手眼矩阵。

几何估计失败但已检测到砖块时也能保存，用于离线排查。请在不同角度和位置各保存一次。
回放只运行保存数据的几何估计，不打开相机：

```bash
/opt/conda/envs/vision_env/bin/python3 /opt/robot_ws/src/estun_vision_grasp/scripts/brick_orientation_preview.py --replay /tmp/estun_brick_orientation_snapshots/实际文件名.npz
```

最新诊断数据单独写入 `/tmp/estun_brick_orientation_preview.yaml`，包括稳定状态、
长短轴、尺寸和建议 tool0 旋转矩阵。矩阵按 `R * [cos(45°), sin(45°), 0]`
对齐砖块短轴，同时 `R * [0, 0, 1]` 朝 base_link -Z；不是将 joint6 直接加减45°。
默认预览模式不发布抓取目标。退出时诊断文件标为 `valid: false`。

现场已保存的 7 组图像通过几何回放，用户已确认不同角度的长短轴显示正确。
现在可以使用 `--publish-grasp-target` 发布当前帧有效的中心和短轴，
此模式显示 `TARGET_READY`，不等待连续 8 帧稳定。实际操作见
[视觉方向抓取](brick_visual_orientation_grasp.md)。
