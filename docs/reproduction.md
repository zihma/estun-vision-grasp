# 复现范围

## 已提供的轻量演示

仓库根目录的回放只依赖 NumPy 与 Plotly。输入是真实采集后提取的物品表面点云、历史 AnyGrasp 候选；不运行模型、不发布 ROS 指令、不打开相机。结果 JSON 保留原实验统计，公开点云仅保留目标表面。

## 源码与外部依赖

`offline/` 和 `robot/` 是开发源码的隐私清理快照。现场绝对路径统一改为 `/opt/robot_ws`、`/opt/anygrasp_lab`、`/opt/graspnet_lab` 与 `/opt/conda` 示例路径，部署时需按自己的目录配置。这些目录不是自动安装结果。

完整神经网络推理另需官方 AnyGrasp SDK、合法授权、检测权重、兼容的 PyTorch/CUDA/PointNet2/MinkowskiEngine/GraspNet API 环境。仓库不分发授权文件、模型权重或 vendor 目录。

真机源码另需 ROS 2 Jazzy、MoveIt 2、厂商提供的 `codroid_msgs`/通信驱动、合法取得的 S3-60 Eco URDF/网格与 MoveIt 配置、`step_motor` 夹爪接口、Orbbec 相机 SDK、YOLO 模型和独立手眼标定。机器人模型、厂商驱动和训练数据没有包含在本发布包。

公开的标定配置为 `calibrated: false`；原场地外参和采样数据不发布。现场旧 README、launch 与工具脚本作为源码参考保留，具体依赖与实验阶段以本文件和根 README 为准。没有在新机器上验证完整真机部署，不能把源码快照描述为克隆后一键驱动机械臂。

运行硬件应用前必须重新标定并验证 TCP、夹爪几何、控制器状态与现场空间。默认展示命令不含硬件执行步骤。
