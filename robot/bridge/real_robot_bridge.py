import time
import threading

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup

from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
from codroid_msgs.msg import RobotInfo


class RealRobotBridge(Node):

    def __init__(self):
        super().__init__('real_robot_bridge')

        # ---------------------------------------------------------
        # 回调组
        # Action 执行会等待真机状态，因此必须允许 RobotInfo 回调并行运行
        # ---------------------------------------------------------
        self.action_group = MutuallyExclusiveCallbackGroup()
        self.state_group = MutuallyExclusiveCallbackGroup()

        # 保留原真机入口为默认值；统一 bringup 可通过参数改为 arm_controller。
        self.action_name = self.declare_parameter(
            'action_name',
            'real_arm_controller/follow_joint_trajectory'
        ).value

        # ---------------------------------------------------------
        # 发布给真机
        # ---------------------------------------------------------
        self.publisher_ = self.create_publisher(
            JointTrajectory,
            '/RobotMove',
            10
        )

        # ---------------------------------------------------------
        # MoveIt FollowJointTrajectory Action Server
        # ---------------------------------------------------------
        self._action_server = ActionServer(
            self,
            FollowJointTrajectory,
            self.action_name,
            self.execute_callback,
            callback_group=self.action_group
        )

        # ---------------------------------------------------------
        # MoveIt 当前关节状态
        # ---------------------------------------------------------
        self.joint_states = None

        self.joint_state_subscription = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10,
            callback_group=self.state_group
        )

        # ---------------------------------------------------------
        # 真机 RobotInfo
        # ---------------------------------------------------------
        self.robot_state = None
        self.robot_fault_flag = None
        self.robot_info_received = False

        self.state_condition = threading.Condition()

        self.robot_info_subscription = self.create_subscription(
            RobotInfo,
            '/RobotInfo',
            self.robot_info_callback,
            10,
            callback_group=self.state_group
        )

        # ---------------------------------------------------------
        # 安全限制
        # ---------------------------------------------------------
        self.MAX_VEL = 0.20
        self.MAX_ACC = 0.20
        self.MAX_DELTA = 0.5

        # ESTUN RobotMove 真机速度 / 加速度
        self.REAL_VEL = 0.10
        self.REAL_ACC = 0.10

        # 状态等待时间
        self.START_TIMEOUT = 5.0
        self.FINISH_TIMEOUT = 60.0

        # 最终位置核对容差
        self.GOAL_TOLERANCE = 0.03   # rad，约 1.72°

        self.get_logger().info(
            '真机桥接节点已启动，等待 MoveIt 轨迹...'
        )

    # =============================================================
    # 状态回调
    # =============================================================

    def joint_state_callback(self, msg):
        self.joint_states = msg

    def robot_info_callback(self, msg):

        with self.state_condition:

            old_state = self.robot_state

            self.robot_state = msg.state
            self.robot_fault_flag = msg.fault_flag
            self.robot_info_received = True

            # 只在状态发生变化时打印，避免刷屏
            if old_state != self.robot_state:
                self.get_logger().info(
                    f'真机状态变化: {old_state} -> {self.robot_state}'
                )

            self.state_condition.notify_all()

    # =============================================================
    # 等待真机进入某状态
    # =============================================================

    def wait_for_state(self, target_state, timeout):

        deadline = time.monotonic() + timeout

        with self.state_condition:

            while True:

                # 真机故障
                if (
                    self.robot_fault_flag is not None
                    and self.robot_fault_flag != 0
                ):
                    return False, (
                        f'真机 fault_flag={self.robot_fault_flag}'
                    )

                # 到达目标状态
                if self.robot_state == target_state:
                    return True, ''

                remaining = deadline - time.monotonic()

                if remaining <= 0.0:
                    return False, (
                        f'等待 {target_state} 超时，'
                        f'当前状态={self.robot_state}'
                    )

                self.state_condition.wait(
                    timeout=min(0.1, remaining)
                )

    # =============================================================
    # 最终位置检查
    # =============================================================

    def check_final_position(self, trajectory):

        if self.joint_states is None:
            return False, '没有 /joint_states，无法核对最终位置'

        final_point = trajectory.points[-1]

        name_to_idx = {
            name: i
            for i, name in enumerate(trajectory.joint_names)
        }

        max_error = 0.0
        max_error_joint = ''

        for name in trajectory.joint_names:

            if name not in self.joint_states.name:
                return False, f'/joint_states 缺少 {name}'

            trajectory_idx = name_to_idx[name]
            current_idx = self.joint_states.name.index(name)

            target = final_point.positions[trajectory_idx]
            actual = self.joint_states.position[current_idx]

            error = abs(target - actual)

            self.get_logger().info(
                f'FINAL CHECK {name}: '
                f'target={target:.9f}, '
                f'actual={actual:.9f}, '
                f'error={error:.9f}'
            )

            if error > max_error:
                max_error = error
                max_error_joint = name

        self.get_logger().info(
            f'最终位置最大误差: '
            f'{max_error:.6f} rad ({max_error_joint})'
        )

        if max_error > self.GOAL_TOLERANCE:
            return False, (
                f'最终位置误差超限: '
                f'{max_error:.6f} > {self.GOAL_TOLERANCE}'
            )

        return True, ''

    # =============================================================
    # MoveIt Execute
    # =============================================================

    def execute_callback(self, goal_handle):

        self.get_logger().info(
            '收到 MoveIt 轨迹，开始安全检查...'
        )

        trajectory = goal_handle.request.trajectory

        urdf_names = [
            'joint1',
            'joint2',
            'joint3',
            'joint4',
            'joint5',
            'joint6'
        ]

        real_names = [
            'Joint1',
            'Joint2',
            'Joint3',
            'Joint4',
            'Joint5',
            'Joint6'
        ]

        signs = [
            1.0,
            -1.0,
            1.0,
            1.0,
            -1.0,
            1.0
        ]

        # ---------------------------------------------------------
        # 0. 基本检查
        # ---------------------------------------------------------

        if not trajectory.points:
            self.get_logger().error('收到空轨迹')
            goal_handle.abort()
            return FollowJointTrajectory.Result()

        if set(trajectory.joint_names) != set(urdf_names):
            self.get_logger().error(
                f'关节名不匹配! 收到: {trajectory.joint_names}'
            )
            goal_handle.abort()
            return FollowJointTrajectory.Result()

        if self.joint_states is None:
            self.get_logger().error(
                '尚未收到 /joint_states，拒绝执行'
            )
            goal_handle.abort()
            return FollowJointTrajectory.Result()

        if not self.robot_info_received:
            self.get_logger().error(
                '尚未收到 /RobotInfo，拒绝执行'
            )
            goal_handle.abort()
            return FollowJointTrajectory.Result()

        if self.robot_fault_flag != 0:
            self.get_logger().error(
                f'真机 fault_flag={self.robot_fault_flag}，拒绝执行'
            )
            goal_handle.abort()
            return FollowJointTrajectory.Result()

        if self.robot_state != 'AutoReady':
            self.get_logger().error(
                f'真机当前不是 AutoReady，而是 '
                f'{self.robot_state}，拒绝执行'
            )
            goal_handle.abort()
            return FollowJointTrajectory.Result()

        # ---------------------------------------------------------
        # 将收到的关节顺序映射为固定 URDF 顺序
        # ---------------------------------------------------------

        name_to_idx = {
            name: i
            for i, name in enumerate(trajectory.joint_names)
        }

        # 执行开始前固定一份当前位置快照
        current_msg = self.joint_states

        current_positions = {}

        for name in urdf_names:

            if name not in current_msg.name:
                self.get_logger().error(
                    f'/joint_states 缺少 {name}'
                )
                goal_handle.abort()
                return FollowJointTrajectory.Result()

            idx = current_msg.name.index(name)
            current_positions[name] = current_msg.position[idx]

        # ---------------------------------------------------------
        # 建立真机轨迹 + 安全检查
        # ---------------------------------------------------------

        real_traj = JointTrajectory()
        real_traj.joint_names = real_names

        # 默认继续使用已经验证的 0.10。调用端只有在轨迹中明确给出
        # 更高的速度/加速度时，才对本次 RobotMove 临时采用该请求；
        # 请求仍受 MAX_VEL / MAX_ACC 的 0.20 上限保护，不改变后续轨迹。
        requested_real_vel = max(
            (
                abs(value)
                for point in trajectory.points
                for value in point.velocities
            ),
            default=0.0,
        )
        requested_real_acc = max(
            (
                abs(value)
                for point in trajectory.points
                for value in point.accelerations
            ),
            default=0.0,
        )

        command_real_vel = self.REAL_VEL
        command_real_acc = self.REAL_ACC

        if requested_real_vel > self.REAL_VEL:
            command_real_vel = min(requested_real_vel, self.MAX_VEL)

        if requested_real_acc > self.REAL_ACC:
            command_real_acc = min(requested_real_acc, self.MAX_ACC)

        self.get_logger().info(
            f'本次 RobotMove 真机速度={command_real_vel:.3f}, '
            f'加速度={command_real_acc:.3f}'
        )

        # Blended trajectory:
        # MAX_DELTA 检查相邻 waypoint，而不是所有点相对最初 Current。
        previous_positions = current_positions.copy()

        for point in trajectory.points:

            if len(point.positions) != 6:
                self.get_logger().error(
                    f'轨迹点 positions 数量异常: '
                    f'{len(point.positions)}'
                )
                goal_handle.abort()
                return FollowJointTrajectory.Result()

            real_point = JointTrajectoryPoint()

            pos = []
            moveit_vel = []
            moveit_acc = []

            for j, name in enumerate(urdf_names):

                idx = name_to_idx[name]

                # MoveIt 坐标 -> 真机坐标
                pos.append(
                    point.positions[idx] * signs[j]
                )

                if point.velocities:
                    moveit_vel.append(
                        point.velocities[idx]
                    )
                else:
                    moveit_vel.append(0.0)

                if point.accelerations:
                    moveit_acc.append(
                        point.accelerations[idx]
                    )
                else:
                    moveit_acc.append(0.0)

                # 相对于执行开始时真实当前位置检查整条轨迹
                delta = abs(
                    point.positions[idx]
                    - previous_positions[name]
                )

                if delta > self.MAX_DELTA:
                    self.get_logger().error(
                        f'{name} 增量超限! '
                        f'{delta} > {self.MAX_DELTA}'
                    )
                    goal_handle.abort()
                    return FollowJointTrajectory.Result()

            # MoveIt 速度检查
            for v in moveit_vel:

                if abs(v) > self.MAX_VEL + 0.001:
                    self.get_logger().error(
                        f'速度超限! '
                        f'{abs(v)} > {self.MAX_VEL}'
                    )
                    goal_handle.abort()
                    return FollowJointTrajectory.Result()

            # MoveIt 加速度检查
            for a in moveit_acc:

                if abs(a) > self.MAX_ACC + 0.001:
                    self.get_logger().error(
                        f'加速度超限! '
                        f'{abs(a)} > {self.MAX_ACC}'
                    )
                    goal_handle.abort()
                    return FollowJointTrajectory.Result()

            # ESTUN RobotMove 协议：
            # positions 与 joint_names 数量一致
            # velocities / accelerations 各只有一个值
            real_point.positions = pos
            real_point.velocities = [command_real_vel]
            real_point.accelerations = [command_real_acc]

            real_traj.points.append(real_point)

            # 本 waypoint 通过安全检查后，
            # 作为下一个 waypoint 的比较起点。
            for name in urdf_names:
                idx = name_to_idx[name]
                previous_positions[name] = point.positions[idx]

        # ---------------------------------------------------------
        # 发送给真机
        # ---------------------------------------------------------

        self.publisher_.publish(real_traj)

        self.get_logger().info(
            '轨迹已安全发送至 /RobotMove'
        )

        # ---------------------------------------------------------
        # 等待真机真正开始
        # ---------------------------------------------------------

        self.get_logger().info(
            '等待真机进入 AutoRunning...'
        )

        ok, reason = self.wait_for_state(
            'AutoRunning',
            self.START_TIMEOUT
        )

        if not ok:
            self.get_logger().error(
                f'真机未开始执行: {reason}'
            )
            goal_handle.abort()
            return FollowJointTrajectory.Result()

        self.get_logger().info(
            '真机已进入 AutoRunning，开始执行'
        )

        # ---------------------------------------------------------
        # 等待真机真正完成
        # ---------------------------------------------------------

        self.get_logger().info(
            '等待真机执行完成并返回 AutoReady...'
        )

        ok, reason = self.wait_for_state(
            'AutoReady',
            self.FINISH_TIMEOUT
        )

        if not ok:
            self.get_logger().error(
                f'真机执行完成等待失败: {reason}'
            )
            goal_handle.abort()
            return FollowJointTrajectory.Result()

        self.get_logger().info(
            '真机已返回 AutoReady'
        )

        # 给 /joint_states 很短时间同步最新 RobotInfo
        time.sleep(0.15)

        # ---------------------------------------------------------
        # 最终位置核对
        # ---------------------------------------------------------

        ok, reason = self.check_final_position(
            trajectory
        )

        if not ok:
            self.get_logger().error(reason)
            goal_handle.abort()
            return FollowJointTrajectory.Result()

        # ---------------------------------------------------------
        # 真正完成
        # ---------------------------------------------------------

        self.get_logger().info(
            '真机执行完成，最终位置检查通过 ✅'
        )

        goal_handle.succeed()

        return FollowJointTrajectory.Result()


def main(args=None):

    rclpy.init(args=args)

    bridge = RealRobotBridge()

    # 必须使用多线程 Executor：
    # execute_callback 等待时，
    # /RobotInfo 和 /joint_states 仍然需要继续更新
    executor = MultiThreadedExecutor(
        num_threads=3
    )

    executor.add_node(bridge)

    try:
        executor.spin()

    except KeyboardInterrupt:
        pass

    finally:
        executor.shutdown()
        bridge.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
