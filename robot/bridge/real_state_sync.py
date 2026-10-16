import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from codroid_msgs.msg import RobotInfo  # 假设你的包名是 codroid_msgs

class RealStateSync(Node):
    def __init__(self):
        super().__init__('real_state_sync')
        self.publisher_ = self.create_publisher(JointState, '/joint_states', 10)
        self.subscription = self.create_subscription(
            RobotInfo, '/RobotInfo', self.robot_info_callback, 10)
        self.get_logger().info('真机状态同步节点已启动，正在转发 /RobotInfo 到 /joint_states')

    def robot_info_callback(self, msg):
        # 真机返回的关节顺序是 Joint1 ~ Joint6
        # MoveIt 需要的顺序是 joint1 ~ joint6
        joint_state = JointState()
        joint_state.header.stamp = self.get_clock().now().to_msg()
        joint_state.name = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
        
        # 注意：根据上次对比，真机 Joint2 和 Joint5 的方向与 URDF 相反
        # 因此需要乘以 -1 转换回 URDF 空间
        raw_positions = msg.joint_positions
        signs = [1.0, -1.0, 1.0, 1.0, -1.0, 1.0]
        
        if len(raw_positions) >= 6:
            joint_state.position = [raw_positions[i] * signs[i] for i in range(6)]
            self.publisher_.publish(joint_state)

def main(args=None):
    rclpy.init(args=args)
    node = RealStateSync()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
