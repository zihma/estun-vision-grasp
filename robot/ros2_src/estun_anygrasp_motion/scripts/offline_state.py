#!/usr/bin/python3
"""Synthetic offline state on a dedicated topic; no robot command APIs."""
import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from sensor_msgs.msg import JointState

def main():
 rclpy.init()
 node=Node('anygrasp_offline_state')
 joints=node.declare_parameter('initial_joints', [0.0]*6).value
 if len(joints)!=6: raise ValueError('Six offline joints required')
 pub=node.create_publisher(JointState, '/anygrasp_offline/joint_states',10)
 def tick():
  m=JointState();m.header.stamp=node.get_clock().now().to_msg()
  m.name=['joint'+str(i) for i in range(1,7)];m.position=list(joints);pub.publish(m)
 node.create_timer(.05,tick)
 node.get_logger().warning('OFFLINE synthetic joint feedback only; no physical robot state')
 try:
  rclpy.spin(node)
 except (KeyboardInterrupt, ExternalShutdownException):
  pass
 finally:
  node.destroy_node()
  if rclpy.ok(): rclpy.shutdown()
if __name__=='__main__':main()
