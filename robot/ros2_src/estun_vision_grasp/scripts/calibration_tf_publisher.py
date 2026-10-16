#!/usr/bin/python3
"""Publish the solved base_link -> Gemini336 optical-frame transform."""

from pathlib import Path

from geometry_msgs.msg import TransformStamped
import numpy as np
import rclpy
from rclpy.node import Node
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
import yaml


def quaternion_matrix(quaternion):
    x, y, z, w = np.asarray(quaternion, dtype=float)
    norm = float(x * x + y * y + z * z + w * w)
    if abs(norm - 1.0) > 1.0e-6:
        raise RuntimeError(f'标定四元数未归一化：norm={norm:.9f}')
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=float)


class CalibrationTfPublisher(Node):
    def __init__(self):
        super().__init__('calibration_tf_publisher')
        self.declare_parameter('calibration_file', '')
        path = Path(self.get_parameter('calibration_file').value)
        data = yaml.safe_load(path.read_text(encoding='utf-8')) or {}
        if not data.get('calibrated', False):
            raise RuntimeError(f'手眼标定尚未完成：{path}')
        if data.get('parent_frame') != 'base_link':
            raise RuntimeError('标定结果 parent_frame 必须是 base_link')

        translation = data['translation']
        quaternion = data['quaternion_xyzw']
        matrix = np.asarray(data['matrix_T_base_camera'], dtype=float)
        if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
            raise RuntimeError('matrix_T_base_camera 必须是有限的 4x4 矩阵')
        rotation = matrix[:3, :3]
        orthogonality_error = float(
            np.linalg.norm(rotation.T @ rotation - np.eye(3)))
        determinant = float(np.linalg.det(rotation))
        if orthogonality_error > 1.0e-6 or abs(determinant - 1.0) > 1.0e-6:
            raise RuntimeError(
                'matrix_T_base_camera 旋转非法：'
                f'orth={orthogonality_error:.3e}, det={determinant:.9f}')
        if not np.allclose(np.asarray(translation, dtype=float), matrix[:3, 3], atol=1.0e-8):
            raise RuntimeError('标定 translation 与 matrix_T_base_camera 不一致')
        if not np.allclose(quaternion_matrix(quaternion), rotation, atol=1.0e-6):
            raise RuntimeError('标定 quaternion 与 matrix_T_base_camera 不一致')
        message = TransformStamped()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = data['parent_frame']
        message.child_frame_id = data['child_frame']
        message.transform.translation.x = float(translation[0])
        message.transform.translation.y = float(translation[1])
        message.transform.translation.z = float(translation[2])
        message.transform.rotation.x = float(quaternion[0])
        message.transform.rotation.y = float(quaternion[1])
        message.transform.rotation.z = float(quaternion[2])
        message.transform.rotation.w = float(quaternion[3])
        self.broadcaster = StaticTransformBroadcaster(self)
        self.broadcaster.sendTransform(message)
        self.get_logger().info(
            f"已发布 {message.header.frame_id} -> {message.child_frame_id}")


def main():
    rclpy.init()
    node = CalibrationTfPublisher()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
