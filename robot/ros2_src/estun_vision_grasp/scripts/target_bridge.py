#!/usr/bin/python3
"""Transform Gemini camera targets into the exact MoveIt base_link frame."""

import os
from pathlib import Path
import time

from geometry_msgs.msg import PoseStamped
import numpy as np
import rclpy
from rclpy.node import Node
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


class TargetBridge(Node):
    def __init__(self):
        super().__init__('target_bridge')
        self.declare_parameter('calibration_file', '')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('camera_frame', 'gemini336_color_optical_frame')
        self.declare_parameter('camera_target_file', '/tmp/estun_foam_target_camera.yaml')
        self.declare_parameter('base_target_file', '/tmp/estun_foam_target_base.txt')
        self.declare_parameter('output_topic', '/foam_block/pose_base')
        self.declare_parameter('max_target_age_sec', 0.5)
        self.declare_parameter('minimum_confidence', 0.5)
        self.declare_parameter('target_offset_xyz', [0.0, 0.0, 0.0])

        calibration_path = Path(self.get_parameter('calibration_file').value)
        calibration = yaml.safe_load(calibration_path.read_text(encoding='utf-8')) or {}
        self.base_frame = self.get_parameter('base_frame').value
        self.camera_frame = self.get_parameter('camera_frame').value
        if not calibration.get('calibrated', False):
            raise RuntimeError(f'标定结果未生效：{calibration_path}')
        if (calibration.get('parent_frame') != self.base_frame or
                calibration.get('child_frame') != self.camera_frame):
            raise RuntimeError('标定文件坐标系与目标桥参数不一致')

        self.transform = np.asarray(calibration['matrix_T_base_camera'], dtype=float)
        if self.transform.shape != (4, 4):
            raise RuntimeError('matrix_T_base_camera 必须为 4x4')
        if not np.all(np.isfinite(self.transform)):
            raise RuntimeError('matrix_T_base_camera 包含非有限数值')
        rotation = self.transform[:3, :3]
        orthogonality_error = float(
            np.linalg.norm(rotation.T @ rotation - np.eye(3)))
        determinant = float(np.linalg.det(rotation))
        if orthogonality_error > 1.0e-6 or abs(determinant - 1.0) > 1.0e-6:
            raise RuntimeError(
                'matrix_T_base_camera 旋转非法：'
                f'orth={orthogonality_error:.3e}, det={determinant:.9f}')
        translation = np.asarray(calibration['translation'], dtype=float)
        if not np.allclose(translation, self.transform[:3, 3], atol=1.0e-8):
            raise RuntimeError('标定 translation 与 matrix_T_base_camera 不一致')
        quaternion_rotation = quaternion_matrix(calibration['quaternion_xyzw'])
        if not np.allclose(quaternion_rotation, rotation, atol=1.0e-6):
            raise RuntimeError('标定 quaternion 与 matrix_T_base_camera 不一致')
        self.input_path = Path(self.get_parameter('camera_target_file').value)
        self.output_path = Path(self.get_parameter('base_target_file').value)
        self.max_age = float(self.get_parameter('max_target_age_sec').value)
        self.min_confidence = float(self.get_parameter('minimum_confidence').value)
        self.offset = np.asarray(self.get_parameter('target_offset_xyz').value, dtype=float)
        self.publisher = self.create_publisher(
            PoseStamped, self.get_parameter('output_topic').value, 10)
        self.last_stamp = 0.0
        self.create_timer(0.05, self.update_target)
        self.get_logger().info('目标桥已启用：camera -> base_link')

    def update_target(self):
        try:
            data = yaml.safe_load(self.input_path.read_text(encoding='utf-8')) or {}
            stamp = float(data['stamp'])
            if stamp <= self.last_stamp:
                return
            age = time.time() - stamp
            if age < -0.2 or age > self.max_age:
                return
            if data.get('camera_frame') != self.camera_frame:
                raise RuntimeError(f"目标相机坐标系错误：{data.get('camera_frame')}")
            confidence = float(data['confidence'])
            if confidence < self.min_confidence:
                return
            camera_point = np.array([*data['xyz_camera_m'], 1.0], dtype=float)
            base_point = (self.transform @ camera_point)[:3] + self.offset
            if not np.all(np.isfinite(base_point)):
                raise RuntimeError('变换后目标包含非有限数值')

            message = PoseStamped()
            message.header.stamp = self.get_clock().now().to_msg()
            message.header.frame_id = self.base_frame
            message.pose.position.x = float(base_point[0])
            message.pose.position.y = float(base_point[1])
            message.pose.position.z = float(base_point[2])
            message.pose.orientation.w = 1.0
            self.publisher.publish(message)

            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.output_path.with_suffix('.tmp')
            temporary.write_text(
                f'{stamp:.9f} {self.base_frame} '
                f'{base_point[0]:.9f} {base_point[1]:.9f} {base_point[2]:.9f} '
                f'{confidence:.6f}\n', encoding='utf-8')
            os.replace(temporary, self.output_path)
            self.last_stamp = stamp
        except FileNotFoundError:
            return
        except Exception as error:
            self.get_logger().error(str(error), throttle_duration_sec=2.0)


def main():
    rclpy.init()
    node = TargetBridge()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
