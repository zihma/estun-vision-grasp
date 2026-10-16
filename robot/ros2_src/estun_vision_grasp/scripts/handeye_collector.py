#!/usr/bin/python3
"""Pair Gemini marker poses with MoveIt TF poses for eye-to-hand calibration."""

import math
import os
import time
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformListener
import yaml


def quaternion_matrix(x, y, z, w):
    q = np.array([x, y, z, w], dtype=float)
    norm = float(np.dot(q, q))
    if norm < 1.0e-16:
        raise ValueError('invalid zero quaternion')
    q *= math.sqrt(2.0 / norm)
    q = np.outer(q, q)
    return np.array([
        [1.0 - q[1, 1] - q[2, 2], q[0, 1] - q[2, 3], q[0, 2] + q[1, 3]],
        [q[0, 1] + q[2, 3], 1.0 - q[0, 0] - q[2, 2], q[1, 2] - q[0, 3]],
        [q[0, 2] - q[1, 3], q[1, 2] + q[0, 3], 1.0 - q[0, 0] - q[1, 1]],
    ])


def matrix_from_pose(translation, quaternion):
    matrix = np.eye(4, dtype=float)
    matrix[:3, :3] = quaternion_matrix(*quaternion)
    matrix[:3, 3] = np.asarray(translation, dtype=float)
    return matrix


def rotation_distance_deg(first, second):
    relative = first[:3, :3].T @ second[:3, :3]
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


class HandeyeCollector(Node):
    def __init__(self):
        super().__init__('handeye_collector')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('tool_frame', 'tool0')
        self.declare_parameter('marker_file', '/tmp/estun_handeye_marker.yaml')
        self.declare_parameter('samples_file', '/tmp/estun_handeye_samples.yaml')
        self.declare_parameter('max_marker_age_sec', 0.5)
        self.declare_parameter('minimum_translation_change_m', 0.02)
        self.declare_parameter('minimum_rotation_change_deg', 8.0)

        self.base_frame = self.get_parameter('base_frame').value
        self.tool_frame = self.get_parameter('tool_frame').value
        self.marker_file = Path(self.get_parameter('marker_file').value)
        self.samples_file = Path(self.get_parameter('samples_file').value)
        self.max_age = float(self.get_parameter('max_marker_age_sec').value)
        self.min_translation = float(
            self.get_parameter('minimum_translation_change_m').value)
        self.min_rotation = float(
            self.get_parameter('minimum_rotation_change_deg').value)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.create_service(Trigger, '/handeye/capture_sample', self.capture)
        self.create_service(Trigger, '/handeye/reset_samples', self.reset)
        self.get_logger().info(
            f'采样服务已就绪：/handeye/capture_sample；输出 {self.samples_file}')

    def load_samples(self):
        if not self.samples_file.exists():
            return []
        document = yaml.safe_load(self.samples_file.read_text(encoding='utf-8')) or {}
        return list(document.get('samples', []))

    def save_samples(self, samples):
        self.samples_file.parent.mkdir(parents=True, exist_ok=True)
        document = {
            'calibration_type': 'eye_to_hand',
            'base_frame': self.base_frame,
            'tool_frame': self.tool_frame,
            'camera_frame': 'gemini336_color_optical_frame',
            'convention': {
                'base_to_tool': 'T_base_tool from MoveIt TF',
                'camera_to_marker': 'T_camera_marker from ArUco PnP',
            },
            'samples': samples,
        }
        temporary = self.samples_file.with_suffix('.tmp')
        temporary.write_text(
            yaml.safe_dump(document, sort_keys=False, allow_unicode=True),
            encoding='utf-8')
        os.replace(temporary, self.samples_file)

    def capture(self, _request, response):
        try:
            marker = yaml.safe_load(self.marker_file.read_text(encoding='utf-8'))
            age = time.time() - float(marker['stamp'])
            if age < -0.2 or age > self.max_age:
                raise RuntimeError(f'ArUco 位姿已过期：{age:.3f} s')
            if marker.get('camera_frame') != 'gemini336_color_optical_frame':
                raise RuntimeError(f"相机坐标系不匹配：{marker.get('camera_frame')}")

            tf = self.tf_buffer.lookup_transform(
                self.base_frame, self.tool_frame, Time())
            base_to_tool = matrix_from_pose(
                [tf.transform.translation.x, tf.transform.translation.y,
                 tf.transform.translation.z],
                [tf.transform.rotation.x, tf.transform.rotation.y,
                 tf.transform.rotation.z, tf.transform.rotation.w])
            camera_to_marker = matrix_from_pose(
                marker['translation'], marker['quaternion_xyzw'])

            samples = self.load_samples()
            if samples:
                previous = np.asarray(samples[-1]['base_to_tool'], dtype=float)
                translation_change = float(
                    np.linalg.norm(base_to_tool[:3, 3] - previous[:3, 3]))
                rotation_change = rotation_distance_deg(previous, base_to_tool)
                if (translation_change < self.min_translation and
                        rotation_change < self.min_rotation):
                    raise RuntimeError(
                        '机械臂姿态变化不足：'
                        f'{translation_change:.3f} m, {rotation_change:.1f} deg')

            samples.append({
                'index': len(samples) + 1,
                'stamp': time.time(),
                'marker_id': int(marker['marker_id']),
                'base_to_tool': base_to_tool.tolist(),
                'camera_to_marker': camera_to_marker.tolist(),
            })
            self.save_samples(samples)
            response.success = True
            response.message = f'已保存第 {len(samples)} 组样本'
            self.get_logger().info(response.message)
        except Exception as error:  # service must return a useful diagnostic
            response.success = False
            response.message = str(error)
            self.get_logger().error(response.message)
        return response

    def reset(self, _request, response):
        self.save_samples([])
        response.success = True
        response.message = f'已清空 {self.samples_file}'
        self.get_logger().warning(response.message)
        return response


def main():
    rclpy.init()
    node = HandeyeCollector()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
