#!/usr/bin/python3
"""Solve fixed-camera eye-to-hand calibration in the MoveIt base_link frame."""

import argparse
import math
import os
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
import yaml


def invert(transform):
    result = np.eye(4, dtype=float)
    result[:3, :3] = transform[:3, :3].T
    result[:3, 3] = -result[:3, :3] @ transform[:3, 3]
    return result


def validate_transform(transform, label):
    matrix = np.asarray(transform, dtype=float)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise RuntimeError(f'{label} 必须是有限的 4x4 矩阵')
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1.0e-8):
        raise RuntimeError(f'{label} 齐次矩阵末行非法')
    rotation = matrix[:3, :3]
    orthogonality_error = float(
        np.linalg.norm(rotation.T @ rotation - np.eye(3)))
    determinant = float(np.linalg.det(rotation))
    if orthogonality_error > 1.0e-6 or abs(determinant - 1.0) > 1.0e-6:
        raise RuntimeError(
            f'{label} 不是合法旋转：orth={orthogonality_error:.3e}, '
            f'det={determinant:.9f}')


def quaternion_from_matrix(rotation):
    matrix = np.asarray(rotation, dtype=float)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quaternion = np.array([
            (matrix[2, 1] - matrix[1, 2]) / scale,
            (matrix[0, 2] - matrix[2, 0]) / scale,
            (matrix[1, 0] - matrix[0, 1]) / scale,
            0.25 * scale,
        ])
    else:
        index = int(np.argmax(np.diag(matrix)))
        if index == 0:
            scale = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            quaternion = np.array([0.25 * scale,
                (matrix[0, 1] + matrix[1, 0]) / scale,
                (matrix[0, 2] + matrix[2, 0]) / scale,
                (matrix[2, 1] - matrix[1, 2]) / scale])
        elif index == 1:
            scale = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            quaternion = np.array([
                (matrix[0, 1] + matrix[1, 0]) / scale, 0.25 * scale,
                (matrix[1, 2] + matrix[2, 1]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale])
        else:
            scale = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            quaternion = np.array([
                (matrix[0, 2] + matrix[2, 0]) / scale,
                (matrix[1, 2] + matrix[2, 1]) / scale, 0.25 * scale,
                (matrix[1, 0] - matrix[0, 1]) / scale])
    quaternion /= np.linalg.norm(quaternion)
    return quaternion


def rotation_angle_deg(rotation):
    cosine = float(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def solve_ax_zb(camera_to_marker, base_to_tool):
    """Solve A*X=Z*B; X=T_marker_tool and Z=T_camera_base."""
    # Relative motions satisfy A_i^-1 A_j X = X B_i^-1 B_j.  Align their
    # rotation vectors to initialize X; zero rotation can converge to a
    # distant local minimum when the marker and tool frames differ greatly.
    marker_motion_vectors = []
    tool_motion_vectors = []
    for first in range(len(camera_to_marker)):
        for second in range(first + 1, len(camera_to_marker)):
            marker_motion = (
                camera_to_marker[first][:3, :3].T @
                camera_to_marker[second][:3, :3])
            tool_motion = (
                base_to_tool[first][:3, :3].T @
                base_to_tool[second][:3, :3])
            marker_motion_vectors.append(Rotation.from_matrix(marker_motion).as_rotvec())
            tool_motion_vectors.append(Rotation.from_matrix(tool_motion).as_rotvec())
    initial_x, _ = Rotation.align_vectors(
        marker_motion_vectors, tool_motion_vectors)
    initial_z = Rotation.from_matrix(np.asarray([
        transform_a[:3, :3] @ initial_x.as_matrix() @ transform_b[:3, :3].T
        for transform_a, transform_b in zip(camera_to_marker, base_to_tool)
    ])).mean()

    def rotation_residual(parameters):
        rotation_x = Rotation.from_rotvec(parameters[:3]).as_matrix()
        rotation_z = Rotation.from_rotvec(parameters[3:]).as_matrix()
        residual = []
        for transform_a, transform_b in zip(camera_to_marker, base_to_tool):
            left = transform_a[:3, :3] @ rotation_x
            right = rotation_z @ transform_b[:3, :3]
            residual.extend(Rotation.from_matrix(left.T @ right).as_rotvec())
        return np.asarray(residual)

    rotation_solution = least_squares(
        rotation_residual,
        np.concatenate((initial_x.as_rotvec(), initial_z.as_rotvec())),
        method='lm', max_nfev=5000)
    if not rotation_solution.success:
        raise RuntimeError(f'旋转求解失败：{rotation_solution.message}')
    rotation_x = Rotation.from_rotvec(rotation_solution.x[:3]).as_matrix()
    rotation_z = Rotation.from_rotvec(rotation_solution.x[3:]).as_matrix()

    # R_A*t_X + t_A = R_Z*t_B + t_Z
    lhs = []
    rhs = []
    for transform_a, transform_b in zip(camera_to_marker, base_to_tool):
        lhs.append(np.hstack((transform_a[:3, :3], -np.eye(3))))
        rhs.append(rotation_z @ transform_b[:3, 3] - transform_a[:3, 3])
    translation_solution, _, _, _ = np.linalg.lstsq(
        np.vstack(lhs), np.concatenate(rhs), rcond=None)

    transform_x = np.eye(4, dtype=float)
    transform_x[:3, :3] = rotation_x
    transform_x[:3, 3] = translation_solution[:3]
    transform_z = np.eye(4, dtype=float)
    transform_z[:3, :3] = rotation_z
    transform_z[:3, 3] = translation_solution[3:]
    return transform_x, transform_z


def solve(input_path, output_path):
    document = yaml.safe_load(input_path.read_text(encoding='utf-8')) or {}
    samples = document.get('samples', [])
    if len(samples) < 6:
        raise RuntimeError(f'至少需要 6 组多方向样本，当前只有 {len(samples)} 组')
    if document.get('base_frame') != 'base_link':
        raise RuntimeError('样本不是在 MoveIt base_link 下采集，拒绝求解')

    camera_to_marker = [np.asarray(item['camera_to_marker'], dtype=float) for item in samples]
    base_to_tool = [np.asarray(item['base_to_tool'], dtype=float) for item in samples]
    for index, transform in enumerate(camera_to_marker, 1):
        validate_transform(transform, f'camera_to_marker[{index}]')
    for index, transform in enumerate(base_to_tool, 1):
        validate_transform(transform, f'base_to_tool[{index}]')

    # Eye-to-hand mapping to A*X=Z*B:
    # A=T_camera_marker, B=T_base_tool (swap the library's robot labels).
    # Z is T_camera_base; invert it to obtain the TF T_base_camera.
    target_to_tool, camera_to_base = solve_ax_zb(camera_to_marker, base_to_tool)
    base_to_camera = invert(camera_to_base)
    validate_transform(base_to_camera, 'matrix_T_base_camera')

    translation_errors = []
    rotation_errors = []
    for camera_marker, base_tool in zip(camera_to_marker, base_to_tool):
        left = camera_marker @ target_to_tool
        right = camera_to_base @ base_tool
        delta = invert(left) @ right
        translation_errors.append(float(np.linalg.norm(delta[:3, 3])))
        rotation_errors.append(rotation_angle_deg(delta[:3, :3]))

    translation_rms = float(np.sqrt(np.mean(np.square(translation_errors))))
    rotation_rms = float(np.sqrt(np.mean(np.square(rotation_errors))))
    quaternion = quaternion_from_matrix(base_to_camera[:3, :3])

    output = {
        'calibrated': translation_rms < 0.01 and rotation_rms < 2.0,
        'calibration_type': 'eye_to_hand',
        'parent_frame': 'base_link',
        'child_frame': 'gemini336_color_optical_frame',
        'translation': [float(value) for value in base_to_camera[:3, 3]],
        'quaternion_xyzw': [float(value) for value in quaternion],
        'sample_count': len(samples),
        'translation_rms_m': translation_rms,
        'rotation_rms_deg': rotation_rms,
        'solver': 'SciPy nonlinear rotation + linear translation A*X=Z*B',
        'matrix_T_base_camera': base_to_camera.tolist(),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix('.tmp')
    temporary.write_text(
        yaml.safe_dump(output, sort_keys=False, allow_unicode=True), encoding='utf-8')
    os.replace(temporary, output_path)
    print(yaml.safe_dump(output, sort_keys=False, allow_unicode=True))
    if translation_rms > 0.01 or rotation_rms > 2.0:
        print('WARNING: 标定残差偏大，请增加姿态多样性并重新采样。')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    arguments = parser.parse_args()
    solve(arguments.input, arguments.output)


if __name__ == '__main__':
    main()
