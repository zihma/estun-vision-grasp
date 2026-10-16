"""Synthetic RGB-D geometry checks; never connects to the robot or camera."""

import math
from pathlib import Path
import sys
import unittest

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from brick_orientation_geometry import InvalidObservation, Stability, closing_rotation, estimate, oriented_target_line


def scene(degrees, length=0.12, width=0.06, height=0.04):
    intrinsics = dict(fx=600.0, fy=600.0, cx=320.0, cy=240.0)
    transform = np.eye(4)
    transform[:3, 3] = [0.10, -0.12, 0.70]
    forward = -transform[:3, 3] / np.linalg.norm(transform[:3, 3])
    x = np.cross(forward, [0, 1, 0])
    x /= np.linalg.norm(x)
    transform[:3, :3] = np.column_stack((x, np.cross(forward, x), forward))
    v, u = np.mgrid[0:480, 0:640]
    rays = np.stack(((u-320)/600, (v-240)/600, np.ones_like(u)), -1) @ transform[:3, :3].T
    origin = transform[:3, 3]
    distance = (height - origin[2]) / rays[..., 2]
    points = origin + distance[..., None] * rays
    angle = math.radians(degrees)
    long = np.array([math.cos(angle), math.sin(angle)])
    short = np.array([-long[1], long[0]])
    surface = (np.abs(points[..., :2] @ long) < length/2) & (np.abs(points[..., :2] @ short) < width/2)
    depth = np.where(surface, distance, -origin[2] / rays[..., 2]) * 1000
    depth += np.random.default_rng(42).normal(0, 0.4, depth.shape)
    rows, columns = np.where(surface)
    bbox = [columns.min()-3, rows.min()-3, columns.max()+4, rows.max()+4]
    return depth, bbox, intrinsics, transform


class GeometryTest(unittest.TestCase):
    def test_execution_target_requires_valid_geometry_not_stability(self):
        result, _ = estimate(*scene(30))
        self.assertFalse(Stability().update(result, 1.0))
        words = oriented_target_line(1000, result, 0.9).split()
        self.assertEqual(words[:4], ['brick_v1', '1000.000000000', 'base_link', 'Foam_Brick'])
        self.assertEqual(len(words), 10)
        np.testing.assert_allclose([float(x) for x in words[4:7]], result['center_base_m'], atol=1e-8)
        np.testing.assert_allclose([float(x) for x in words[8:10]], result['short_axis_base_xy'], atol=1e-8)
        with self.assertRaises(InvalidObservation):
            oriented_target_line(1000, result, float('nan'))
        pringles = oriented_target_line(1000, result, 0.9, 'Pringles').split()
        self.assertEqual(pringles[:4], ['pringles_v1', '1000.000000000', 'base_link', 'Pringles'])
        with self.assertRaises(InvalidObservation):
            oriented_target_line(1000, result, 0.9, 'Bottle_half')

    def test_rotated_bricks_with_oblique_camera(self):
        for degrees in [0, 30, 60, 90, 150, 179]:
            with self.subTest(degrees=degrees):
                result, _ = estimate(*scene(degrees))
                error = abs((result['long_axis_deg_mod180']-degrees+90) % 180 - 90)
                self.assertLess(error, 3)
                self.assertAlmostEqual(result['length_m'], 0.12, delta=0.005)
                self.assertAlmostEqual(result['width_m'], 0.06, delta=0.005)
                self.assertAlmostEqual(result['center_base_m'][2], 0.04, delta=0.002)

    def test_table_only_rejected(self):
        with self.assertRaises(InvalidObservation):
            estimate(*scene(30, height=0))

    def test_square_has_no_reliable_long_axis(self):
        with self.assertRaises(InvalidObservation):
            estimate(*scene(30, width=0.12))

    def test_missing_depth_rejected(self):
        depth, bbox, intrinsics, transform = scene(0)
        depth[:] = 0
        with self.assertRaises(InvalidObservation):
            estimate(depth, bbox, intrinsics, transform)

    def test_gripper_45_degree_mounting_and_downward_axis(self):
        for degrees in [0, 30, 90, 170]:
            axis = np.array([math.cos(math.radians(degrees)), math.sin(math.radians(degrees)), 0])
            rotation = closing_rotation(axis)
            np.testing.assert_allclose(rotation @ np.array([2**-0.5, 2**-0.5, 0]), axis, atol=1e-12)
            np.testing.assert_allclose(rotation[:, 2], [0, 0, -1])
            self.assertAlmostEqual(np.linalg.det(rotation), 1)

    def test_temporal_wrap_and_motion_reset(self):
        stability = Stability()
        result, _ = estimate(*scene(0))
        for i in range(8):
            result = dict(result, long_axis_deg_mod180=179.5 if i % 2 else 0.5)
            self.assertEqual(stability.update(result, i*0.1), i == 7)
        self.assertFalse(stability.update(dict(result, long_axis_deg_mod180=45), 0.8))
        stability.clear()
        self.assertFalse(stability.update(result, 0.9))

    def test_status_serializes_after_brick_moves(self):
        stability = Stability()
        result, _ = estimate(*scene(0))
        sequence = [result] * 8 + [dict(result, long_axis_deg_mod180=45)] * 8
        # Direction changes, translation changes and dimension changes all
        # short-circuit through NumPy comparisons on different branches.
        sequence += [dict(sequence[-1], center_base_m=[0.1, 0.0, 0.04])]
        sequence += [dict(sequence[-1], length_m=0.16)]
        flags = []
        for i, observation in enumerate(sequence):
            valid = stability.update(observation, i * 0.1)
            self.assertIs(type(valid), bool)
            document = yaml.safe_load(yaml.safe_dump({'valid': valid, 'estimate': observation}))
            flags.append(document['valid'])
        self.assertTrue(flags[7])
        self.assertFalse(flags[8])
        self.assertTrue(flags[15])
        self.assertFalse(flags[16])
        self.assertFalse(flags[17])


if __name__ == '__main__':
    unittest.main()
