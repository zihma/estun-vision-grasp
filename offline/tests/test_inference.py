import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from inference import predict

class Group(list):
    def nms(self): return self
    def sort_by_score(self): return self

def grasp(x):
    return SimpleNamespace(translation=np.array([x, 0., 0.]), rotation_matrix=np.eye(3),
                           depth=.02, width=.05, height=.03, score=.9, object_id=-1)

class InferenceTests(unittest.TestCase):
    def test_target_filter_before_limit_and_steering(self):
        options = {}
        group = Group([grasp(1.) for _ in range(25)] + [grasp(0.)])
        def get_grasp(points, params):
            options.update(params)
            return group
        sdk = SimpleNamespace(check_license=lambda _: True,
                              create_detector=lambda _: SimpleNamespace(get_grasp=get_grasp))
        with patch.dict(sys.modules, {'gsnet': sdk}):
            result = predict(np.zeros((32, 3), np.float32), np.ones(32, bool), 'model', .1, .03,
                             approach_direction=[0, 0, 1], approach_thresh_deg=25)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['position_xyz_m'], [0., 0., 0.])
        self.assertTrue(options['collision_detection'])
        self.assertAlmostEqual(options['approach_thresh'], np.deg2rad(25))

    def test_invalid_license_blocks_detector(self):
        def unexpected(_): raise AssertionError('Detector must not load')
        sdk = SimpleNamespace(check_license=lambda _: False, create_detector=unexpected)
        with patch.dict(sys.modules, {'gsnet': sdk}), self.assertRaises(RuntimeError):
            predict(np.zeros((32, 3), np.float32), np.ones(32, bool), 'model', .1, .03)

    def test_nonfinite_input_rejected(self):
        with self.assertRaises(ValueError):
            predict(np.full((32, 3), np.nan, np.float32), np.ones(32, bool), 'model', .1, .03)

if __name__ == '__main__': unittest.main()
