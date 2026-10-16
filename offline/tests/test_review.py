import sys
import unittest
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from review_candidates import review

class ReviewTests(unittest.TestCase):
    def test_tip_offset_and_both_closing_symmetries(self):
        c = {'rotation_matrix': [[0,1,0],[0,0,-1],[-1,0,0]],
             'position_xyz_m': [.4,0,.1], 'insertion_depth_m': .02,
             'model_score': .9, 'opening_width_m': .09}
        result = review(c, np.eye(4))
        np.testing.assert_allclose(result['grasp_reference_base_m'], [.4,0,.08])
        np.testing.assert_allclose(result['estimated_tip_base_m'], [.4,0,.05])
        np.testing.assert_allclose(result['estimated_tool0_base_m'], [.4,0,.08])
        np.testing.assert_allclose(result['estimated_pregrasp_tool0_base_m'], [.4,0,.23])
        for sign, q in zip([1,-1], result['equivalent_tool0_quaternions_xyzw']):
            r = Rotation.from_quat(q).as_matrix()
            np.testing.assert_allclose(r @ [0,0,1], [0,0,-1], atol=1e-7)
            np.testing.assert_allclose(r @ [2**-.5,2**-.5,0], [sign,0,0], atol=1e-7)
        self.assertTrue(result['within_25deg_down_cone'])
        self.assertTrue(result['opening_plus_10mm_within_measured_120mm'])

    def test_reflection_rejected(self):
        with self.assertRaises(ValueError):
            review({'rotation_matrix': np.diag([1,1,-1]).tolist()}, np.eye(4))

if __name__ == '__main__': unittest.main()
