import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from selection import mask_from_polygon
from input_cloud import bounded_voxel_input


class SelectionBudgetTests(unittest.TestCase):
    def test_polygon_excludes_connected_same_depth_table(self):
        depth = np.full((100, 100), 860.0, np.float32)
        depth[45:50, 45:50] = 0
        mask = mask_from_polygon(depth, [(35,35),(65,35),(65,65),(35,65)])
        self.assertTrue(mask[40,40])
        self.assertFalse(mask[10,10])
        self.assertFalse(mask[46,46])

    def test_budget_preserves_target_in_shared_voxel(self):
        points = np.array([[0,0,0],[0.0001,0,0],[0.01,0,0]],np.float32)
        cloud, mask, voxel = bounded_voxel_input(points,[False,True,False],0.005)
        self.assertEqual(len(cloud),2)
        self.assertEqual(mask.tolist(),[True,False])

    def test_adapts_or_rejects_excess_input(self):
        points = np.random.default_rng(42).uniform(0,0.1,(5000,3)).astype(np.float32)
        cloud, mask, voxel = bounded_voxel_input(points,np.ones(5000,bool),0.003,2500)
        self.assertLessEqual(len(cloud),2500)
        self.assertGreater(voxel,0.003)
        with self.assertRaises(RuntimeError):
            bounded_voxel_input(points,np.ones(5000,bool),0.003,30)


if __name__ == '__main__':
    unittest.main()
