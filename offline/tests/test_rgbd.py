import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rgbd import project
from selection import mask_from_click


class TestProjection(unittest.TestCase):
    def test_explicit_millimetres_and_optical_axes(self):
        rgb = np.array([[[255, 0, 0], [0, 255, 0]],
                        [[0, 0, 255], [255, 255, 255]]], dtype=np.uint8)
        depth = np.array([[1000, 2000], [0, np.nan]], dtype=np.float32)
        points, colors, valid = project(rgb, depth, {"fx": 2, "fy": 2, "cx": 0, "cy": 0})
        np.testing.assert_allclose(points, [[0, 0, 1], [1, 0, 2]])
        np.testing.assert_allclose(colors, [[1, 0, 0], [0, 1, 0]])
        self.assertEqual(valid.sum(), 2)

    def test_rejects_bad_alignment(self):
        with self.assertRaises(ValueError):
            project(np.zeros((2, 2, 3), np.uint8), np.zeros((3, 2)),
                    {"fx": 1, "fy": 1, "cx": 0, "cy": 0})

    def test_click_selects_connected_depth_layer(self):
        depth = np.full((20, 20), 900, np.float32)
        depth[2:12, 2:12] = 800
        depth[15:19, 15:19] = 800
        selected = mask_from_click(depth, 4, 4)
        self.assertEqual(int(selected.sum()), 100)
        self.assertFalse(selected[16, 16])


if __name__ == "__main__":
    unittest.main()
