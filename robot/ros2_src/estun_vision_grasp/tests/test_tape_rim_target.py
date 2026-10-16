import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from tape_rim_target import InvalidTapeTarget, select_left_rim


def test_left_rim_uses_raised_material_and_radial_axis():
    yy, xx = np.mgrid[:140, :140]
    radial_pixels = np.hypot(xx - 70, yy - 70)
    depth = np.full((140, 140), 1000, dtype=np.uint16)
    depth[(radial_pixels >= 18) & (radial_pixels <= 37)] = 950
    transform = np.diag([1.0, -1.0, -1.0, 1.0])
    transform[2, 3] = 1.0
    intrinsics = {'fx': 400.0, 'fy': 400.0, 'cx': 70.0, 'cy': 70.0}

    point, pixel, diagnostics = select_left_rim(
        depth, (30, 30, 110, 110), intrinsics, transform)

    assert pixel[0] < 70
    assert abs(pixel[1] - 70) <= 2
    assert point[2] == pytest.approx(0.950)
    assert diagnostics['height_m'] == pytest.approx(0.050)
    assert diagnostics['radial_axis_base_xy'][0] < -0.99
    assert abs(diagnostics['radial_axis_base_xy'][1]) < 0.05


def test_no_raised_rim_is_rejected():
    depth = np.full((140, 140), 1000, dtype=np.uint16)
    transform = np.diag([1.0, -1.0, -1.0, 1.0])
    transform[2, 3] = 1.0
    intrinsics = {'fx': 400.0, 'fy': 400.0, 'cx': 70.0, 'cy': 70.0}

    with pytest.raises(InvalidTapeTarget):
        select_left_rim(depth, (30, 30, 110, 110), intrinsics, transform)
