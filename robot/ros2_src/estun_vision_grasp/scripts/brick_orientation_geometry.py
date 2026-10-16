"""Offline geometry for a single flat brick; no ROS or camera access."""

from collections import deque
import math

import cv2
import numpy as np


class InvalidObservation(ValueError):
    pass


def oriented_target_line(stamp, result, confidence, label='Foam_Brick'):
    """Dedicated execution interface. Caller must have a valid current geometry estimate."""
    versions = {'Foam_Brick': 'brick_v1', 'Pringles': 'pringles_v1'}
    if label not in versions:
        raise InvalidObservation('UNSUPPORTED_ORIENTED_CLASS')
    xyz = np.asarray(result['center_base_m'], dtype=float)
    axis = np.asarray(result['short_axis_base_xy'], dtype=float)
    if (xyz.shape != (3,) or axis.shape != (2,)
            or not np.isfinite([stamp, confidence, *xyz, *axis]).all()
            or not 0.5 <= confidence <= 1.0 or abs(np.linalg.norm(axis) - 1) > 0.02):
        raise InvalidObservation('BAD_EXECUTION_TARGET')
    return (f'{versions[label]} {stamp:.9f} base_link {label} '
            f'{xyz[0]:.9f} {xyz[1]:.9f} {xyz[2]:.9f} {confidence:.6f} '
            f'{axis[0]:.9f} {axis[1]:.9f}\n')


def validate_calibration(data):
    if (not data.get('calibrated') or data.get('parent_frame') != 'base_link'
            or data.get('child_frame') != 'gemini336_color_optical_frame'):
        raise ValueError('需要有效的 base_link -> Gemini optical 手眼标定')
    transform = np.asarray(data['matrix_T_base_camera'], dtype=float)
    if (transform.shape != (4, 4) or not np.isfinite(transform).all()
            or not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-8)
            or not np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-6)
            or abs(np.linalg.det(transform[:3, :3]) - 1) > 1e-6):
        raise ValueError('手眼矩阵不是有效刚体变换')
    for key, limit in [('translation_rms_m', 0.010), ('rotation_rms_deg', 2.0)]:
        value = float(data.get(key, float('nan')))
        if not math.isfinite(value) or value >= limit:
            raise ValueError(f'标定残差不合格：{key}={value}')
    return transform


def closing_rotation(short_axis, installation_deg=45.0):
    """R maps tool0 to base; local [cos(a),sin(a),0] must map to short_axis."""
    closing = np.array([short_axis[0], short_axis[1], 0.0], dtype=float)
    closing /= np.linalg.norm(closing)
    down = np.array([0.0, 0.0, -1.0])
    angle = math.radians(installation_deg)
    x_axis = math.cos(angle) * closing - math.sin(angle) * np.cross(down, closing)
    return np.column_stack((x_axis, np.cross(down, x_axis), down))


def estimate(depth_mm, bbox, intrinsics, transform, band_m=0.006):
    """Segment a raised horizontal top face, then fit a rectangle in base XY.

    Assumes aligned RGB-depth, one flat brick, visible tabletop around its box.
    Returns diagnostics and a full-resolution accepted-surface mask.
    """
    if depth_mm.ndim != 2:
        raise InvalidObservation('BAD_DEPTH_SHAPE')
    h, w = depth_mm.shape
    x1, y1, x2, y2 = [int(v) for v in bbox]
    if x1 < 3 or y1 < 3 or x2 >= w - 3 or y2 >= h - 3:
        raise InvalidObservation('BRICK_AT_IMAGE_EDGE')
    if x2 - x1 < 16 or y2 - y1 < 16:
        raise InvalidObservation('BOX_TOO_SMALL')
    pad = max(10, int(max(x2 - x1, y2 - y1) * 0.25))
    left, top = max(0, x1 - pad), max(0, y1 - pad)
    right, bottom = min(w, x2 + pad), min(h, y2 + pad)
    vv, uu = np.mgrid[top:bottom, left:right]
    z = depth_mm[top:bottom, left:right].astype(float) / 1000.0
    valid = np.isfinite(z) & (z > 0.10) & (z < 3.0)
    fx, fy, cx, cy = [float(intrinsics[k]) for k in ('fx', 'fy', 'cx', 'cy')]
    if not np.isfinite([fx, fy, cx, cy]).all() or min(fx, fy) <= 0:
        raise InvalidObservation('BAD_INTRINSICS')
    camera = np.stack(((uu - cx) * z / fx, (vv - cy) * z / fy, z), axis=-1)
    base = camera @ transform[:3, :3].T + transform[:3, 3]
    inside = (uu >= x1) & (uu < x2) & (vv >= y1) & (vv < y2)
    ring = (~inside) & valid
    if np.count_nonzero(ring) < 150:
        raise InvalidObservation('NEED_VISIBLE_TABLE_AROUND_BRICK')
    table_values = base[..., 2][ring]
    table_z = float(np.median(table_values))
    if np.median(np.abs(table_values - table_z)) > 0.005:
        raise InvalidObservation('TABLE_NOT_FLAT_OR_BACKGROUND_OCCLUDED')
    above = inside & valid & (base[..., 2] > table_z + 0.008)
    if np.count_nonzero(above) < 150:
        raise InvalidObservation('TOP_NOT_SEPARATED_FROM_TABLE')
    heights = base[..., 2][above]
    bins = np.arange(float(np.min(heights)), float(np.max(heights)) + 0.004, 0.002)
    counts, edges = np.histogram(heights, bins=bins)
    peak = int(np.argmax(counts))
    peak_z = (edges[peak] + edges[peak + 1]) * 0.5
    top_z = float(np.median(heights[np.abs(heights - peak_z) <= band_m]))
    raw = (above & (np.abs(base[..., 2] - top_z) <= band_m)).astype(np.uint8)
    clean = cv2.morphologyEx(raw, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(clean, 8)
    if count <= 1:
        raise InvalidObservation('NO_CONNECTED_TOP_FACE')
    order = np.argsort(stats[1:, cv2.CC_STAT_AREA])[::-1] + 1
    label = int(order[0])
    if len(order) > 1 and stats[order[1], cv2.CC_STAT_AREA] > stats[label, cv2.CC_STAT_AREA] * 0.25:
        raise InvalidObservation('MULTIPLE_SURFACES')
    selected = labels == label
    points = base[selected]
    if len(points) < 150 or len(points) < np.count_nonzero(inside) * 0.12:
        raise InvalidObservation('TOO_LITTLE_TOP_FACE')
    centered = points - np.mean(points, axis=0)
    _, _, vectors = np.linalg.svd(centered[::max(1, len(points) // 3000)], full_matrices=False)
    normal = vectors[-1]
    tilt = math.degrees(math.acos(float(np.clip(abs(normal[2]), 0, 1))))
    plane_rms = float(np.sqrt(np.mean((centered @ normal) ** 2)))
    if tilt > 12 or plane_rms > 0.004:
        raise InvalidObservation('TOP_NOT_HORIZONTAL_OR_DEPTH_NOISY')
    rect = cv2.minAreaRect(points[:, :2].astype(np.float32))
    corners = cv2.boxPoints(rect).astype(float)
    edges_xy = np.roll(corners, -1, axis=0) - corners
    lengths = np.linalg.norm(edges_xy, axis=1)
    length, width = float(np.max(lengths)), float(np.min(lengths))
    if width < 0.015 or length > 0.40 or length / width < 1.35 or length / width > 6:
        raise InvalidObservation('SIZE_OR_LONG_AXIS_AMBIGUOUS')
    hull = cv2.convexHull(points[:, :2].astype(np.float32))
    rectangularity = float(cv2.contourArea(hull) / (length * width))
    if rectangularity < 0.78:
        raise InvalidObservation('INCOMPLETE_RECTANGULAR_TOP')
    # Convex hull alone hides holes: require adequate occupied cells in fitted axes.
    long_axis = edges_xy[int(np.argmax(lengths))] / length
    angle = math.atan2(long_axis[1], long_axis[0]) % math.pi
    long_axis = np.array([math.cos(angle), math.sin(angle)])
    short_axis = np.array([-long_axis[1], long_axis[0]])
    xy = points[:, :2] - np.asarray(rect[0])
    normalized = np.column_stack((xy @ long_axis / length, xy @ short_axis / width)) + 0.5
    cells = np.clip((normalized * 12).astype(int), 0, 11)
    coverage = len(np.unique(cells[:, 0] * 12 + cells[:, 1])) / 144
    if coverage < 0.75:
        raise InvalidObservation('TOP_HAS_LARGE_HOLES_OR_OCCLUSION')
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[top:bottom, left:right] = selected.astype(np.uint8) * 255
    result = {
        'center_base_m': [float(rect[0][0]), float(rect[0][1]), top_z],
        'long_axis_base_xy': long_axis.tolist(),
        'short_axis_base_xy': short_axis.tolist(),
        'long_axis_deg_mod180': math.degrees(angle),
        'short_axis_deg_mod180': math.degrees(math.atan2(short_axis[1], short_axis[0])) % 180,
        'length_m': length, 'width_m': width,
        'table_z_m': table_z, 'height_above_table_m': top_z - table_z,
        'surface_points': len(points), 'top_plane_rms_m': plane_rms,
        'top_tilt_deg': tilt, 'rectangularity': rectangularity, 'surface_coverage': coverage,
        'suggested_tool0_rotation_base': closing_rotation(short_axis).tolist(),
        'gripper_closing_axis_offset_deg': 45.0,
    }
    return result, mask


class Stability:
    def __init__(self, frames=8):
        self.history = deque(maxlen=frames)

    def clear(self):
        self.history.clear()

    def update(self, result, stamp):
        if self.history and stamp - self.history[-1][0] > 1.5:
            self.clear()
        self.history.append((stamp, result))
        angles = np.radians([r['long_axis_deg_mod180'] for _, r in self.history])
        mean = np.angle(np.mean(np.exp(2j * angles))) / 2
        error = np.abs(np.angle(np.exp(2j * (angles - mean)))) / 2
        centers = np.array([r['center_base_m'] for _, r in self.history])
        sizes = np.array([[r['length_m'], r['width_m']] for _, r in self.history])
        stable = (np.max(error) <= math.radians(3)
                  and np.max(np.linalg.norm(centers - centers.mean(axis=0), axis=1)) <= 0.005
                  and np.max(np.ptp(sizes, axis=0)) <= 0.010)
        if not stable:
            last = self.history[-1]
            self.clear()
            self.history.append(last)
        # A failed NumPy comparison yields np.bool_(False); YAML safe_dump
        # cannot serialize it. Always expose a native bool on every branch.
        return bool(stable and len(self.history) == self.history.maxlen)


def project(base_points, intrinsics, transform):
    camera = (np.asarray(base_points) - transform[:3, 3]) @ transform[:3, :3]
    if np.any(camera[:, 2] <= 0):
        raise InvalidObservation('PROJECTION_BEHIND_CAMERA')
    pixels = camera[:, :2] / camera[:, 2, None]
    return np.rint(pixels * [intrinsics['fx'], intrinsics['fy']]
                   + [intrinsics['cx'], intrinsics['cy']]).astype(int)
