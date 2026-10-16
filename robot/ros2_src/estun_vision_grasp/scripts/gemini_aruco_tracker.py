#!/usr/bin/env python3
"""Gemini336 ArUco pose producer. Run with the vision_env Conda Python."""

import argparse
import math
import os
from pathlib import Path
import time

import cv2
import numpy as np
import yaml

from gemini336_camera import GeminiCamera


def quaternion_from_rotation(rotation):
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quaternion = np.array([
            (rotation[2, 1] - rotation[1, 2]) / scale,
            (rotation[0, 2] - rotation[2, 0]) / scale,
            (rotation[1, 0] - rotation[0, 1]) / scale,
            0.25 * scale,
        ])
    else:
        diagonal = np.diag(rotation)
        index = int(np.argmax(diagonal))
        if index == 0:
            scale = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
            quaternion = np.array([0.25 * scale,
                (rotation[0, 1] + rotation[1, 0]) / scale,
                (rotation[0, 2] + rotation[2, 0]) / scale,
                (rotation[2, 1] - rotation[1, 2]) / scale])
        elif index == 1:
            scale = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
            quaternion = np.array([
                (rotation[0, 1] + rotation[1, 0]) / scale, 0.25 * scale,
                (rotation[1, 2] + rotation[2, 1]) / scale,
                (rotation[0, 2] - rotation[2, 0]) / scale])
        else:
            scale = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
            quaternion = np.array([
                (rotation[0, 2] + rotation[2, 0]) / scale,
                (rotation[1, 2] + rotation[2, 1]) / scale, 0.25 * scale,
                (rotation[1, 0] - rotation[0, 1]) / scale])
    return (quaternion / np.linalg.norm(quaternion)).tolist()


def atomic_yaml(path, document):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(yaml.safe_dump(document, sort_keys=False), encoding='utf-8')
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='/tmp/estun_handeye_marker.yaml', type=Path)
    parser.add_argument('--marker-id', default=0, type=int)
    parser.add_argument('--marker-length', default=0.10, type=float)
    parser.add_argument('--show', action='store_true')
    args = parser.parse_args()

    camera = GeminiCamera()
    camera.start()
    intrinsics = camera.intrinsics
    camera_matrix = np.array([
        [intrinsics['fx'], 0.0, intrinsics['cx']],
        [0.0, intrinsics['fy'], intrinsics['cy']],
        [0.0, 0.0, 1.0]], dtype=np.float64)
    distortion = np.zeros((5, 1), dtype=np.float64)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
    half = args.marker_length * 0.5
    object_points = np.array([
        [-half, half, 0.0], [half, half, 0.0],
        [half, -half, 0.0], [-half, -half, 0.0]], dtype=np.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(f'Gemini336 ArUco tracker: marker={args.marker_id}, output={args.output}')

    try:
        while True:
            frames = camera.get_frames()
            if frames is None:
                continue
            image = frames['rgb']
            corners, ids, _ = detector.detectMarkers(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
            if ids is not None:
                ids_flat = ids.reshape(-1)
                matches = np.where(ids_flat == args.marker_id)[0]
                if matches.size:
                    image_points = corners[int(matches[0])].reshape(4, 2).astype(np.float32)
                    ok, rvec, tvec = cv2.solvePnP(
                        object_points, image_points, camera_matrix, distortion,
                        flags=cv2.SOLVEPNP_IPPE_SQUARE)
                    if ok:
                        rotation, _ = cv2.Rodrigues(rvec)
                        atomic_yaml(args.output, {
                            'stamp': time.time(),
                            'camera_frame': 'gemini336_color_optical_frame',
                            'marker_id': args.marker_id,
                            'translation': [float(value) for value in tvec.reshape(3)],
                            'quaternion_xyzw': quaternion_from_rotation(rotation),
                        })
                        if args.show:
                            cv2.drawFrameAxes(image, camera_matrix, distortion, rvec, tvec, 0.05)
                if args.show:
                    cv2.aruco.drawDetectedMarkers(image, corners, ids)
            if args.show:
                cv2.imshow('Gemini336 hand-eye ArUco', image)
                if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q')):
                    break
    finally:
        camera.stop()
        if args.show:
            cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
