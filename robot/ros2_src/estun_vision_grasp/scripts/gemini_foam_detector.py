#!/usr/bin/env python3
"""Gemini336 RGB-D + YOLO foam-brick detector in camera optical coordinates."""

import argparse
import os
from pathlib import Path
import time

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

from gemini336_camera import GeminiCamera
from tape_rim_target import InvalidTapeTarget, select_left_rim

TAPE_ORIENTED_OUTPUT = Path('/tmp/estun_tape_oriented_target_base.txt')


def atomic_yaml(path, document):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(yaml.safe_dump(document, sort_keys=False), encoding='utf-8')
    os.replace(temporary, path)


def atomic_text(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(value, encoding='utf-8')
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, type=Path)
    parser.add_argument('--output', default='/tmp/estun_foam_target_camera.yaml', type=Path)
    parser.add_argument('--confidence', default=0.50, type=float)
    parser.add_argument('--class-name', default='Foam_Brick')
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--target-point', choices=['box-center', 'tape-left-rim'],
                        default='box-center')
    parser.add_argument('--calibration', type=Path,
                        default=Path(__file__).resolve().parents[1] / 'config/handeye_result.yaml')
    args = parser.parse_args()
    os.environ.setdefault('YOLO_CONFIG_DIR', '/tmp/estun_yolo_config')
    transform = None
    if args.target_point == 'tape-left-rim':
        if args.class_name != 'Scotch_Duct_Tape' or not args.show:
            parser.error('tape-left-rim 只支持 Scotch_Duct_Tape，且必须使用 --show 检查落点')
        calibration = yaml.safe_load(args.calibration.read_text(encoding='utf-8'))
        if (not calibration.get('calibrated') or
                calibration.get('parent_frame') != 'base_link' or
                calibration.get('child_frame') != 'gemini336_color_optical_frame'):
            parser.error('手眼标定无效或坐标系不匹配')
        transform = np.asarray(calibration['matrix_T_base_camera'], dtype=float)
        atomic_text(TAPE_ORIENTED_OUTPUT, 'INVALID STARTING\n')

    model = YOLO(str(args.model))
    camera = GeminiCamera()
    camera.start()
    intrinsics = camera.intrinsics
    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(f'Gemini336 detector: model={args.model}, output={args.output}')

    try:
        while True:
            frames = camera.get_frames()
            if frames is None:
                if transform is not None:
                    atomic_text(TAPE_ORIENTED_OUTPUT, 'INVALID NO_CAMERA_FRAME\n')
                continue
            capture_stamp = time.time()
            image = frames['rgb']
            depth = frames['depth']
            result = model.predict(source=image, conf=args.confidence, verbose=False)[0]
            candidates = []
            if result.boxes is not None:
                for index, box in enumerate(result.boxes):
                    class_id = int(box.cls[0])
                    if str(model.names[class_id]) == args.class_name:
                        candidates.append(index)
            if candidates:
                index = max(candidates, key=lambda item: float(result.boxes[item].conf[0]))
                box = result.boxes[index]
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                confidence = float(box.conf[0])
                pixel_x = int((x1 + x2) * 0.5)
                pixel_y = int((y1 + y2) * 0.5)
                point = None
                error = None
                if args.target_point == 'tape-left-rim':
                    try:
                        point, (pixel_x, pixel_y), diagnostics = select_left_rim(
                            depth, (x1, y1, x2, y2), intrinsics, transform)
                        if time.time() - capture_stamp > 1.0:
                            raise InvalidTapeTarget('图像处理超时')
                    except InvalidTapeTarget as exc:
                        error = str(exc)
                else:
                    radius_x = max(3, int((x2 - x1) * 0.15))
                    radius_y = max(3, int((y2 - y1) * 0.15))
                    roi = depth[
                        max(0, pixel_y - radius_y):min(depth.shape[0], pixel_y + radius_y + 1),
                        max(0, pixel_x - radius_x):min(depth.shape[1], pixel_x + radius_x + 1)]
                    valid = roi[np.isfinite(roi) & (roi > 0.0)]
                    if valid.size:
                        z = float(np.median(valid)) / 1000.0
                        point = [(pixel_x - intrinsics['cx']) * z / intrinsics['fx'],
                                 (pixel_y - intrinsics['cy']) * z / intrinsics['fy'], z]
                if point is not None:
                    target_stamp = capture_stamp if transform is not None else time.time()
                    atomic_yaml(args.output, {
                        'stamp': target_stamp,
                        'camera_frame': 'gemini336_color_optical_frame',
                        'xyz_camera_m': [float(v) for v in point],
                        'confidence': confidence,
                        'pixel': [pixel_x, pixel_y],
                        'target_point': args.target_point,
                        **({'rim_diagnostics': diagnostics} if args.target_point == 'tape-left-rim'
                           else {'depth_median_sample_count': int(valid.size)}),
                    })
                    if transform is not None:
                        base_point = transform[:3, :3] @ np.asarray(point) + transform[:3, 3]
                        axis = diagnostics['radial_axis_base_xy']
                        atomic_text(TAPE_ORIENTED_OUTPUT,
                                    f'tape_v1 {target_stamp:.9f} base_link Scotch_Duct_Tape '
                                    f'{base_point[0]:.9f} {base_point[1]:.9f} {base_point[2]:.9f} '
                                    f'{confidence:.6f} {axis[0]:.9f} {axis[1]:.9f}\n')
                elif transform is not None:
                    atomic_text(TAPE_ORIENTED_OUTPUT, 'INVALID RIM_GEOMETRY\n')
                if args.show:
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(image, f'{args.class_name} {confidence:.2f}',
                        (x1, max(30, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2)
                    if point is not None:
                        cv2.circle(image, (pixel_x, pixel_y), 5, (0, 0, 255), -1)
                        if transform is not None:
                            center = ((x1 + x2) // 2, (y1 + y2) // 2)
                            other = (2 * center[0] - pixel_x, 2 * center[1] - pixel_y)
                            cv2.line(image, (pixel_x, pixel_y), other, (255, 0, 255), 2)
                            cv2.circle(image, center, 3, (255, 255, 0), -1)
                            cv2.putText(image, 'RADIAL CLOSING AXIS', (10, image.shape[0] - 16),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 255), 2)
                    elif error:
                        cv2.putText(image, 'RIM TARGET INVALID', (15, 35), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5, (0, 0, 255), 2)
            elif transform is not None:
                atomic_text(TAPE_ORIENTED_OUTPUT, 'INVALID NO_TAPE_DETECTION\n')
            if args.show:
                cv2.imshow('Gemini336 foam detector', image)
                if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q')):
                    break
    finally:
        if transform is not None:
            atomic_text(TAPE_ORIENTED_OUTPUT, 'INVALID STOPPED\n')
        camera.stop()
        if args.show:
            cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
