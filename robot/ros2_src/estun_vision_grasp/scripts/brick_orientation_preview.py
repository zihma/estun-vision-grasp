#!/usr/bin/env python3
"""Brick direction display, optionally writing a live oriented-grasp target. No robot commands."""

import argparse
import json
import os
from pathlib import Path
import time

import cv2
import numpy as np
import yaml

from brick_orientation_geometry import InvalidObservation, Stability, estimate, project, validate_calibration, oriented_target_line


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path('/tmp/estun_brick_orientation_preview.yaml')
GRASP_OUTPUTS = {
    'Foam_Brick': Path('/tmp/estun_brick_oriented_target_base.txt'),
    'Pringles': Path('/tmp/estun_pringles_oriented_target_base.txt'),
}


def write_grasp_target(path, line):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(line, encoding='utf-8')
    os.replace(temporary, path)


def write_status(data):
    temporary = OUTPUT.with_suffix('.tmp')
    temporary.write_text(yaml.safe_dump(data, sort_keys=False), encoding='utf-8')
    os.replace(temporary, OUTPUT)


def draw(image, result, mask, intrinsics, transform):
    highlighted = image.copy()
    highlighted[mask > 0] = (180, 100, 20)
    image[:] = cv2.addWeighted(image, 0.65, highlighted, 0.35, 0)
    center = np.array(result['center_base_m'])
    for axis_key, size_key, color in [
            ('long_axis_base_xy', 'length_m', (0, 255, 255)),
            ('short_axis_base_xy', 'width_m', (255, 0, 255))]:
        axis = np.array([*result[axis_key], 0.0])
        ends = project([center - axis * result[size_key] * 0.5,
                        center + axis * result[size_key] * 0.5], intrinsics, transform)
        cv2.line(image, tuple(ends[0]), tuple(ends[1]), color, 3)
    point = project([center], intrinsics, transform)[0]
    cv2.circle(image, tuple(point), 5, (255, 255, 255), -1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=ROOT / 'models/competition/group1/best.pt')
    parser.add_argument('--class-name', choices=tuple(GRASP_OUTPUTS), default='Foam_Brick')
    parser.add_argument('--calibration', type=Path, default=ROOT / 'config/handeye_result.yaml')
    parser.add_argument('--replay', type=Path, help='saved NPZ snapshot; no camera needed')
    parser.add_argument('--snapshots', type=Path, default=Path('/tmp/estun_brick_orientation_snapshots'))
    parser.add_argument('--publish-grasp-target', action='store_true',
                        help='publish single-frame valid geometry to the dedicated oriented-grasp file; no robot commands')
    args = parser.parse_args()
    grasp_output = GRASP_OUTPUTS[args.class_name]
    if args.replay and args.publish_grasp_target:
        parser.error('回放数据不能作为实时抓取目标')
    if args.publish_grasp_target:
        write_grasp_target(grasp_output, 'INVALID STARTING\n')
    transform = validate_calibration(yaml.safe_load(args.calibration.read_text()))
    if args.replay:
        with np.load(args.replay, allow_pickle=False) as data:
            result, _ = estimate(data['depth_mm'], data['bbox'],
                                 json.loads(str(data['intrinsics_json'])), data['transform'])
        print(yaml.safe_dump(result, sort_keys=False))
        return

    os.environ.setdefault('YOLO_CONFIG_DIR', '/tmp/estun_yolo_config')
    from ultralytics import YOLO
    from gemini336_camera import GeminiCamera

    model = YOLO(str(args.model))
    if args.class_name not in model.names.values():
        raise ValueError(f'模型中没有 {args.class_name} 类别')
    camera = GeminiCamera()
    stability = Stability()
    last_status = None
    print('方向预览：黄色=长轴，紫色=短轴/夹爪闭合方向；蓝色区域=提取顶面。')
    print('S=保存诊断快照，Q=退出。仅支持单个平放、顶面几何有效且周围桌面可见的目标。')
    if args.publish_grasp_target:
        print(f'方向抓取输出：{grasp_output}；TARGET_READY=当前帧几何有效，不等待8帧稳定。', flush=True)
    else:
        print(f'绿色 STABLE=连续8帧稳定；仅预览，不写抓取目标。诊断输出：{OUTPUT}', flush=True)
    try:
        camera.start()
        while True:
            frames = camera.get_frames()
            capture_stamp = time.time()
            status = 'NO_CAMERA_FRAME'
            result, mask, bbox = None, None, None
            valid = False
            if frames is None:
                stability.clear()
                if args.publish_grasp_target:
                    write_grasp_target(grasp_output, 'INVALID NO_CAMERA_FRAME\n')
                write_status({'stamp': time.time(), 'preview_only': not args.publish_grasp_target, 'valid': False, 'reason': status})
                missing = np.zeros((240, 640, 3), dtype=np.uint8)
                cv2.putText(missing, 'NO_CAMERA_FRAME - orientation invalid', (15, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow('Brick orientation preview', missing)
                if cv2.waitKey(1) & 0xff in (ord('q'), ord('Q')):
                    break
                continue
            image = frames['rgb'].copy()
            depth = frames['depth']
            original = image.copy()
            try:
                if depth.shape != image.shape[:2]:
                    raise InvalidObservation('RGB_DEPTH_RESOLUTION_MISMATCH')
                prediction = model.predict(image, conf=0.50, device='cpu', verbose=False)[0]
                boxes = [box for box in prediction.boxes
                         if model.names[int(box.cls[0])] == args.class_name]
                if len(boxes) != 1:
                    raise InvalidObservation('NEED_EXACTLY_ONE_TARGET')
                bbox = boxes[0].xyxy[0].cpu().numpy().astype(int)
                cv2.rectangle(image, tuple(bbox[:2]), tuple(bbox[2:]), (0, 200, 0), 2)
                result, mask = estimate(depth, bbox, camera.intrinsics, transform)
                if args.publish_grasp_target:
                    if time.time() - capture_stamp > 1.0:
                        raise InvalidObservation('FRAME_PROCESSING_TOO_SLOW')
                    valid, status = True, 'TARGET_READY'
                else:
                    valid = stability.update(result, time.monotonic())
                    status = 'STABLE' if valid else f'WAIT_STABLE {len(stability.history)}/8'
                draw(image, result, mask, camera.intrinsics, transform)
            except InvalidObservation as error:
                stability.clear()
                valid = False
                result = None
                status = str(error)
            if args.publish_grasp_target:
                write_grasp_target(grasp_output,
                                   oriented_target_line(capture_stamp, result, float(boxes[0].conf[0]),
                                                        args.class_name)
                                   if valid else f'INVALID {status}\n')
            write_status({'stamp': time.time(), 'preview_only': not args.publish_grasp_target, 'valid': valid,
                          'reason': status, 'estimate': result})
            if status != last_status:
                print(status, flush=True)
                last_status = status
            lines = [('GRASP TARGET LIVE - local +45 deg closing axis' if args.publish_grasp_target
                      else 'PREVIEW ONLY - no robot motion'), status,
                     'YELLOW: long axis | MAGENTA: closing axis', 'S: snapshot | Q: quit']
            if result:
                lines += [f"base XY long={result['long_axis_deg_mod180']:.1f} deg  short={result['short_axis_deg_mod180']:.1f} deg",
                          f"L/W={result['length_m']*1000:.0f}/{result['width_m']*1000:.0f} mm  height={result['height_above_table_m']*1000:.0f} mm"]
            for i, line in enumerate(lines):
                cv2.putText(image, line, (12, 26 + 25 * i), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, (0, 255, 0) if valid else (0, 190, 255), 2)
            cv2.imshow('Brick orientation preview', image)
            key = cv2.waitKey(1) & 0xff
            if key in (ord('q'), ord('Q')):
                break
            if key in (ord('s'), ord('S')) and bbox is not None:
                args.snapshots.mkdir(parents=True, exist_ok=True)
                prefix = args.snapshots / f'brick_{time.time_ns()}'
                np.savez_compressed(str(prefix) + '.npz', rgb=original, depth_mm=depth,
                                    bbox=bbox, intrinsics_json=json.dumps(camera.intrinsics),
                                    transform=transform)
                cv2.imwrite(str(prefix) + '.png', image)
                print(f'已保存 {prefix}.npz 和 .png', flush=True)
    finally:
        if args.publish_grasp_target:
            write_grasp_target(grasp_output, 'INVALID STOPPED\n')
        write_status({'stamp': time.time(), 'preview_only': not args.publish_grasp_target, 'valid': False, 'reason': 'STOPPED'})
        camera.stop()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
