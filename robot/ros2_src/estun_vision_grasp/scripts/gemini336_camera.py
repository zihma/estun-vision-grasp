import cv2
import numpy as np
from pyorbbecsdk import *


class GeminiCamera:
    """
    Gemini 336 相机模块

    输出：
        rgb       : OpenCV BGR 彩色图
        depth     : 对齐到 RGB 坐标系的深度图，单位 mm
        left_ir   : 左 IR 灰度图
        right_ir  : 右 IR 灰度图
        timestamps: 四路时间戳
    """

    def __init__(self):
        self.pipeline = Pipeline()
        self.config = Config()

        self.started = False
        self.intrinsics = None
        self.color_profile = None
        self.depth_profile = None
        self.left_profile = None
        self.right_profile = None

    # =========================================================
    # 找一个与 Depth 分辨率和 FPS 完全相同的 IR Profile
    # =========================================================
    @staticmethod
    def _find_matching_ir_profile(
            profile_list,
            width,
            height,
            fps
    ):

        # 第一轮：优先 Y8
        for i in range(len(profile_list)):

            profile = profile_list[i]

            if (
                    profile.get_width() == width
                    and profile.get_height() == height
                    and profile.get_fps() == fps
                    and profile.get_format() == OBFormat.Y8
            ):
                return profile

        # 第二轮：如果没有 Y8，只要求尺寸和 FPS 一致
        for i in range(len(profile_list)):

            profile = profile_list[i]

            if (
                    profile.get_width() == width
                    and profile.get_height() == height
                    and profile.get_fps() == fps
            ):
                return profile

        return None

    # =========================================================
    # 启动相机
    # =========================================================
    def start(self):

        # 获取三类 Profile 列表
        color_profiles = (
            self.pipeline.get_stream_profile_list(
                OBSensorType.COLOR_SENSOR
            )
        )

        left_profiles = (
            self.pipeline.get_stream_profile_list(
                OBSensorType.LEFT_IR_SENSOR
            )
        )

        right_profiles = (
            self.pipeline.get_stream_profile_list(
                OBSensorType.RIGHT_IR_SENSOR
            )
        )

        found = False

        # =====================================================
        # 自动寻找：
        #
        # Color
        #   ↓
        # 支持硬件 D2C 的 Depth
        #   ↓
        # 与 Depth 完全相同尺寸/FPS 的 Left IR + Right IR
        # =====================================================
        for i in range(len(color_profiles)):

            candidate_color = color_profiles[i]

            # 优先使用 RGB 格式
            if candidate_color.get_format() != OBFormat.RGB:
                continue

            try:

                d2c_depth_profiles = (
                    self.pipeline.get_d2c_depth_profile_list(
                        candidate_color,
                        OBAlignMode.HW_MODE
                    )
                )

            except Exception:
                continue

            if len(d2c_depth_profiles) == 0:
                continue

            for j in range(len(d2c_depth_profiles)):

                candidate_depth = (
                    d2c_depth_profiles[j]
                )

                depth_width = (
                    candidate_depth.get_width()
                )

                depth_height = (
                    candidate_depth.get_height()
                )

                depth_fps = (
                    candidate_depth.get_fps()
                )

                # 找匹配的 Left IR
                candidate_left = (
                    self._find_matching_ir_profile(
                        left_profiles,
                        depth_width,
                        depth_height,
                        depth_fps
                    )
                )

                if candidate_left is None:
                    continue

                # 找匹配的 Right IR
                candidate_right = (
                    self._find_matching_ir_profile(
                        right_profiles,
                        depth_width,
                        depth_height,
                        depth_fps
                    )
                )

                if candidate_right is None:
                    continue

                # 找到了完整兼容组合
                self.color_profile = (
                    candidate_color
                )

                self.depth_profile = (
                    candidate_depth
                )

                self.left_profile = (
                    candidate_left
                )

                self.right_profile = (
                    candidate_right
                )

                found = True
                break

            if found:
                break

        # =====================================================
        # 没找到兼容组合
        # =====================================================
        if not found:

            raise RuntimeError(
                "没有找到同时满足 "
                "D2C + Depth/Left IR/Right IR "
                "同分辨率同帧率的配置"
            )

        # =====================================================
        # 打印最终选择的配置
        # =====================================================
        print()
        print("========== 最终相机配置 ==========")

        print(
            "Color：",
            self.color_profile.get_width(),
            "x",
            self.color_profile.get_height(),
            "@",
            self.color_profile.get_fps(),
            "FPS"
        )

        print(
            "Depth：",
            self.depth_profile.get_width(),
            "x",
            self.depth_profile.get_height(),
            "@",
            self.depth_profile.get_fps(),
            "FPS"
        )

        print(
            "Left IR：",
            self.left_profile.get_width(),
            "x",
            self.left_profile.get_height(),
            "@",
            self.left_profile.get_fps(),
            "FPS"
        )

        print(
            "Right IR：",
            self.right_profile.get_width(),
            "x",
            self.right_profile.get_height(),
            "@",
            self.right_profile.get_fps(),
            "FPS"
        )

        print("=================================")
        print()

        # =====================================================
        # 开启四路
        # =====================================================
        self.config.enable_stream(
            self.color_profile
        )

        self.config.enable_stream(
            self.depth_profile
        )

        self.config.enable_stream(
            self.left_profile
        )

        self.config.enable_stream(
            self.right_profile
        )

        # Depth → RGB 硬件对齐
        self.config.set_align_mode(
            OBAlignMode.HW_MODE
        )

        # 四路全部存在才输出
        self.config.set_frame_aggregate_output_mode(
            OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE
        )

        # 启动
        self.pipeline.start(
            self.config
        )
        # 获取RGB内参

        profile = (
            self.color_profile
        )

        video_intrinsic = (
            profile.as_video_stream_profile()
            .get_intrinsic()
        )

        self.intrinsics = {

            "fx": video_intrinsic.fx,

            "fy": video_intrinsic.fy,

            "cx": video_intrinsic.cx,

            "cy": video_intrinsic.cy
        }

        print(
            "RGB内参:",
            self.intrinsics
        )
        self.started = True

        print(
            "Gemini 336 相机模块启动成功"
        )

        print(
            "RGB + Aligned Depth + "
            "Left IR + Right IR"
        )

    # =========================================================
    # Color → BGR
    # =========================================================
    @staticmethod
    def _color_to_bgr(color_frame):

        width = color_frame.get_width()
        height = color_frame.get_height()

        color_format = (
            color_frame.get_format()
        )

        data = np.frombuffer(
            color_frame.get_data(),
            dtype=np.uint8
        )

        if color_format == OBFormat.MJPG:

            return cv2.imdecode(
                data,
                cv2.IMREAD_COLOR
            )

        elif color_format == OBFormat.RGB:

            image = data.reshape(
                (height, width, 3)
            )

            return cv2.cvtColor(
                image,
                cv2.COLOR_RGB2BGR
            )

        elif color_format == OBFormat.BGR:

            return data.reshape(
                (height, width, 3)
            )

        elif color_format == OBFormat.YUYV:

            image = data.reshape(
                (height, width, 2)
            )

            return cv2.cvtColor(
                image,
                cv2.COLOR_YUV2BGR_YUYV
            )

        return None

    # =========================================================
    # Depth → mm
    # =========================================================
    @staticmethod
    def _depth_to_mm(depth_frame):

        width = depth_frame.get_width()
        height = depth_frame.get_height()

        depth_data = np.frombuffer(
            depth_frame.get_data(),
            dtype=np.uint16
        ).reshape(
            (height, width)
        )

        depth_scale = (
            depth_frame.get_depth_scale()
        )

        return (
            depth_data.astype(np.float32)
            * depth_scale
        )

    # =========================================================
    # IR 转换
    # =========================================================
    @staticmethod
    def _ir_to_image(ir_frame):

        width = ir_frame.get_width()
        height = ir_frame.get_height()

        ir_format = (
            ir_frame.get_format()
        )

        if ir_format == OBFormat.Y8:

            data = np.frombuffer(
                ir_frame.get_data(),
                dtype=np.uint8
            )

            return data.reshape(
                (height, width)
            )

        elif ir_format == OBFormat.Y16:

            data = np.frombuffer(
                ir_frame.get_data(),
                dtype=np.uint16
            ).reshape(
                (height, width)
            )

            image = cv2.normalize(
                data,
                None,
                0,
                255,
                cv2.NORM_MINMAX
            ).astype(np.uint8)

            return image

        return None

    # =========================================================
    # 获取一组数据
    # =========================================================
    def get_frames(
            self,
            timeout_ms=1000
    ):

        if not self.started:

            raise RuntimeError(
                "相机尚未启动，请先调用 start()"
            )

        frames = (
            self.pipeline.wait_for_frames(
                timeout_ms
            )
        )

        if frames is None:
            return None

        color_frame = (
            frames.get_color_frame()
        )

        depth_frame = (
            frames.get_depth_frame()
        )

        left_ir_frame = (
            frames.get_left_ir_frame()
        )

        right_ir_frame = (
            frames.get_right_ir_frame()
        )

        if (
                color_frame is None
                or depth_frame is None
                or left_ir_frame is None
                or right_ir_frame is None
        ):
            return None

        rgb = self._color_to_bgr(
            color_frame
        )

        depth_mm = self._depth_to_mm(
            depth_frame
        )

        left_ir = self._ir_to_image(
            left_ir_frame
        )

        right_ir = self._ir_to_image(
            right_ir_frame
        )

        if (
                rgb is None
                or depth_mm is None
                or left_ir is None
                or right_ir is None
        ):
            return None

        timestamps = {

            "rgb":
                color_frame.get_timestamp_us(),

            "depth":
                depth_frame.get_timestamp_us(),

            "left_ir":
                left_ir_frame.get_timestamp_us(),

            "right_ir":
                right_ir_frame.get_timestamp_us()
        }

        return {

            "rgb": rgb,

            "depth": depth_mm,

            "left_ir": left_ir,

            "right_ir": right_ir,

            "timestamps": timestamps
        }

    # =========================================================
    # RGB坐标附近 7×7 深度中位数
    # =========================================================
    @staticmethod
    def get_distance(
            depth_mm,
            x,
            y,
            radius=3
    ):

        if depth_mm is None:
            return None

        height, width = (
            depth_mm.shape
        )

        x = int(x)
        y = int(y)

        if not (
                0 <= x < width
                and 0 <= y < height
        ):
            return None

        x1 = max(
            0,
            x - radius
        )

        x2 = min(
            width,
            x + radius + 1
        )

        y1 = max(
            0,
            y - radius
        )

        y2 = min(
            height,
            y + radius + 1
        )

        region = depth_mm[
            y1:y2,
            x1:x2
        ]

        valid_depth = (
            region[region > 0]
        )

        if valid_depth.size == 0:
            return None

        return float(
            np.median(valid_depth)
        )

    # =========================================================
    # 根据YOLO bbox求距离
    # =========================================================
    @staticmethod
    def get_bbox_distance(
            depth_mm,
            x1,
            y1,
            x2,
            y2
    ):

        center_x = int(
            (x1 + x2) / 2
        )

        center_y = int(
            (y1 + y2) / 2
        )

        return GeminiCamera.get_distance(
            depth_mm,
            center_x,
            center_y,
            radius=3
        )

    # =========================================================
    # 关闭
    # =========================================================
    def stop(self):

        if self.started:

            self.pipeline.stop()

            self.started = False

            print(
                "Gemini 336 已关闭"
            )