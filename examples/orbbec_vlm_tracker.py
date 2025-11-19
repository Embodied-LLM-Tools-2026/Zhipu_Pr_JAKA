"""
Orbbec RGB capture + VLM guided CSRT tracking demo.

Usage example:
  python examples/orbbec_vlm_tracker.py --target-name "红色水杯"

Prerequisites:
  - pyorbbecsdk correctly installed and the camera connected.
  - OpenCV built with the CSRT tracker (opencv-contrib).
  - dashscope + tools/vision/upload_image configured with Zhipu_real_demo_API_KEY.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import cv2
import dashscope  # type: ignore
import numpy as np

try:
    from pyorbbecsdk import (  # type: ignore
        Config as OBCfg,
        FormatConvertFilter,
        OBConvertFormat,
        OBFormat,
        OBSensorType,
        Pipeline,
    )
    PYORBBEC_AVAILABLE = True
except Exception:  # pragma: no cover
    OBCfg = None
    FormatConvertFilter = None
    OBConvertFormat = None
    OBFormat = None
    OBSensorType = None
    Pipeline = None
    PYORBBEC_AVAILABLE = False

from voice.utils.config import Config as AppConfig  # type: ignore
from tools.vision.upload_image import upload_file_and_get_url  # type: ignore


def _i420_to_bgr(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    y = frame[0:height, :]
    u = frame[height:height + height // 4].reshape(height // 2, width // 2)
    v = frame[height + height // 4:].reshape(height // 2, width // 2)
    yuv_image = cv2.merge([y, u, v])
    return cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR_I420)


def _nv21_to_bgr(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    y = frame[0:height, :]
    uv = frame[height:height + height // 2].reshape(height // 2, width)
    yuv_image = cv2.merge([y, uv])
    return cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR_NV21)


def _nv12_to_bgr(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    y = frame[0:height, :]
    uv = frame[height:height + height // 2].reshape(height // 2, width)
    yuv_image = cv2.merge([y, uv])
    return cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR_NV12)


def _determine_convert_format(frame) -> Optional[int]:
    if OBFormat is None or OBConvertFormat is None or frame is None:
        return None
    fmt = frame.get_format()
    if fmt == OBFormat.I420:
        return OBConvertFormat.I420_TO_RGB888
    if fmt == OBFormat.MJPG:
        return OBConvertFormat.MJPG_TO_RGB888
    if fmt == OBFormat.YUYV:
        return OBConvertFormat.YUYV_TO_RGB888
    if fmt == OBFormat.UYVY:
        return OBConvertFormat.UYVY_TO_RGB888
    if fmt == OBFormat.NV21:
        return OBConvertFormat.NV21_TO_RGB888
    if fmt == OBFormat.NV12:
        return OBConvertFormat.NV12_TO_RGB888
    return None


def _frame_to_bgr_image(frame) -> Optional[np.ndarray]:
    if frame is None or OBFormat is None:
        return None
    width = frame.get_width()
    height = frame.get_height()
    color_format = frame.get_format()
    data = np.asanyarray(frame.get_data())

    if color_format == OBFormat.RGB:
        image = np.resize(data, (height, width, 3))
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if color_format == OBFormat.BGR:
        return np.resize(data, (height, width, 3))
    if color_format == OBFormat.YUYV:
        image = np.resize(data, (height, width, 2))
        return cv2.cvtColor(image, cv2.COLOR_YUV2BGR_YUYV)
    if color_format == OBFormat.MJPG:
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    if color_format == OBFormat.I420:
        return _i420_to_bgr(data, width, height)
    if color_format == OBFormat.NV12:
        return _nv12_to_bgr(data, width, height)
    if color_format == OBFormat.NV21:
        return _nv21_to_bgr(data, width, height)
    if color_format == OBFormat.UYVY:
        image = np.resize(data, (height, width, 2))
        return cv2.cvtColor(image, cv2.COLOR_YUV2BGR_UYVY)

    convert_format = _determine_convert_format(frame)
    if convert_format is None or FormatConvertFilter is None:
        return None
    convert_filter = FormatConvertFilter()
    convert_filter.set_format_convert_format(convert_format)
    rgb_frame = convert_filter.process(frame)
    if rgb_frame is None:
        return None
    return _frame_to_bgr_image(rgb_frame)


class OrbbecColorCamera:
    """Minimal Orbbec color pipeline wrapper."""

    def __init__(self, enable_depth: bool = False) -> None:
        if not PYORBBEC_AVAILABLE or None in (Pipeline, OBCfg, OBSensorType):
            raise RuntimeError("pyorbbecsdk 不可用，请安装SDK或改用 --camera-index 指定普通摄像头")
        self.pipeline = Pipeline()
        self.config = OBCfg()
        self.last_failure_reason = ""
        self._configure(enable_depth)
        self.pipeline.start(self.config)

    def _configure(self, enable_depth: bool) -> None:
        for sensor_type in [OBSensorType.COLOR_SENSOR]:
            profile_list = self.pipeline.get_stream_profile_list(sensor_type)
            if profile_list is None:
                raise RuntimeError("无法获取彩色相机参数")
            profile = profile_list.get_default_video_stream_profile()
            if profile is None:
                raise RuntimeError("未找到彩色相机默认配置")
            self.config.enable_stream(profile)
        if enable_depth:
            try:
                depth_profiles = self.pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
                if depth_profiles:
                    depth_profile = depth_profiles.get_default_video_stream_profile()
                    if depth_profile:
                        self.config.enable_stream(depth_profile)
            except Exception:
                pass

    def _format_name(self, color_frame) -> str:
        if color_frame is None:
            return "unknown"
        try:
            fmt = color_frame.get_format()
            return getattr(fmt, "name", str(fmt))
        except Exception:
            return "unknown"

    def read(self, timeout_ms: int = 1000) -> Optional[np.ndarray]:
        frames = self.pipeline.wait_for_frames(timeout_ms)
        if frames is None:
            self.last_failure_reason = "wait_for_frames 超时"
            return None
        color_frame = frames.get_color_frame()
        if color_frame is None:
            self.last_failure_reason = "未获取到彩色帧"
            return None
        image = _frame_to_bgr_image(color_frame)
        if image is None:
            fmt_name = self._format_name(color_frame)
            self.last_failure_reason = f"彩色帧格式 {fmt_name} 转换失败，请确认已安装opencv-contrib"
            return None
        self.last_failure_reason = ""
        return image

    def read_with_retry(self, attempts: int = 30, delay: float = 0.05, timeout_ms: int = 1000) -> Optional[np.ndarray]:
        """Try grabbing a frame multiple times to let auto-exposure/streams stabilize."""
        for _ in range(max(1, attempts)):
            frame = self.read(timeout_ms)
            if frame is not None:
                return frame
            time.sleep(delay)
        return None

    def last_error(self) -> str:
        return self.last_failure_reason

    def stop(self) -> None:
        try:
            self.pipeline.stop()
        except Exception:
            pass


class OpenCVCamera:
    """Fallback to a standard VideoCapture device."""

    def __init__(self, index: int):
        self.index = index
        self.cap = cv2.VideoCapture(index)
        if not self.cap or not self.cap.isOpened():
            raise RuntimeError(f"无法打开摄像头索引 {index}，请确认设备存在")
        self.last_failure_reason = ""

    def read(self, timeout_ms: int = 0) -> Optional[np.ndarray]:
        ok, frame = self.cap.read()
        if not ok:
            self.last_failure_reason = f"cv2.VideoCapture({self.index}) 读取失败"
            return None
        self.last_failure_reason = ""
        return frame

    def read_with_retry(self, attempts: int = 30, delay: float = 0.05, timeout_ms: int = 0) -> Optional[np.ndarray]:
        for _ in range(max(1, attempts)):
            frame = self.read(timeout_ms)
            if frame is not None:
                return frame
            time.sleep(delay)
        return None

    def last_error(self) -> str:
        return self.last_failure_reason

    def stop(self) -> None:
        if self.cap:
            self.cap.release()


def _create_csrt_tracker():
    if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerCSRT_create"):
        return cv2.legacy.TrackerCSRT_create()
    if hasattr(cv2, "TrackerCSRT_create"):
        return cv2.TrackerCSRT_create()
    raise RuntimeError("当前OpenCV不支持CSRT跟踪器，请安装opencv-contrib-python")


PROMPT_TEMPLATE = """你是部署在服务机器人上的视觉助手。目标物体是“{target}”。
请分析图像并返回结构化JSON，仅包含视觉信息，不要给出动作建议。
如果未发现目标，请保持found=false并给出简要中文分析。
字段说明：
{{
  "found": true/false,
  "bbox": [x_min, y_max, x_max, y_min],
  "confidence": number (0-1)
}}
禁止返回其他键。确保JSON合法。
"""


@dataclass
class DetectionResult:
    bbox_xyxy: Tuple[int, int, int, int]
    confidence: float
    raw: dict


class VLMDetector:
    """Wrap dashscope VLM detection for convenience."""

    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None):
        self.api_key = api_key or os.getenv("Zhipu_real_demo_API_KEY")
        if not self.api_key:
            raise RuntimeError("请先设置 Zhipu_real_demo_API_KEY 环境变量用于DashScope调用")
        self.model = model_name or AppConfig.VLM_NAME

    def detect(self, image_path: Path, target_name: str) -> DetectionResult:
        image_url = upload_file_and_get_url(self.api_key, self.model, str(image_path))
        prompt = PROMPT_TEMPLATE.format(target=target_name)
        response = dashscope.MultiModalConversation.call(
            api_key=self.api_key,
            model=self.model,
            messages=[{"role": "user", "content": [{"image": image_url}, {"text": prompt}]}],
            max_tokens=400,
            temperature=0.01,
            response_format={"type": "json_object"},
        )
        content = response.output.choices[0].message.content  # type: ignore[index]
        if isinstance(content, list):
            payload_text = next((c.get("text") for c in content if isinstance(c, dict) and "text" in c), "")
        elif isinstance(content, dict):
            payload_text = content.get("text", "")
        else:
            payload_text = str(content)
        data = json.loads(payload_text)
        bbox = data.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise RuntimeError(f"VLM 未返回有效 bbox: {data}")
        x_min, y_max, x_max, y_min = [float(v) for v in bbox]
        x1 = int(max(0, min(x_min, x_max)))
        x2 = int(max(0, max(x_min, x_max)))
        y1 = int(max(0, min(y_min, y_max)))
        y2 = int(max(0, max(y_min, y_max)))
        if x2 <= x1 or y2 <= y1:
            raise RuntimeError(f"VLM bbox 结果异常: {bbox}")
        confidence = float(data.get("confidence", 0.0) or 0.0)
        return DetectionResult(bbox_xyxy=(x1, y1, x2, y2), confidence=confidence, raw=data)


def _save_frame_to_jpeg(frame: np.ndarray, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"vlm_capture_{int(time.time() * 1000)}.jpg"
    if not cv2.imwrite(str(path), frame):
        raise RuntimeError(f"保存帧到 {path} 失败")
    return path


def _xyxy_to_xywh(bbox: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    return x1, y1, x2 - x1, y2 - y1


def _clip_bbox_to_frame(bbox: Tuple[int, int, int, int], frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    h, w = frame.shape[:2]
    x1 = int(max(0, min(bbox[0], w - 1)))
    x2 = int(max(0, min(bbox[2], w - 1)))
    y1 = int(max(0, min(bbox[1], h - 1)))
    y2 = int(max(0, min(bbox[3], h - 1)))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return (x1, y1, x2, y2)


def _draw_bbox(frame: np.ndarray, bbox_xyxy: Tuple[int, int, int, int], label: str) -> None:
    x1, y1, x2, y2 = bbox_xyxy
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame, label, (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


TARGET_RESOLUTION = (1000, 1000)


def _resize_to_target(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    target_w, target_h = TARGET_RESOLUTION
    if (w, h) == (target_w, target_h):
        return frame
    interpolation = cv2.INTER_AREA if w >= target_w and h >= target_h else cv2.INTER_LINEAR
    return cv2.resize(frame, TARGET_RESOLUTION, interpolation=interpolation)


def run_tracking_loop(args) -> None:
    if args.camera_index is not None and args.camera_index >= 0:
        camera = OpenCVCamera(args.camera_index)
        if args.enable_depth:
            print("⚠️ 使用普通摄像头时 --enable-depth 选项被忽略")
    else:
        camera = OrbbecColorCamera(enable_depth=args.enable_depth)
    vlm = VLMDetector(model_name=args.vlm_model)
    tracker = None
    bbox_xyxy: Optional[Tuple[int, int, int, int]] = None
    snapshot_dir = Path(args.snapshot_dir) if args.snapshot_dir else Path(tempfile.gettempdir()) / "orbbec_vlm"
    try:
        if args.warmup_frames > 0:
            for _ in range(args.warmup_frames):
                camera.read()
                time.sleep(0.01)

        frame = camera.read_with_retry(attempts=args.init_attempts, delay=0.05)
        if frame is None:
            hint = camera.last_error()
            raise RuntimeError(
                f"未读取到初始彩色帧，最近错误: {hint or '无'}；"
                f"请检查相机连接/驱动，或增大 --init-attempts (当前 {args.init_attempts})"
            )
        frame = _resize_to_target(frame)
        snapshot = _save_frame_to_jpeg(frame, snapshot_dir)
        detection = vlm.detect(snapshot, args.target_name)
        bbox_xyxy = _clip_bbox_to_frame(detection.bbox_xyxy, frame)
        if bbox_xyxy is None:
            raise RuntimeError(
                f"VLM 返回的bbox超出画面或尺寸过小: {detection.bbox_xyxy}，"
                "请确认拍摄到目标或调整提示词"
            )
        print(f"VLM 检测成功，bbox={bbox_xyxy}, confidence={detection.confidence:.2f}")
        tracker = _create_csrt_tracker()
        tracker.init(frame, _xyxy_to_xywh(bbox_xyxy))
        frame_count = 0
        start = time.time()
        while True:
            frame = camera.read()
            if frame is None:
                time.sleep(0.01)
                continue
            frame = _resize_to_target(frame)
            frame_count += 1
            ok, tracked_bbox = tracker.update(frame)
            if ok:
                x, y, w, h = tracked_bbox
                bbox_xyxy = (int(x), int(y), int(x + w), int(y + h))
                _draw_bbox(frame, bbox_xyxy, f"CSRT {detection.confidence:.2f}")
            else:
                cv2.putText(frame, "Tracking lost", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                if args.auto_redetect:
                    snapshot = _save_frame_to_jpeg(frame, snapshot_dir)
                    try:
                        detection = vlm.detect(snapshot, args.target_name)
                        bbox_xyxy = _clip_bbox_to_frame(detection.bbox_xyxy, frame)
                        if bbox_xyxy is None:
                            raise RuntimeError("VLM bbox 仍不在画面内")
                        tracker = _create_csrt_tracker()
                        tracker.init(frame, _xyxy_to_xywh(bbox_xyxy))
                        print("重新检测并初始化跟踪成功")
                    except Exception as exc:
                        print(f"重新检测失败: {exc}")
                else:
                    print("跟踪失败，退出")
                    break

            if args.show_window:
                fps = frame_count / max(1e-3, (time.time() - start))
                cv2.putText(frame, f"FPS: {fps:.1f}", (20, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.imshow("Orbbec VLM CSRT Tracker", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

            if args.max_frames and frame_count >= args.max_frames:
                break
    finally:
        camera.stop()
        if args.show_window:
            cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(description="使用奥比中光RGB相机 + VLM + CSRT 实时跟踪")
    parser.add_argument("--target-name", required=True, help="需识别的目标名称，用于VLM提示词")
    parser.add_argument("--vlm-model", default=AppConfig.VLM_NAME, help="DashScope VLM 模型名称")
    parser.add_argument("--snapshot-dir", default="", help="保存上传快照的目录，默认使用临时目录")
    parser.add_argument("--max-frames", type=int, default=0, help="最多处理多少帧，0表示无限")
    parser.add_argument("--auto-redetect", action="store_true", help="跟踪失败后自动重新调用VLM检测")
    parser.add_argument("--enable-depth", action="store_true", help="同时开启深度流（若后续要做3D定位）")
    parser.add_argument("--show-window", action="store_true", help="显示OpenCV窗口并可视化bbox")
    parser.add_argument("--camera-index", type=int, default=-1, help="指定普通OpenCV摄像头索引，>=0时跳过Orbbec SDK")
    parser.add_argument("--init-attempts", type=int, default=60, help="初始化阶段允许的读帧重试次数")
    parser.add_argument("--warmup-frames", type=int, default=15, help="启动后丢弃的暖机帧数")
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    run_tracking_loop(cli_args)
