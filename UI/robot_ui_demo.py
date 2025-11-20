import io, os, threading, time, base64, random, sys
from pathlib import Path
from typing import List, Tuple, Optional, Any, Dict

import numpy as np
from PIL import Image, ImageDraw
from fastapi import FastAPI, Response, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
# Ensure project root is on sys.path so imports like `from tools.xxx import ...` work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# 尝试使用摄像头；若不可用则用占位图
try:
  import cv2
except Exception:
  cv2 = None

try:
  from pyorbbecsdk import (
    AlignFilter,
    Config,
    Pipeline,
    OBSensorType,
    OBFormat,
    OBStreamType,
    OBError,
    FormatConvertFilter,
    VideoFrame,
    OBConvertFormat,
    transformation2dto3d,
    OBPoint2f,
  )
except Exception as exc:  # noqa: F841
  AlignFilter = None
  Config = None
  Pipeline = None
  OBSensorType = None
  OBFormat = None
  OBStreamType = None
  OBError = Exception  # type: ignore
  FormatConvertFilter = None
  VideoFrame = None
  OBConvertFormat = None

FRONT_WIDTH = 1280
FRONT_HEIGHT = 800
HAS_ORBBEC = all(
  obj is not None for obj in (Config, Pipeline, OBSensorType, OBFormat, OBError)
) and cv2 is not None


def _create_placeholder_image(name: str, width: int, height: int) -> np.ndarray:
  """Generate a simple grid placeholder when no live frame is available."""
  img = Image.new("RGB", (width, height), (240, 240, 240))
  draw = ImageDraw.Draw(img)
  step = max(40, min(width, height) // 20)
  for x in range(0, width, step):
    draw.line([(x, 0), (x, height)], fill=(200, 200, 200))
  for y in range(0, height, step):
    draw.line([(0, y), (width, y)], fill=(200, 200, 200))
  draw.text((20, 20), f"Camera: {name} (placeholder)", fill=(20, 20, 20))
  return np.array(img, dtype=np.uint8)


def _serialize_intrinsics(video_profile) -> Dict[str, Any]:
  """Convert Orbbec video/intrinsic profile into a JSON-friendly dict."""
  if video_profile is None:
    return {}
  data: Dict[str, Any] = {}
  for getter_name, key in (("get_width", "width"), ("get_height", "height"), ("get_fps", "fps")):
    getter = getattr(video_profile, getter_name, None)
    if callable(getter):
      try:
        data[key] = int(getter())
      except Exception:
        pass
  try:
    intr = video_profile.get_intrinsic()
  except Exception:
    intr = None
  if intr is not None:
    for attr in ("fx", "fy", "cx", "cy"):
      if hasattr(intr, attr):
        try:
          data[attr] = float(getattr(intr, attr))
        except Exception:
          pass
    model = getattr(intr, "distortion_model", None)
    if model is not None:
      data["distortion_model"] = model
    coeffs = getattr(intr, "distortion_coeffs", None)
    if coeffs is not None:
      try:
        data["distortion_coeffs"] = [float(c) for c in coeffs]
      except Exception:
        pass
  return data


def _serialize_extrinsic(extrinsic) -> Optional[Dict[str, Any]]:
  """Convert Orbbec extrinsic data to dict and homogeneous matrix."""
  if extrinsic is None:
    return None

  def _value(name: str):
    attr = getattr(extrinsic, name, None)
    if attr is None:
      return None
    return attr() if callable(attr) else attr

  rotation = None
  for candidate in ("rotation", "get_rotation", "rot", "get_rot", "rotation_matrix", "get_rotation_matrix"):
    rotation = _value(candidate)
    if rotation is not None:
      break

  translation = None
  for candidate in ("translation", "get_translation", "trans", "get_trans", "t"):
    translation = _value(candidate)
    if translation is not None:
      break

  transform = None
  if rotation is None or translation is None:
    for candidate in ("transform", "get_transform", "matrix", "get_matrix"):
      transform = _value(candidate)
      if transform is not None:
        break

  rot_arr = None
  trans_arr = None
  if rotation is not None:
    try:
      rot_arr = np.asarray(rotation, dtype=float)
      if rot_arr.size == 9:
        rot_arr = rot_arr.reshape(3, 3)
    except Exception:
      rot_arr = None
  if translation is not None:
    try:
      trans_arr = np.asarray(translation, dtype=float)
      if trans_arr.size == 3:
        trans_arr = trans_arr.reshape(3)
    except Exception:
      trans_arr = None

  if (rot_arr is None or trans_arr is None) and transform is not None:
    try:
      tf_arr = np.asarray(transform, dtype=float).ravel()
      size = tf_arr.size
      if size == 3 and trans_arr is None:
        trans_arr = tf_arr.reshape(3)
      elif size == 9 and rot_arr is None:
        rot_arr = tf_arr.reshape(3, 3)
      elif size == 12:
        mat34 = tf_arr.reshape(3, 4)
        if rot_arr is None:
          rot_arr = mat34[:, :3]
        if trans_arr is None:
          trans_arr = mat34[:, 3]
      elif size == 16:
        mat4 = tf_arr.reshape(4, 4)
        if rot_arr is None:
          rot_arr = mat4[:3, :3]
        if trans_arr is None:
          trans_arr = mat4[:3, 3]
    except Exception:
      pass

  if rot_arr is None or trans_arr is None:
    return None

  mat = np.eye(4, dtype=float)
  mat[:3, :3] = rot_arr
  mat[:3, 3] = trans_arr
  return {
    "rotation": rot_arr.tolist(),
    "translation": trans_arr.tolist(),
    "matrix": mat.tolist(),
  }


def _yuyv_to_bgr(frame: np.ndarray, width: int, height: int) -> np.ndarray:
  yuyv = frame.reshape((height, width, 2))
  return cv2.cvtColor(yuyv, cv2.COLOR_YUV2BGR_YUY2)


def _uyvy_to_bgr(frame: np.ndarray, width: int, height: int) -> np.ndarray:
  uyvy = frame.reshape((height, width, 2))
  return cv2.cvtColor(uyvy, cv2.COLOR_YUV2BGR_UYVY)


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


def _determine_convert_format(frame: Any):
  if OBFormat is None or OBConvertFormat is None or frame is None:
    return None
  fmt = frame.get_format()
  if fmt == OBFormat.I420:
    return OBConvertFormat.I420_TO_RGB888
  if fmt == OBFormat.MJPG:
    return OBConvertFormat.MJPG_TO_RGB888
  if fmt == OBFormat.YUYV:
    return OBConvertFormat.YUYV_TO_RGB888
  if fmt == OBFormat.NV21:
    return OBConvertFormat.NV21_TO_RGB888
  if fmt == OBFormat.NV12:
    return OBConvertFormat.NV12_TO_RGB888
  if fmt == OBFormat.UYVY:
    return OBConvertFormat.UYVY_TO_RGB888
  return None


def _frame_to_bgr_image(frame: Any) -> Optional[np.ndarray]:
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
    image = np.resize(data, (height, width, 3))
    return image
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

# AGV 实时位置存储（后端通过 API 接口更新）
AGV_POSE = {
    "theta": 0.0,      # 角度（度）
    "x": 0.0,          # x 坐标（米）
    "y": 0.0,          # y 坐标（米）
    "timestamp": int(time.time() * 1000),
}

APP = FastAPI(title="Robot Monitor Demo")

# 目录：静态文件与截帧输出
BASE = Path(__file__).resolve().parent
STATIC_DIR = BASE / "static"
CAP_DIR = STATIC_DIR / "captures"
os.makedirs(CAP_DIR, exist_ok=True)

# ============ 摄像头/帧源 ============

class FrameSource:
    """简易帧源：优先用本机摄像头，支持备用索引，否则生成占位图"""
    def __init__(self, cam_index: int = 0, name: str = "front", fallback_indexes: list = None):
        self.name = name
        self.cap = None
        self.frame = None
        self.lock = threading.Lock()
        self.running = True

        # 摄像头索引列表：先尝试主索引，再尝试备用索引
        indexes_to_try = [cam_index]
        if fallback_indexes:
            indexes_to_try.extend(fallback_indexes)
        
        if cv2 is not None:
            for idx in indexes_to_try:
                try:
                    cap = cv2.VideoCapture(idx)
                    if cap is not None and cap.isOpened():
                        self.cap = cap
                        print(f"✅ Camera '{self.name}' opened with index {idx}")
                        break
                    else:
                        print(f"⚠️  Camera '{self.name}' index {idx} failed to open")
                        if cap is not None:
                            cap.release()
                except Exception as e:
                    print(f"❌ Error opening camera '{self.name}' index {idx}: {e}")
            
            if self.cap is None:
                print(f"⚠️  All camera indexes failed for '{self.name}', will use placeholder")
        
        # 后台线程循环取帧
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        while self.running:
            if self.cap is not None and self.cap.isOpened():
                ok, frame = self.cap.read()
                if ok:
                    # BGR -> RGB
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    with self.lock:
                        self.frame = frame
                else:
                    self._gen_placeholder()
            else:
                self._gen_placeholder()
            time.sleep(0.05)

    def _gen_placeholder(self, W=960, H=540):
        placeholder = _create_placeholder_image(self.name, W, H)
        with self.lock:
            self.frame = placeholder

    def get_jpeg_bytes(self, w: Optional[int] = None, h: Optional[int] = None) -> bytes:
        with self.lock:
            frame = None if self.frame is None else self.frame.copy()

        if frame is None:
            # 无可用帧时生成占位符
            self._gen_placeholder(W=w or 960, H=h or 540)
            with self.lock:
                frame = self.frame.copy()

        img = Image.fromarray(frame)
        if w is not None and h is not None and img.size != (w, h):
            img = img.resize((w, h), Image.BILINEAR)

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
        return buf.getvalue()

    def capture_and_save(self) -> str:
        """保存一张JPEG到 /static/captures，并返回相对URL"""
        jpeg = self.get_jpeg_bytes()
        ts = int(time.time() * 1000)
        out_path = CAP_DIR / f"{self.name}_{ts}.jpg"
        with open(out_path, "wb") as f:
            f.write(jpeg)
        # 用于前端展示，返回静态资源挂载路径
        rel_url = f"/static/captures/{out_path.name}"
        return rel_url

    def stop(self):
        self.running = False
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass

class OrbbecStreamBase:
    def __init__(self, name: str):
        self.name = name
        self.pipeline: Any = None
        self.config: Any = None
        self.align_filter: Optional[Any] = None
        self.thread: Optional[threading.Thread] = None
        self.running = False
        self.lock = threading.Lock()
        self.frame = _create_placeholder_image(name, FRONT_WIDTH, FRONT_HEIGHT)
        self.available = HAS_ORBBEC
        self.color_frame: Any = None
        self.depth_frame: Any = None
        self.surface_points_hint: Optional[List[List[float]]] = None
    def _build_stream(self) -> Tuple[Any, Any]:
        raise NotImplementedError

    def _extract_frame(self, frames: Any) -> Optional[np.ndarray]:
        raise NotImplementedError

    def start(self) -> bool:
        if not self.available:
            return False
        if self.running:
            return True
        try:
            if self.pipeline is None or self.config is None:
                self.pipeline, self.config = self._build_stream()
            if self.pipeline is None or self.config is None:
                raise RuntimeError("stream not configured")
            self.pipeline.start(self.config)
            self.running = True
            self.thread = threading.Thread(target=self._loop, daemon=True)
            self.thread.start()
            return True
        except Exception as exc:
            print(f"⚠️  Unable to start stream '{self.name}': {exc}")
            self.available = False
            self.running = False
            return False

    def stop(self):
        self.running = False
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=0.2)
        if self.pipeline is not None:
            try:
                self.pipeline.stop()
            except Exception:
                pass

    def _loop(self):
        while self.running:
            try:
                if self.pipeline is None:
                    time.sleep(0.05)
                    continue
                frames = self.pipeline.wait_for_frames(100)
                if frames is None:
                    continue
                if self.align_filter is not None:
                    try:
                        aligned = self.align_filter.process(frames)
                    except Exception as exc:
                        print(f"⚠️  Align filter failed: {exc}")
                        aligned = None
                    if not aligned:
                        continue
                    try:
                        frames = aligned.as_frame_set()
                    except AttributeError:
                        frames = aligned
                image = self._extract_frame(frames)
                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()
                if color_frame is not None:
                    self.color_frame = color_frame
                if depth_frame is not None:
                    self.depth_frame = depth_frame
                if image is None:
                    continue
                if image.shape[1] != FRONT_WIDTH or image.shape[0] != FRONT_HEIGHT:
                    image = cv2.resize(image, (FRONT_WIDTH, FRONT_HEIGHT), interpolation=cv2.INTER_AREA)
                image = np.clip(image, 0, 255).astype(np.uint8)
                with self.lock:
                    self.frame = image
            except Exception as exc:
                print(f"⚠️  Stream '{self.name}' loop error: {exc}")
                time.sleep(0.05)

    @staticmethod
    def _to_serializable(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.floating, np.integer)):
            return value.item()
        if isinstance(value, dict):
            return {k: OrbbecStreamBase._to_serializable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [OrbbecStreamBase._to_serializable(v) for v in value]
        return value

    def set_surface_points_hint(self, points: Optional[List[List[float]]]) -> None:
        if points and isinstance(points, list):
            try:
                self.surface_points_hint = [
                    [float(p[0]), float(p[1])] for p in points if isinstance(p, (list, tuple)) and len(p) >= 2
                ]
                if self.surface_points_hint:
                    return
            except (TypeError, ValueError):
                pass
        self.surface_points_hint = None

    def get_jpeg_bytes(self, w: Optional[int] = None, h: Optional[int] = None) -> bytes:
        width = w or FRONT_WIDTH
        height = h or FRONT_HEIGHT
        with self.lock:
            frame = None if self.frame is None else self.frame.copy()
        if frame is None:
            frame = _create_placeholder_image(self.name, width, height)
        elif frame.shape[1] != width or frame.shape[0] != height:
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        img = Image.fromarray(frame)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
        return buf.getvalue()

    def capture_and_save(self) -> str:
        jpeg = self.get_jpeg_bytes(FRONT_WIDTH, FRONT_HEIGHT)
        ts = int(time.time() * 1000)
        out_path = CAP_DIR / f"{self.name}_{ts}.jpg"
        with open(out_path, "wb") as f:
            f.write(jpeg)
        return f"/static/captures/{out_path.name}"
class OrbbecColorStream(OrbbecStreamBase):
    def __init__(self, name: str, *, align_to_color: bool = True):
        super().__init__(name)
        self.align_to_color = bool(align_to_color and HAS_ORBBEC and AlignFilter is not None and OBStreamType is not None)
        if self.align_to_color:
            try:
                self.align_filter = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)
                print("[Orbbec] Depth-to-color alignment enabled for", name)
            except Exception as exc:
                print(f"[Orbbec] Failed to create AlignFilter: {exc}")
                self.align_filter = None

    def _build_stream(self) -> Tuple[Any, Any]:
        if not HAS_ORBBEC:
            raise RuntimeError("Orbbec SDK unavailable")
        pipeline = Pipeline()
        config = Config()
        try:
          # Enable depth and color sensors
          for sensor_type in [OBSensorType.DEPTH_SENSOR, OBSensorType.COLOR_SENSOR]:
              profile_list = pipeline.get_stream_profile_list(sensor_type)
              assert profile_list is not None
              profile = profile_list.get_default_video_stream_profile()
              assert profile is not None
              print(f"{sensor_type} profile:", profile)
              config.enable_stream(profile)  # Enable the stream for the sensor
        except Exception as e:
            print(e)
            return
        if self.align_to_color and hasattr(pipeline, "enable_frame_sync"):
            try:
                pipeline.enable_frame_sync()
            except Exception as exc:
                print(f"[Orbbec] enable_frame_sync failed: {exc}")
        return pipeline, config

    def _extract_frame(self, frames: Any) -> Optional[np.ndarray]:
        if cv2 is None:
            return None
        color_frame = frames.get_color_frame()
        if color_frame is None:
            return None
        bgr = _frame_to_bgr_image(color_frame)
        if bgr is None:
            return None
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

class FrontCameraMultiplexer:
    def __init__(self):
        self.lock = threading.Lock()
        self.streams = {
            "color": OrbbecColorStream("front_color"),
        }
        self.active_key = "color"
        if HAS_ORBBEC:
            self.streams[self.active_key].start()

    def set_active(self, key: str) -> str:
        if key not in self.streams:
            raise ValueError(f"Unknown front stream '{key}'")
        with self.lock:
            current_key = self.active_key
            if key == current_key:
                return self.active_key
        current_stream = self.streams.get(current_key)
        if current_stream is not None:
            current_stream.stop()
        target_stream = self.streams[key]
        if not target_stream.start():
            print(f"⚠️  Failed to activate front stream '{key}', reverting to '{current_key}'")
            if current_stream is not None:
                current_stream.start()
            return current_key
        with self.lock:
            self.active_key = key
        return self.active_key

    def get_jpeg_bytes(self, w: Optional[int] = None, h: Optional[int] = None) -> bytes:
        width = w or FRONT_WIDTH
        height = h or FRONT_HEIGHT
        stream = self.streams.get(self.active_key)
        if stream is None:
            placeholder = _create_placeholder_image(self.active_key, width, height)
            img = Image.fromarray(placeholder)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=90)
            return buf.getvalue()
        return stream.get_jpeg_bytes(width, height)

    def capture_and_save(self) -> str:
        stream = self.streams.get(self.active_key)
        if stream is None:
            jpeg = self.get_jpeg_bytes(FRONT_WIDTH, FRONT_HEIGHT)
            ts = int(time.time() * 1000)
            out_path = CAP_DIR / f"front_{self.active_key}_{ts}.jpg"
            with open(out_path, "wb") as f:
                f.write(jpeg)
            return f"/static/captures/{out_path.name}"
        return stream.capture_and_save()

    def stop(self):
        for stream in self.streams.values():
            stream.stop()

    def active_mode(self) -> str:
        return self.active_key

    def latest_depth_snapshot(self):
        depth_stream = self.streams.get("depth")
        if depth_stream is None:
            return None
        return depth_stream.latest_depth_snapshot()


FRONT_CAMERA = FrontCameraMultiplexer()


def activate_front_color() -> str:
    return FRONT_CAMERA.set_active("color")



class DepthBBoxPayload(BaseModel):
  bbox: List[float]


# 初始化多路"摄像头"（可按需增减/改索引）
CAMERAS = {
    "front": FRONT_CAMERA,
    # "front": FrameSource(cam_index=3, name="front", fallback_indexes=[]),
    # "left": FrameSource(cam_index=14, name="left", fallback_indexes=[]),
    "right": FrameSource(cam_index=4, name="right", fallback_indexes=[4, 5]),
}

# ============ Telemetry / 状态（演示用随机/模拟）===========
STATUS_STEPS = [
    "listening", "searching", "waiting_api", "navigating", "grasping"
]
_status_idx = 0

def mock_telemetry():
    """获取实时遥测数据，包括从 AGV 的位置信息"""
    # 如果 Navigator 可用，使用实时 AGV 位置数据
    yaw = (time.time() * 20) % 360
    
    pitch = 2.0 * np.sin(time.time() / 3)
    roll = 1.5 * np.cos(time.time() / 2)
    v_lin = 0.2 * np.sin(time.time() / 1.7)
    v_ang = 0.4 * np.cos(time.time() / 2.3)
    action = random.choice(["idle", "forward", "turn_left", "turn_right", "stop"])
    return {
        "orientation": {"yaw": yaw, "pitch": pitch, "roll": roll},
        "velocity": {"linear": v_lin, "angular": v_ang},
        "chassis_action": action
    }

def mock_status():
    global _status_idx
    # 每3秒推进一步
    _status_idx = int(time.time() / 3) % len(STATUS_STEPS)
    return {
        "steps": STATUS_STEPS,
        "current": STATUS_STEPS[_status_idx]
    }

# ============ 静态资源挂载 ============
APP.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ============ 路由：页面 ============
try:
    from html_parts import INDEX_HTML
except ImportError:
    INDEX_HTML = "<h1>Error: html_parts.py not found</h1>"
LATEST_CAPTURED_IMG = {"url": "", "camera": "front", "w": FRONT_WIDTH, "h": FRONT_HEIGHT, "ts": 0}

@APP.get("/", response_class=HTMLResponse)
def index():
    return INDEX_HTML

@APP.get("/test", response_class=HTMLResponse)
def test_page():
    """AGV Position Direct Test Page"""
    return """<!DOCTYPE html>
<html>
<head>
    <title>AGV Position Direct Test</title>
    <style>
        body { font-family: monospace; margin: 20px; background: #f0f0f0; }
        div { background: white; padding: 15px; margin: 10px 0; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        button { padding: 10px 15px; margin: 5px; background: #007bff; color: white; border: none; border-radius: 3px; cursor: pointer; }
        button:hover { background: #0056b3; }
        pre { background: #f8f9fa; padding: 10px; overflow-x: auto; border-radius: 3px; }
        #fetch-result { white-space: pre-wrap; word-wrap: break-word; }
    </style>
</head>
<body>
    <h1>🔬 AGV Position Direct Update Test</h1>
    
    <div>
        <h2>Test 1: Direct Manual Update</h2>
        <p>AGV Position: θ=<span id="theta">-</span>rad, x=<span id="agv-x">-</span>m, y=<span id="agv-y">-</span>m</p>
        <button onclick="testManualUpdate()">Test Manual Update</button>
        <p id="test1-result"></p>
    </div>

    <div>
        <h2>Test 2: Fetch from API (Single)</h2>
        <p id="fetch-result">Click button to fetch from API</p>
        <button onclick="testFetchAPI()">Test Fetch from API</button>
    </div>

    <div>
        <h2>Test 3: Auto Update Every 1s</h2>
        <p id="auto-result">Waiting...</p>
        <p><small>Check if this updates automatically every second</small></p>
    </div>

    <script>
        console.log('Test page loaded');
        
        function testManualUpdate() {
            console.log('[Test1] Starting manual update');
            try {
                const theta = document.getElementById('theta');
                const x = document.getElementById('agv-x');
                const y = document.getElementById('agv-y');
                
                console.log('[Test1] Elements found:', {theta, x, y});
                
                theta.textContent = '3.14';
                x.textContent = '100.5';
                y.textContent = '200.3';
                
                console.log('[Test1] Manual update completed');
                document.getElementById('test1-result').innerHTML = '✅ Manual update successful - check if values changed above';
            } catch(e) {
                console.error('[Test1] Error:', e);
                document.getElementById('test1-result').innerHTML = '❌ Error: ' + e.message;
            }
        }

        async function testFetchAPI() {
            console.log('[Test2] Starting API fetch');
            try {
                const resp = await fetch('/api/agv/pose');
                console.log('[Test2] Response received, status:', resp.status);
                
                const data = await resp.json();
                console.log('[Test2] JSON parsed:', data);
                
                document.getElementById('fetch-result').innerHTML = '<pre>Status: ' + resp.status + '\\n' + JSON.stringify(data, null, 2) + '</pre>';
                
                if (data.status === 'ok' && data.pose) {
                    console.log('[Test2] Updating DOM with pose data');
                    document.getElementById('theta').textContent = data.pose.theta.toFixed(2);
                    document.getElementById('agv-x').textContent = data.pose.x.toFixed(2);
                    document.getElementById('agv-y').textContent = data.pose.y.toFixed(2);
                    console.log('[Test2] DOM updated successfully');
                } else {
                    console.warn('[Test2] Unexpected response format');
                }
            } catch(e) {
                console.error('[Test2] Error:', e);
                document.getElementById('fetch-result').textContent = '❌ Error: ' + e.message;
            }
        }

        // Auto update every 1s
        console.log('[Test3] Starting auto-update loop');
        setInterval(async () => {
            try {
                const resp = await fetch('/api/agv/pose');
                const data = await resp.json();
                if (data.status === 'ok' && data.pose) {
                    const now = new Date().toLocaleTimeString();
                    const msg = '[' + now + '] θ=' + data.pose.theta.toFixed(2) + 
                                ', x=' + data.pose.x.toFixed(2) + 
                                ', y=' + data.pose.y.toFixed(2);
                    document.getElementById('auto-result').textContent = msg;
                    console.log('[Test3] Updated: ' + msg);
                }
            } catch(e) {
                console.error('[Test3] Auto update error:', e);
            }
        }, 1000);
    </script>
</body>
</html>"""

# ============ 路由：相机帧 ============
@APP.get("/api/cam/{name}")
def api_cam(name: str):
    src = CAMERAS.get(name)
    if not src:
        return Response(status_code=404)
    if name == "front":
        target_w, target_h = FRONT_WIDTH, FRONT_HEIGHT
    else:
        target_w, target_h = 960, 540
    jpeg = src.get_jpeg_bytes(w=target_w, h=target_h)
    return Response(content=jpeg, media_type="image/jpeg")

# 截帧保存（展示“传给VLM的图片”）

@APP.get("/api/capture")
def api_capture(cam: str = Query("front")):
    src = CAMERAS.get(cam)
    if not src:
        return JSONResponse({"error": "no such camera"}, status_code=400)
    url = src.capture_and_save()
    abs_path = (BASE / url.lstrip("/")).resolve()
    width, height = 0, 0
    try:
        with Image.open(abs_path) as img:
            width, height = img.size
    except Exception:
        pass
    print(f"Captured image from {cam}: {url} ({width}x{height})")
    global LATEST_CAPTURED_IMG
    LATEST_CAPTURED_IMG = {"url": url, "camera": cam, "w": width, "h": height, "ts": int(time.time() * 1000)}
    return {"url": str(abs_path), "camera": cam, "w": width, "h": height}
# ============ VLM结果存储 ============
LATEST_VLM_RESULT = {
  "boxes": [],
  "image_size": [FRONT_WIDTH, FRONT_HEIGHT],
  "ts": 0,
  "annotated_url": "",
  "mask_url": "",
  "mask_score": None,
}

# ============ 任务日志存储 ============
TASK_LOGS = []
MAX_LOGS = 10000

# ============ 建议/提示存储 ============
SUGGESTIONS = []
MAX_SUGGESTIONS = 200

def add_suggestion(message: str, level: str = "info"):
  global SUGGESTIONS
  import datetime
  entry = {
    "ts": int(time.time() * 1000),
    "time": datetime.datetime.now().strftime("%H:%M:%S"),
    "level": level,
    "message": message,
  }
  SUGGESTIONS.append(entry)
  if len(SUGGESTIONS) > MAX_SUGGESTIONS:
    SUGGESTIONS = SUGGESTIONS[-MAX_SUGGESTIONS:]
  print(f"[SUGGESTION][{level.upper()}] {message}")
  return entry

# ============ 世界模型 & 行为树状态 ============
WORLD_MODEL_STATE = {
  "snapshot": None,
  "ts": 0,
}
PLAN_STATE = {
  "root": None,
  "steps": [],
  "metadata": {},
  "current_index": -1,
  "current_node": None,
  "timeline": [],
  "ts": 0,
}

CONTROL_STATE = {
  "last_action": None,
  "ts": 0,
}

def add_task_log(message: str, level: str = "info"):
  """添加任务日志"""
  global TASK_LOGS
  import datetime
  log_entry = {
    "ts": int(time.time() * 1000),
    "time": datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3],
    "level": level,  # info, success, warning, error
    "message": message,
  }
  TASK_LOGS.append(log_entry)
  # 保持最多100条日志
  if len(TASK_LOGS) > MAX_LOGS:
    TASK_LOGS = TASK_LOGS[-MAX_LOGS:]
  print(f"[{level.upper()}] {message}")
  return log_entry

@APP.get("/api/capture/latest")
def api_capture_latest():
  return LATEST_CAPTURED_IMG

# VLM结果提交接口（主控流程调用）
@APP.post("/api/front_cam/change")
async def api_front_cam_change(request: Request, cam_type: str):
    data = await request.json()
    cam_type = data.get("cam_type", "color")
    if cam_type == "color":
        activate_front_color()
    elif cam_type == "depth":
        activate_front_depth()
    else:
        return JSONResponse({"error": "invalid cam_type"}, status_code=400)
    return {"ok": True, "cam_type": cam_type}

@APP.post("/api/vlm/result")
async def api_vlm_result(request: Request):
  data = await request.json()
  # 期望格式：{"boxes": [[x1,y1,x2,y2], ...], "image_size": [w,h], ...}
  data["ts"] = int(time.time() * 1000)
  global LATEST_VLM_RESULT
  LATEST_VLM_RESULT = data
  bbox = data.get("mapped_bbox")
  print(f"Received VLM result with bbox: {bbox}")
  if bbox:
      src = CAMERAS.get("front")
      if src:
          stream = src.streams.get("color")
          if stream and hasattr(stream, "set_surface_points_hint"):
              stream.set_surface_points_hint(data.get("surface_points"))
              # 确保只临时显示
              threading.Timer(1.0, lambda: stream.set_surface_points_hint(None)).start()
          # 后端负责后续深度计算
  return {
      "ok": True,
      "surface_points": data.get("surface_points"),
      "range_estimate": data.get("range_estimate"),
  }

@APP.post("/api/vlm/mask")
async def api_vlm_mask(request: Request):
  data = await request.json()
  mask_url = data.get("mask_url") or data.get("url")
  if not mask_url:
    return JSONResponse({"ok": False, "error": "missing mask_url"}, status_code=400)
  global LATEST_VLM_RESULT
  image_path = data.get("image_path")
  last_image = LATEST_VLM_RESULT.get("original_image_path")
  if image_path and last_image and image_path != last_image:
    print(f"[VLM] Received mask for {image_path}, but latest image is {last_image}")
  LATEST_VLM_RESULT["mask_url"] = mask_url
  if "path" in data:
    LATEST_VLM_RESULT["surface_mask_path"] = data.get("path")
  LATEST_VLM_RESULT["mask_score"] = data.get("mask_score") or data.get("score")
  LATEST_VLM_RESULT["mask_job_id"] = data.get("job_id")
  LATEST_VLM_RESULT["mask_ts"] = data.get("timestamp") or int(time.time() * 1000)
  LATEST_VLM_RESULT["ts"] = int(time.time() * 1000)
  return {"ok": True}

# VLM结果获取接口（UI端轮询）
@APP.get("/api/vlm/latest")
def api_vlm_latest():
  return LATEST_VLM_RESULT

# VLM mock：返回一些随机框（像素xyxy）
@APP.get("/api/vlm/mock")
def api_vlm_mock():
    # 使用最后一张图的尺寸估算框
    files = sorted(CAP_DIR.glob("*.jpg"))
    if not files:
        # 默认尺寸
        W, H = 960, 540
    else:
        img = Image.open(files[-1])
        W, H = img.size

    boxes: List[Tuple[int,int,int,int]] = []
    for _ in range(random.randint(1, 4)):
        x1 = random.randint(0, max(0, W-200))
        y1 = random.randint(0, max(0, H-150))
        x2 = min(W-1, x1 + random.randint(60, 220))
        y2 = min(H-1, y1 + random.randint(40, 180))
        boxes.append([x1, y1, x2, y2])

    return {"boxes": boxes, "mode": "xyxy", "image_size": [W, H]}

# Telemetry / Status
@APP.get("/api/telemetry")
def api_telemetry():
    return mock_telemetry()

@APP.get("/api/status")
def api_status():
    return mock_status()


@APP.get("/api/depth/frame")
def api_depth_frame():
  """Return latest depth frame bundle for backend processing."""
  front_cam = CAMERAS.get("front")
  if front_cam is None:
    return JSONResponse({"error": "front camera unavailable"}, status_code=503)
  stream = front_cam.streams.get("color")
  if stream is None:
    return JSONResponse({"error": "color stream unavailable"}, status_code=503)
  color_frame = getattr(stream, "color_frame", None)
  depth_frame = getattr(stream, "depth_frame", None)
  if depth_frame is None or color_frame is None:
    return JSONResponse({"error": "depth frame unavailable"}, status_code=503)

  try:
    color_frame = color_frame.as_video_frame()
    depth_frame = depth_frame.as_video_frame()

    depth_width = depth_frame.get_width()
    depth_height = depth_frame.get_height()

    color_profile = color_frame.get_stream_profile()
    depth_profile = depth_frame.get_stream_profile()
    color_video_profile = color_profile.as_video_stream_profile()
    depth_video_profile = depth_profile.as_video_stream_profile()
    color_intrinsics = _serialize_intrinsics(color_video_profile)
    depth_intrinsics = _serialize_intrinsics(depth_video_profile)
    extrinsic = _serialize_extrinsic(depth_profile.get_extrinsic_to(color_profile))
    depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape(depth_height, depth_width)
  except Exception as exc:
    return JSONResponse({"error": f"prepare depth bundle failed: {exc}"}, status_code=500)

  depth_bytes = depth_data.astype(np.uint16, copy=False).tobytes()
  depth_b64 = base64.b64encode(depth_bytes).decode("ascii")

  return {
      "width": depth_width,
      "height": depth_height,
      "timestamp": int(time.time() * 1000),
      "depth_intrinsics": depth_intrinsics,
      "color_intrinsics": color_intrinsics,
      "extrinsic": extrinsic,
      "depth_b64": depth_b64,
      "dtype": str(depth_data.dtype),
  }


@APP.post("/api/depth/bbox_center")
def api_depth_bbox_center(payload: DepthBBoxPayload):
  bbox = payload.bbox
  if len(bbox) != 4:
    return JSONResponse({"error": "bbox must contain four values"}, status_code=400)

  depth_stream = FRONT_CAMERA.streams.get("depth")
  if depth_stream is None:
    return JSONResponse({"error": "depth stream unavailable"}, status_code=503)

  started_here = False
  if not depth_stream.running:
    if not depth_stream.start():
      return JSONResponse({"error": "failed to start depth stream"}, status_code=503)
    started_here = True
    time.sleep(0.2)

  snapshot = depth_stream.latest_depth_snapshot()
  deadline = time.time() + 0.6
  while snapshot is None and time.time() < deadline:
    time.sleep(0.05)
    snapshot = depth_stream.latest_depth_snapshot()

  if snapshot is None:
    if started_here and FRONT_CAMERA.active_mode() != "depth":
      depth_stream.stop()
    return JSONResponse({"error": "no depth frame available"}, status_code=504)

  depth_map = snapshot["depth"]
  scale = float(snapshot.get("scale", 1.0) or 1.0)
  intr = snapshot.get("intrinsics", {})
  width = int(intr.get("width", depth_map.shape[1]))
  height = int(intr.get("height", depth_map.shape[0]))

  x1, y1, x2, y2 = bbox
  x_min = max(0, min(width - 1, int(min(x1, x2))))
  x_max = max(0, min(width, int(max(x1, x2))))
  y_min = max(0, min(height - 1, int(min(y1, y2))))
  y_max = max(0, min(height, int(max(y1, y2))))

  if x_max <= x_min or y_max <= y_min:
    if started_here and FRONT_CAMERA.active_mode() != "depth":
      depth_stream.stop()
    return JSONResponse({"error": "bbox collapsed after clamping"}, status_code=400)

  roi = depth_map[y_min:y_max, x_min:x_max]
  if roi.size == 0:
    if started_here and FRONT_CAMERA.active_mode() != "depth":
      depth_stream.stop()
    return JSONResponse({"error": "empty roi"}, status_code=400)

  valid_mask = roi > 0
  if not valid_mask.any():
    if started_here and FRONT_CAMERA.active_mode() != "depth":
      depth_stream.stop()
    return JSONResponse({"error": "no valid depth pixels"}, status_code=404)

  x_coords = np.arange(x_min, x_max, dtype=np.float32)
  y_coords = np.arange(y_min, y_max, dtype=np.float32)
  grid_x, grid_y = np.meshgrid(x_coords, y_coords)

  depth_values = roi[valid_mask].astype(np.float32)
  grid_x = grid_x[valid_mask]
  grid_y = grid_y[valid_mask]

  fx = float(intr.get("fx", 0.0))
  fy = float(intr.get("fy", 0.0))
  cx = float(intr.get("cx", width / 2.0))
  cy = float(intr.get("cy", height / 2.0))

  if fx == 0.0 or fy == 0.0:
    if started_here and FRONT_CAMERA.active_mode() != "depth":
      depth_stream.stop()
    return JSONResponse({"error": "invalid camera intrinsics"}, status_code=503)

  if np.issubdtype(roi.dtype, np.integer):
    depth_m = depth_values * scale
    depth_scale_used = scale
  else:
    depth_m = depth_values
    depth_scale_used = 1.0
  depth_unit = "meters"
  x_m = (grid_x - cx) * depth_m / fx
  y_m = (grid_y - cy) * depth_m / fy
  points = np.stack((x_m, y_m, depth_m), axis=-1)
  center = points.mean(axis=0)

  if started_here and FRONT_CAMERA.active_mode() != "depth":
    depth_stream.stop()

  return {
    "bbox": [x_min, y_min, x_max, y_max],
    "points_used": int(points.shape[0]),
    "center": {
      "x": float(center[0]),
      "y": float(center[1]),
      "z": float(center[2]),
    },
    "pixel_center": {
      "x": float((x_min + x_max) / 2.0),
      "y": float((y_min + y_max) / 2.0),
    },
    "depth_scale_reported": scale,
    "depth_scale_used": depth_scale_used,
    "depth_unit": depth_unit,
    "timestamp": snapshot.get("timestamp", 0.0),
  }

@APP.get("/api/agv/pose")
def api_agv_pose():
    """获取 AGV 实时位置信息"""
    return {
        "status": "ok",
        "pose": {
            "theta": AGV_POSE["theta"],
            "x": AGV_POSE["x"],
            "y": AGV_POSE["y"],
        },
        "timestamp": AGV_POSE["timestamp"]
    }

@APP.post("/api/agv/pose/update")
async def api_agv_pose_update(request: Request):
    """后端更新 AGV 实时位置信息"""
    global AGV_POSE
    data = await request.json()
    
    # 更新位置数据
    AGV_POSE["theta"] = float(data.get("theta", AGV_POSE["theta"]))
    AGV_POSE["x"] = float(data.get("x", AGV_POSE["x"]))
    AGV_POSE["y"] = float(data.get("y", AGV_POSE["y"]))
    AGV_POSE["timestamp"] = int(time.time() * 1000)
    # print(f"[AGV Pose Updated] θ={AGV_POSE['theta']}, x={AGV_POSE['x']}, y={AGV_POSE['y']}")
    return {
        "status": "ok",
        "message": "AGV pose updated",
        "pose": {
            "theta": AGV_POSE["theta"],
            "x": AGV_POSE["x"],
            "y": AGV_POSE["y"],
        }
    }

# 任务日志接口
@APP.post("/api/task/log")
async def api_task_log(request: Request):
  """后端推送任务日志"""
  data = await request.json()
  message = data.get("message", "")
  level = data.get("level", "info")
  log_entry = add_task_log(message, level)
  return {"ok": True, "log": log_entry}

@APP.get("/api/task/logs")
def api_task_logs():
  """获取所有任务日志"""
  return {"logs": TASK_LOGS}

@APP.delete("/api/task/logs")
def api_clear_logs():
  """清空任务日志"""
  global TASK_LOGS
  TASK_LOGS = []
  return {"ok": True}

@APP.post("/api/suggestions")
async def api_add_suggestion(request: Request):
  data = await request.json()
  message = data.get("message", "")
  level = data.get("level", "info")
  if not message:
    return JSONResponse({"error": "message required"}, status_code=400)
  entry = add_suggestion(message, level)
  return {"ok": True, "suggestion": entry}

@APP.get("/api/suggestions")
def api_list_suggestions():
  return {"suggestions": SUGGESTIONS}

@APP.delete("/api/suggestions")
def api_clear_suggestions():
  global SUGGESTIONS
  SUGGESTIONS = []
  return {"ok": True}

@APP.post("/api/world_model/update")
async def api_world_model_update(request: Request):
  """外部推送世界模型快照"""
  data = await request.json()
  WORLD_MODEL_STATE["snapshot"] = data
  WORLD_MODEL_STATE["ts"] = int(time.time() * 1000)
  return {"ok": True, "ts": WORLD_MODEL_STATE["ts"]}

@APP.get("/api/world_model")
def api_world_model_get():
  """获取最近一次世界模型快照"""
  return WORLD_MODEL_STATE

@APP.post("/api/plan/update")
async def api_plan_update(request: Request):
  """外部推送行为树/执行步骤"""
  data = await request.json()
  if "root" in data:
    PLAN_STATE["root"] = data["root"]
  if "steps" in data:
    PLAN_STATE["steps"] = data.get("steps") or []
  if "metadata" in data:
    PLAN_STATE["metadata"] = data.get("metadata") or {}
  if "current_index" in data:
    PLAN_STATE["current_index"] = int(data.get("current_index", -1))
  if "current_node" in data:
    PLAN_STATE["current_node"] = data.get("current_node")
  if "timeline" in data:
    PLAN_STATE["timeline"] = data.get("timeline") or []
  PLAN_STATE["ts"] = int(time.time() * 1000)
  return {"ok": True, "ts": PLAN_STATE["ts"]}

@APP.get("/api/plan")
def api_plan_get():
  """获取最近的行为树/执行步骤"""
  return PLAN_STATE

@APP.post("/api/task/control")
async def api_task_control(request: Request):
  data = await request.json()
  action = (data.get("action") or "").strip().lower()
  if not action:
    return JSONResponse({"error": "action required"}, status_code=400)
  CONTROL_STATE["last_action"] = action
  CONTROL_STATE["ts"] = int(time.time() * 1000)
  add_task_log(f"控制指令: {action}", "warning")
  return {"ok": True, "action": action, "ts": CONTROL_STATE["ts"]}

# ============ 启动 ============
def _cleanup():
    for c in CAMERAS.values():
        c.stop()


if __name__ == "__main__":
    import atexit
    atexit.register(_cleanup)
    

    import uvicorn
    print("Open http://127.0.0.1:8000")
    uvicorn.run(APP, host="0.0.0.0", port=8000)
