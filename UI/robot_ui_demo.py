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
INDEX_HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Robot Monitor Demo</title>
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body {
      font-family: 'Segoe UI', -apple-system, Roboto, 'Helvetica Neue', 'PingFang SC', 'Hiragino Sans GB', 'Microsoft YaHei', sans-serif;
      background: linear-gradient(135deg, #0f0f23 0%, #1a1a3e 50%, #0f0f23 100%);
      color: #e0e0e0;
      min-height: 100vh;
      overflow: hidden;
    }
    .app {
      display: grid;
      grid-template-columns: 2fr 1fr;
      gap: 20px;
      padding: 24px;
      height: 100vh;
      overflow: hidden;
    }
    .grid { display: grid; gap: 16px; overflow-y: auto; padding-right: 8px; }
    .grid::-webkit-scrollbar { width: 6px; }
    .grid::-webkit-scrollbar-track { background: rgba(255,255,255,0.05); border-radius: 3px; }
    .grid::-webkit-scrollbar-thumb { background: rgba(100,200,255,0.4); border-radius: 3px; }
    .grid::-webkit-scrollbar-thumb:hover { background: rgba(100,200,255,0.6); }
    
    .card {
      background: linear-gradient(135deg, rgba(20,30,60,0.8) 0%, rgba(15,25,50,0.8) 100%);
      border: 1px solid rgba(100,200,255,0.2);
      border-radius: 12px;
      padding: 16px;
      box-shadow: 0 8px 32px rgba(0,0,0,0.3), inset 0 1px 0 rgba(255,255,255,0.1);
      backdrop-filter: blur(10px);
      transition: all 0.3s cubic-bezier(0.23, 1, 0.320, 1);
    }
    .card:hover {
      border-color: rgba(100,200,255,0.4);
      box-shadow: 0 12px 48px rgba(100,200,255,0.15), inset 0 1px 0 rgba(255,255,255,0.2);
      transform: translateY(-2px);
    }
    .card h3 {
      font-size: 14px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 1px;
      margin-bottom: 12px;
      background: linear-gradient(135deg, #64c8ff, #00d4ff);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      background-clip: text;
    }
    
    .cams { grid-template-columns: repeat(3, 1fr); }
    .cam-wrap {
      position: relative;
      border-radius: 10px;
      overflow: hidden;
      border: 1px solid rgba(100,200,255,0.3);
      background: rgba(0,0,0,0.4);
      box-shadow: inset 0 0 20px rgba(100,200,255,0.1);
      transition: all 0.3s ease;
    }
    .cam-wrap:hover {
      border-color: rgba(100,200,255,0.6);
      box-shadow: inset 0 0 20px rgba(100,200,255,0.2), 0 0 20px rgba(100,200,255,0.1);
    }
    .cam-wrap img {
      width: 100%;
      height: 220px;
      object-fit: cover;
      display: block;
    }
    .cam-wrap.front-cam {
      grid-column: span 3;
      width: 1280px;
      height: 800px;
      justify-self: center;
    }
    .cam-wrap.front-cam img {
      width: 1280px;
      height: 800px;
      object-fit: contain;
      background: rgba(0,0,0,0.6);
    }
    .cam-toolbar {
      position: absolute;
      left: 8px;
      bottom: 8px;
      display: flex;
      gap: 6px;
    }
    button {
      padding: 6px 12px;
      border-radius: 6px;
      border: 1px solid rgba(100,200,255,0.4);
      background: linear-gradient(135deg, rgba(100,200,255,0.2), rgba(0,212,255,0.1));
      color: #64c8ff;
      cursor: pointer;
      font-size: 12px;
      font-weight: 500;
      transition: all 0.2s ease;
      text-shadow: 0 0 10px rgba(100,200,255,0.3);
    }
    button:hover {
      background: linear-gradient(135deg, rgba(100,200,255,0.4), rgba(0,212,255,0.2));
      border-color: rgba(100,200,255,0.8);
      box-shadow: 0 0 15px rgba(100,200,255,0.3), inset 0 0 10px rgba(255,255,255,0.1);
      transform: translateY(-2px);
    }
    button:active {
      transform: translateY(0);
    }
    
    .vlm-box { position: relative; }
    .vlm-img {
      max-width: 100%;
      border: 2px solid rgba(100,200,255,0.3);
      border-radius: 10px;
      display: block;
      background: rgba(0,0,0,0.6);
      box-shadow: 0 0 30px rgba(100,200,255,0.2), inset 0 0 20px rgba(0,0,0,0.3);
      transition: all 0.3s ease;
    }
    .vlm-img:hover {
      border-color: rgba(100,200,255,0.6);
      box-shadow: 0 0 40px rgba(100,200,255,0.3), inset 0 0 20px rgba(0,0,0,0.3);
    }
    .mask-card img {
      width: 100%;
      max-height: 420px;
      object-fit: contain;
      border-radius: 10px;
      background: #050505;
      border: 1px solid rgba(255,255,255,0.08);
      display: none;
    }
    .mask-placeholder {
      border: 1px dashed rgba(255,255,255,0.15);
      border-radius: 10px;
      padding: 30px;
      text-align: center;
      color: rgba(255,255,255,0.4);
      font-size: 12px;
    }
    .overlay { position: absolute; left:0; top:0; pointer-events:none; }
    
    .row { display: flex; gap: 12px; align-items: center; }
    .kpi { display: grid; grid-template-columns: repeat(3,1fr); gap: 8px; }
    .kpi .item {
      background: linear-gradient(135deg, rgba(100,200,255,0.1), rgba(0,212,255,0.05));
      border: 1px solid rgba(100,200,255,0.2);
      border-radius: 8px;
      padding: 10px;
      transition: all 0.3s ease;
    }
    .kpi .item:hover {
      border-color: rgba(100,200,255,0.4);
      background: linear-gradient(135deg, rgba(100,200,255,0.15), rgba(0,212,255,0.1));
      box-shadow: 0 0 15px rgba(100,200,255,0.2);
    }
    .kpi .item > div:first-child {
      font-size: 11px;
      color: #64c8ff;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      margin-bottom: 4px;
      font-weight: 600;
    }
    
    .progress { display: flex; gap: 8px; }
    .step {
      padding: 6px 12px;
      border-radius: 20px;
      border: 1px solid rgba(100,200,255,0.3);
      background: rgba(100,200,255,0.05);
      color: #999;
      font-size: 12px;
      text-transform: capitalize;
      transition: all 0.3s ease;
      cursor: default;
    }
    .step.active {
      background: linear-gradient(135deg, #64c8ff, #00d4ff);
      color: #000;
      border-color: #00d4ff;
      box-shadow: 0 0 20px rgba(100,200,255,0.4), inset 0 0 10px rgba(255,255,255,0.2);
      font-weight: 600;
    }
    
    .mono {
      font-family: 'Courier New', 'JetBrains Mono', monospace;
      font-size: 12px;
      color: #64c8ff;
      line-height: 1.6;
      text-shadow: 0 0 5px rgba(100,200,255,0.3);
    }
    
    .task-logs-container {
      height: 300px;
      background: rgba(0,0,0,0.6);
      border: 1px solid rgba(100,200,255,0.2);
      border-radius: 8px;
      padding: 12px;
      overflow-y: auto;
      font-family: 'Courier New', monospace;
      font-size: 11px;
      line-height: 1.6;
    }
    .task-logs-container::-webkit-scrollbar { width: 4px; }
    .task-logs-container::-webkit-scrollbar-track { background: rgba(255,255,255,0.05); }
    .task-logs-container::-webkit-scrollbar-thumb { background: rgba(100,200,255,0.3); border-radius: 2px; }
    .task-logs-container::-webkit-scrollbar-thumb:hover { background: rgba(100,200,255,0.5); }
    
    .log-entry {
      display: flex;
      gap: 8px;
      margin-bottom: 4px;
      padding: 4px 6px;
      border-radius: 4px;
      transition: background 0.2s ease;
      align-items: flex-start;
    }
    .log-entry:hover { background: rgba(100,200,255,0.1); }
    
    .log-time {
      color: #888;
      flex-shrink: 0;
      min-width: 70px;
    }
    .log-level {
      flex-shrink: 0;
      min-width: 50px;
      font-weight: 600;
      text-transform: uppercase;
      font-size: 10px;
    }
    .log-level.info { color: #64c8ff; text-shadow: 0 0 5px rgba(100,200,255,0.4); }
    .log-level.success { color: #00ff88; text-shadow: 0 0 5px rgba(0,255,136,0.4); }
    .log-level.warning { color: #ffaa00; text-shadow: 0 0 5px rgba(255,170,0,0.4); }
    .log-level.error { color: #ff4444; text-shadow: 0 0 5px rgba(255,68,68,0.4); }
    
    .log-message {
      flex: 1;
      color: #d0d0d0;
      word-break: break-word;
    }
    
    .wm-list {
      margin-top: 8px;
      max-height: 260px;
      overflow-y: auto;
      display: flex;
      flex-direction: column;
      gap: 8px;
    }
    .wm-list::-webkit-scrollbar { width: 4px; }
    .wm-item {
      border: 1px solid rgba(100,200,255,0.2);
      border-radius: 8px;
      padding: 8px 10px;
      background: rgba(7,15,35,0.6);
      display: flex;
      flex-direction: column;
      gap: 4px;
      transition: all 0.2s ease;
    }
    .wm-item.visible {
      border-color: rgba(0,210,120,0.6);
      box-shadow: 0 0 16px rgba(0,210,120,0.3);
    }
    .wm-id {
      font-size: 13px;
      font-weight: 600;
      color: #64c8ff;
    }
    .wm-meta {
      font-size: 11px;
      color: #9bd;
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
    }
    .wm-empty {
      font-size: 12px;
      color: #888;
      padding: 12px;
      text-align: center;
      border: 1px dashed rgba(100,200,255,0.3);
      border-radius: 8px;
    }
    
    .plan-steps {
      margin-top: 10px;
      display: flex;
      flex-direction: column;
      gap: 8px;
      max-height: 220px;
      overflow-y: auto;
    }
    .plan-steps::-webkit-scrollbar { width: 4px; }
    .plan-node {
      border: 1px solid rgba(100,200,255,0.2);
      border-radius: 8px;
      padding: 8px 10px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      background: rgba(7,15,35,0.6);
      transition: all 0.2s ease;
    }
    .plan-node.active {
      border-color: rgba(255,200,0,0.8);
      box-shadow: 0 0 20px rgba(255,200,0,0.4);
      background: rgba(255,200,0,0.15);
    }
    .plan-node .plan-label {
      font-size: 13px;
      color: #e0e0e0;
      font-weight: 600;
    }
    .plan-node .plan-type {
      font-size: 11px;
      color: #9bd;
      text-transform: uppercase;
    }
    .plan-placeholder {
      padding: 12px;
      font-size: 12px;
      color: #888;
      border: 1px dashed rgba(100,200,255,0.3);
      border-radius: 8px;
      text-align: center;
    }
    .plan-json {
      margin-top: 10px;
      max-height: 160px;
      overflow-y: auto;
      border: 1px solid rgba(100,200,255,0.2);
      border-radius: 8px;
      padding: 8px;
      background: rgba(0,0,0,0.4);
      font-size: 11px;
      color: #9bd;
    }
    .timeline {
      margin-top: 10px;
      max-height: 200px;
      overflow-y: auto;
      display: flex;
      flex-direction: column;
      gap: 6px;
    }
    .timeline::-webkit-scrollbar { width: 4px; }
    .timeline-entry {
      border: 1px solid rgba(100,200,255,0.2);
      border-radius: 8px;
      padding: 6px 8px;
      background: rgba(7,15,35,0.55);
      display: flex;
      flex-direction: column;
      gap: 4px;
    }
    .timeline-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-size: 11px;
      color: #9bd;
    }
    .timeline-stage {
      font-weight: 600;
      letter-spacing: 0.5px;
      text-transform: uppercase;
      color: #64c8ff;
    }
    .timeline-node {
      font-size: 11px;
      color: #ccc;
    }
    .timeline-status {
      font-weight: 600;
      text-transform: uppercase;
      font-size: 10px;
    }
    .timeline-status.success { color: #00ff88; }
    .timeline-status.failure { color: #ff6666; }
    .timeline-status.observe { color: #64c8ff; }
    .timeline-detail {
      font-size: 11px;
      color: #b0c4ff;
      white-space: pre-wrap;
      word-break: break-word;
    }
    .suggestion-list {
      margin-top: 8px;
      max-height: 220px;
      overflow-y: auto;
      display: flex;
      flex-direction: column;
      gap: 6px;
    }
    .suggestion-entry {
      border: 1px solid rgba(100,200,255,0.2);
      border-radius: 6px;
      padding: 6px 8px;
      font-size: 12px;
      color: #d0d9ff;
      background: rgba(10,18,40,0.7);
    }
    .suggestion-entry .tag {
      font-size: 10px;
      text-transform: uppercase;
      color: #64c8ff;
      margin-right: 6px;
    }
    .suggestion-entry.warning { border-color: rgba(255,170,0,0.6); color: #ffd899; }
    .suggestion-entry.error { border-color: rgba(255,80,80,0.7); color: #ffb0b0; }
    .control-bar {
      display: flex;
      gap: 8px;
      margin-top: 10px;
      flex-wrap: wrap;
    }
    .control-bar button {
      flex: 1;
      min-width: 90px;
    }
    .small-label {
      font-size: 11px;
      color: #9bd;
      margin-top: 4px;
    }
  </style>
</head>
<body>
  <div class="app">
    <!-- 左侧：相机 & 状态 -->
    <div class="grid">
      <div class="card">
        <h3>Camera Feeds</h3>
        <div class="grid cams">
          <div class="cam-wrap front-cam">
            <img id="cam-front" class="front-cam-img" width="1280" height="800" src="/api/cam/front?ts=0" alt="front" />
            <div class="cam-toolbar">
              <button onclick="capture('front')">Capture</button>
            </div>
          </div>
          <div class="cam-wrap">
            <img id="cam-left" src="/api/cam/left?ts=0" alt="left" />
            <div class="cam-toolbar">
              <button onclick="capture('left')">Capture</button>
            </div>
          </div>
          <div class="cam-wrap">
            <img id="cam-right" src="/api/cam/right?ts=0" alt="right" />
            <div class="cam-toolbar">
              <button onclick="capture('right')">Capture</button>
            </div>
          </div>
        </div>
      </div>

      <div class="card">
        <h3>Robot Telemetry</h3>
        <div class="kpi">
          <div class="item">
            <div>AGV Position (Real-time)</div>
            <div class="mono">θ=<span id="theta">-</span>rad, x=<span id="agv-x">-</span>m, y=<span id="agv-y">-</span>m</div>
          </div>
          <div class="item">
            <div>Velocity</div>
            <div class="mono">v=<span id="vlin">0</span> m/s, ω=<span id="vang">0</span> rad/s</div>
          </div>
          <div class="item">
            <div>Chassis Action</div>
            <div class="mono" id="action">idle</div>
          </div>
        </div>
      </div>

      <div class="card">
        <h3>Task Progress / Logs</h3>
        <div class="row" style="margin-bottom: 8px;">
          <button onclick="clearTaskLogs()" style="font-size: 11px; padding: 4px 8px;">Clear Logs</button>
          <span style="font-size: 11px; color: #888; flex: 1; text-align: right;" id="log-count">0 entries</span>
        </div>
        <div id="task-logs" class="task-logs-container"></div>
      </div>
    </div>

    <!-- 右侧：VLM Payload & 结果 -->
    <div class="grid">
      <div class="card vlm-box">
        <h3>VLM Payload / Result</h3>
        <div style="position:relative; margin-top:8px;">
          <img id="vlm-img" class="vlm-img" src="" alt="captured" onload="syncCanvasSize()" />
          <canvas id="overlay" class="overlay"></canvas>
        </div>
        <div class="mono" id="bbox-json" style="margin-top:8px; font-size:12px; color:#555;"></div>
      </div>

      <div class="card mask-card">
        <h3>SAM Mask Preview</h3>
        <div class="mask-preview">
          <img id="sam-mask-img" src="" alt="sam-mask" />
          <div class="mask-placeholder" id="sam-mask-placeholder">等待SAM分割结果...</div>
        </div>
        <div class="mono" id="sam-mask-meta" style="margin-top:8px; font-size:12px; color:#999;">暂无mask</div>
      </div>
      <div class="card">
        <h3>World Model State</h3>
        <div class="small-label" id="world-model-updated">等待数据...</div>
        <div id="world-model-entries" class="wm-list">
          <div class="wm-empty">尚未收到世界模型快照</div>
        </div>
      </div>

      <div class="card">
        <h3>Assistant Suggestions</h3>
        <div class="row" style="margin-bottom:8px; gap:8px; align-items:center;">
          <button onclick="clearSuggestions()" style="font-size:11px; padding:4px 8px;">Clear</button>
          <span style="font-size:11px; color:#888; flex:1; text-align:right;" id="suggestion-count">0 entries</span>
        </div>
        <div id="suggestion-list" class="suggestion-list">
          <div class="plan-placeholder">暂无建议</div>
        </div>
        <div class="small-label" style="margin-top:12px;">Quick Controls</div>
        <div class="control-bar">
          <button onclick="sendControl('pause')">Pause Plan</button>
          <button onclick="sendControl('resume')">Resume</button>
          <button onclick="sendControl('force_observe')">Force Observe</button>
        </div>
      </div>

      <div class="card">
        <h3>Behavior Tree Monitor</h3>
        <div class="small-label" id="plan-updated">等待计划...</div>
        <div id="plan-steps" class="plan-steps">
          <div class="plan-placeholder">当前没有激活的行为树</div>
        </div>
        <div class="small-label" style="margin-top: 10px;">Execution Timeline</div>
        <div id="execution-timeline" class="timeline">
          <div class="plan-placeholder">等待执行记录</div>
        </div>
        <pre id="plan-tree-json" class="plan-json mono"></pre>
      </div>
    </div>
  </div>

<script>
  let lastWorldTs = -1;
  let lastPlanTs = -1;
  let lastSuggestionSize = 0;
  // 简单轮询刷新三个相机图像
  setInterval(() => {
    ['front','left','right'].forEach(id => {
      const img = document.getElementById('cam-' + id);
      img.src = '/api/cam/' + id + '?ts=' + Date.now(); // 防缓存
    });
  }, 50);

  // Capture：把当前帧保存，并把“传给VLM的图片”显示到右侧
  async function capture(which) {
    await fetch('/api/capture?cam=' + which);
    clearBoxes();
    document.getElementById('bbox-json').textContent = '';
  }
  let lastCapturePath = '';
  async function pollLatestCapture() {
    try {
      const r = await fetch('/api/capture/latest');
      const j = await r.json();
      if (j.url && j.url !== lastCapturePath) {
        lastCapturePath = j.url;
        const imgEl = document.getElementById('vlm-img');
        // 新的capture，显示原图
        imgEl.src = j.url + '?ts=' + Date.now();
        clearBoxes();
        document.getElementById('bbox-json').textContent = '';
        console.log('[Capture] 显示原图:', j.url);
      }
    } catch(e) {}
  }
  setInterval(pollLatestCapture, 500);
  pollLatestCapture();

  // 画框：像素坐标 xyxy（左上到右下）
  function drawBoxes(boxes) {
    const img = document.getElementById('vlm-img');
    const canvas = document.getElementById('overlay');
    const ctx = canvas.getContext('2d');
    syncCanvasSize(); // 对齐尺寸
    ctx.clearRect(0,0,canvas.width, canvas.height);
    ctx.lineWidth = 3;
    ctx.strokeStyle = 'red';
    boxes.forEach(b => {
      const [x1,y1,x2,y2] = b;
      ctx.strokeRect(x1, y1, x2-x1, y2-y1);
    });
  }

  function clearBoxes() {
    const canvas = document.getElementById('overlay');
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0,0,canvas.width, canvas.height);
  }

  function syncCanvasSize() {
    const img = document.getElementById('vlm-img');
    const canvas = document.getElementById('overlay');
    const r = img.getBoundingClientRect();
    canvas.style.width = r.width + 'px';
    canvas.style.height = r.height + 'px';
    canvas.width = r.width;
    canvas.height = r.height;
  }

  // Telemetry & Status 轮询
  async function tick() {
    try {
      const t = await (await fetch('/api/telemetry')).json();
      // 仅在元素存在时才更新（防止 null 错误）
      const yaw = document.getElementById('yaw');
      if (yaw) yaw.textContent = t.orientation.yaw.toFixed(1);
      const pitch = document.getElementById('pitch');
      if (pitch) pitch.textContent = t.orientation.pitch.toFixed(1);
      const roll = document.getElementById('roll');
      if (roll) roll.textContent = t.orientation.roll.toFixed(1);
      const vlin = document.getElementById('vlin');
      if (vlin) vlin.textContent = t.velocity.linear.toFixed(2);
      const vang = document.getElementById('vang');
      if (vang) vang.textContent = t.velocity.angular.toFixed(2);
      const action = document.getElementById('action');
      if (action) action.textContent = t.chassis_action;

      const s = await (await fetch('/api/status')).json();
      const steps = s.steps || [];
      const cur = s.current;
      // 仅在 #steps 元素存在时才更新
      const box = document.getElementById('steps');
      if (box) {
        box.innerHTML = '';
        steps.forEach(step => {
          const div = document.createElement('div');
          div.className = 'step' + (step === cur ? ' active' : '');
          div.textContent = step;
          box.appendChild(div);
        });
      }

      // 控制信号示例

      // 获取 AGV 实时位置信息
      try {
        const agvPoseResp = await (await fetch('/api/agv/pose')).json();
        console.log('[AGV位置] 收到响应:', agvPoseResp);
        if (agvPoseResp.status === 'ok' && agvPoseResp.pose) {
          const pose = agvPoseResp.pose;
          console.log('[AGV位置] 更新DOM - theta=' + pose.theta.toFixed(2) + ', x=' + pose.x.toFixed(2) + ', y=' + pose.y.toFixed(2));
          document.getElementById('theta').textContent = pose.theta.toFixed(2);
          document.getElementById('agv-x').textContent = pose.x.toFixed(2);
          document.getElementById('agv-y').textContent = pose.y.toFixed(2);
        } else {
          console.warn('[AGV位置] 异常响应:', agvPoseResp);
        }
      } catch(e) {
        console.error('[AGV位置] 获取失败:', e);
      }

    } catch(e) {
      console.error('[tick轮询] 错误:', e);
    }
  }
  setInterval(tick, 100);
  tick();

  // 任务日志轮询
  let lastLogCount = 0;
  async function pollTaskLogs() {
    try {
      const r = await fetch('/api/task/logs');
      const j = await r.json();
      const logs = j.logs || [];
      renderTaskLogs(logs);
    } catch(e) {
      console.error('[日志轮询] 错误:', e);
    }
  }
  
  function renderTaskLogs(logs) {
    const container = document.getElementById('task-logs');
    document.getElementById('log-count').textContent = logs.length + ' entries';
    
    // 仅添加新日志，不清空旧日志
    if (logs.length > lastLogCount) {
      // 有新日志，添加新的条目
      for (let i = lastLogCount; i < logs.length; i++) {
        const log = logs[i];
        const entry = document.createElement('div');
        entry.className = 'log-entry';
        entry.innerHTML = `
          <span class="log-time">${log.time}</span>
          <span class="log-level ${log.level}">[${log.level}]</span>
          <span class="log-message">${log.message}</span>
        `;
        container.appendChild(entry);
      }
      
      // 仅当用户在底部时自动滚动到底部
      if (isScrolledToBottom(container)) {
        container.scrollTop = container.scrollHeight;
      }
      
      lastLogCount = logs.length;
    } else if (logs.length < lastLogCount) {
      // 日志被清空了
      container.innerHTML = '';
      lastLogCount = 0;
    }
  }
  
  // 检查容器是否滚动到底部
  function isScrolledToBottom(container) {
    return container.scrollHeight - container.scrollTop - container.clientHeight < 5; // 允许5px误差
  }
  
  async function clearTaskLogs() {
    if (confirm('确定要清空所有日志吗？')) {
      try {
        await fetch('/api/task/logs', { method: 'DELETE' });
        document.getElementById('task-logs').innerHTML = '';
        document.getElementById('log-count').textContent = '0 entries';
        lastLogCount = 0;  // 重置日志计数
      } catch(e) {
        console.error('[清空日志] 错误:', e);
      }
    }
  }
  
  setInterval(pollTaskLogs, 300);
  pollTaskLogs();

  // VLM最新结果轮询展示
  function updateMaskPreview(data) {
    const maskImg = document.getElementById('sam-mask-img');
    const placeholder = document.getElementById('sam-mask-placeholder');
    const meta = document.getElementById('sam-mask-meta');
    if (!maskImg || !placeholder || !meta) {
      return;
    }
    if (data && data.mask_url) {
      maskImg.src = data.mask_url + '?ts=' + Date.now();
      maskImg.style.display = 'block';
      placeholder.style.display = 'none';
      if (typeof data.mask_score === 'number' && !Number.isNaN(data.mask_score)) {
        meta.textContent = `score: ${data.mask_score.toFixed(3)}`;
      } else {
        meta.textContent = 'score: -';
      }
    } else {
      maskImg.removeAttribute('src');
      maskImg.style.display = 'none';
      placeholder.style.display = 'block';
      meta.textContent = '暂无mask';
    }
  }

  async function pollWorldModel() {
    try {
      const resp = await fetch('/api/world_model');
      if (!resp.ok) {
        return;
      }
      const data = await resp.json();
      if (!data) {
        return;
      }
      if (typeof data.ts === 'number' && data.ts === lastWorldTs) {
        return;
      }
      lastWorldTs = typeof data.ts === 'number' ? data.ts : Date.now();
      renderWorldModel(data);
    } catch (e) {
      console.error('[WorldModel] 获取失败:', e);
    }
  }

  function renderWorldModel(state) {
    const statusEl = document.getElementById('world-model-updated');
    const listEl = document.getElementById('world-model-entries');
    if (!statusEl || !listEl) {
      return;
    }
    const snapshot = state.snapshot || {};
    const tsLabel = typeof state.ts === 'number' ? new Date(state.ts).toLocaleTimeString() : '--';
    const goal = snapshot.goal || '无';
    statusEl.textContent = `goal: ${goal} • updated ${tsLabel}`;
    const objects = snapshot.objects || {};
    const entries = Object.entries(objects);
    if (!entries.length) {
      listEl.innerHTML = '<div class="wm-empty">暂无对象记录</div>';
      return;
    }
    const sorted = entries
      .map(([id, obj]) => ({ id, obj }))
      .sort((a, b) => {
        const da = (a.obj.attrs && typeof a.obj.attrs.range_estimate === 'number') ? a.obj.attrs.range_estimate : 999;
        const db = (b.obj.attrs && typeof b.obj.attrs.range_estimate === 'number') ? b.obj.attrs.range_estimate : 999;
        return da - db;
      });
    listEl.innerHTML = sorted.slice(0, 10).map(({ id, obj }) => {
      const attrs = obj.attrs || {};
      const dist = typeof attrs.range_estimate === 'number'
        ? attrs.range_estimate.toFixed(2) + ' m'
        : (attrs.range_estimate || '未知');
      const worldCenter = Array.isArray(obj.world_center)
        ? obj.world_center.slice(0, 2).map(v => (typeof v === 'number' ? v.toFixed(2) : v)).join(', ')
        : '-';
      const confidence = typeof obj.confidence === 'number'
        ? obj.confidence.toFixed(2)
        : (obj.confidence ?? '-');
      const seenIn = obj.seen_in || '-';
      return `
        <div class="wm-item ${obj.visible ? 'visible' : ''}">
          <div class="wm-id">${id}</div>
          <div class="wm-meta">
            <span>dist: ${dist}</span>
            <span>visible: ${obj.visible ? 'true' : 'false'}</span>
            <span>conf: ${confidence}</span>
          </div>
          <div class="wm-meta">
            <span>world: ${worldCenter}</span>
            <span>seen_in: ${seenIn}</span>
          </div>
        </div>
      `;
    }).join('');
  }

  async function pollPlanState() {
    try {
      const resp = await fetch('/api/plan');
      if (!resp.ok) {
        return;
      }
      const data = await resp.json();
      if (!data) {
        return;
      }
      if (typeof data.ts === 'number' && data.ts === lastPlanTs) {
        return;
      }
      lastPlanTs = typeof data.ts === 'number' ? data.ts : Date.now();
      renderPlanState(data);
    } catch (e) {
      console.error('[Plan] 获取失败:', e);
    }
  }

  async function pollSuggestions() {
    try {
      const resp = await fetch('/api/suggestions');
      if (!resp.ok) {
        return;
      }
      const data = await resp.json();
      renderSuggestions(data.suggestions || []);
    } catch (e) {
      console.error('[Suggestions] 获取失败:', e);
    }
  }

  function renderSuggestions(items) {
    const listEl = document.getElementById('suggestion-list');
    const countEl = document.getElementById('suggestion-count');
    if (!listEl || !countEl) {
      return;
    }
    countEl.textContent = items.length + ' entries';
    if (!items.length) {
      listEl.innerHTML = '<div class="plan-placeholder">暂无建议</div>';
      lastSuggestionSize = 0;
      return;
    }
    if (items.length === lastSuggestionSize) {
      return;
    }
    lastSuggestionSize = items.length;
    listEl.innerHTML = items.slice(-50).map(entry => {
      const cls = entry.level ? entry.level.toLowerCase() : 'info';
      const time = entry.time || '--:--:--';
      const msg = entry.message || '';
      return `
        <div class="suggestion-entry ${cls}">
          <div><span class="tag">${cls}</span>${msg}</div>
          <div style="font-size:10px; color:#888;">${time}</div>
        </div>
      `;
    }).join('');
    listEl.scrollTop = listEl.scrollHeight;
  }

  async function clearSuggestions() {
    try {
      await fetch('/api/suggestions', { method: 'DELETE' });
      renderSuggestions([]);
    } catch (e) {
      console.error('[Suggestions] 清空失败:', e);
    }
  }

  async function sendControl(action) {
    try {
      await fetch('/api/task/control', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action }),
      });
      console.log('[Control] sent', action);
    } catch (e) {
      console.error('[Control] 发送失败:', e);
    }
  }

  function renderPlanState(state) {
    const statusEl = document.getElementById('plan-updated');
    const stepsEl = document.getElementById('plan-steps');
    const jsonEl = document.getElementById('plan-tree-json');
    if (!statusEl || !stepsEl || !jsonEl) {
      return;
    }
    const tsLabel = typeof state.ts === 'number' ? new Date(state.ts).toLocaleTimeString() : '--';
    const metadata = state.metadata || {};
    const source = metadata.source || 'unknown';
    const pointer = state.current_node ? ` • current: ${state.current_node}` : '';
    statusEl.textContent = `source: ${source}${pointer} • ${tsLabel}`;
    const steps = state.steps || [];
    if (!steps.length) {
      stepsEl.innerHTML = '<div class="plan-placeholder">暂无行为树</div>';
      jsonEl.textContent = '// waiting for plan';
      return;
    }
    const currentIndex = typeof state.current_index === 'number' ? state.current_index : -1;
    stepsEl.innerHTML = steps.map((step, idx) => {
      const label = step.name || step.type || `node-${idx}`;
      const nodeType = step.type || (step.children ? 'sequence' : 'action');
      return `
        <div class="plan-node ${idx === currentIndex ? 'active' : ''}">
          <div class="plan-label">${idx + 1}. ${label}</div>
          <div class="plan-type">${nodeType}</div>
        </div>
      `;
    }).join('');
    renderTimeline(state.timeline || []);
    try {
      jsonEl.textContent = JSON.stringify(state.root || {}, null, 2);
    } catch (err) {
      jsonEl.textContent = '// 无法解析行为树';
    }
  }

  function renderTimeline(entries) {
    const container = document.getElementById('execution-timeline');
    if (!container) {
      return;
    }
    if (!entries.length) {
      container.innerHTML = '<div class="plan-placeholder">暂无执行记录</div>';
      return;
    }
    container.innerHTML = entries.map(entry => {
      const status = (entry.status || 'info').toLowerCase();
      const stage = (entry.stage || 'event').toUpperCase();
      const node = entry.node || '-';
      const timeLabel = entry.time || '--:--:--';
      const detail = entry.detail || '';
      const elapsed = typeof entry.elapsed === 'number'
        ? entry.elapsed.toFixed(2) + 's'
        : (entry.elapsed || '--');
      return `
        <div class="timeline-entry">
          <div class="timeline-header">
            <span class="timeline-stage">${stage} · ${node}</span>
            <span class="timeline-status ${status}">${status}</span>
          </div>
          <div class="timeline-header" style="font-size:10px;">
            <span>${timeLabel}</span>
            <span class="timeline-node">elapsed: ${elapsed}</span>
          </div>
          <div class="timeline-detail">${detail}</div>
        </div>
      `;
    }).join('');
  }

  setInterval(pollWorldModel, 700);
  pollWorldModel();
  setInterval(pollPlanState, 700);
  pollPlanState();
  setInterval(pollSuggestions, 1000);
  pollSuggestions();
</script>
</body>
</html>
"""
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
