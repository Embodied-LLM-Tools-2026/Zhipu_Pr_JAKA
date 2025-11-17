import base64
import io
import json
import math
import os
import random
import sys
import time
from datetime import datetime
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import requests
from PIL import Image
from pyorbbecsdk import (
    Config,
    Pipeline,
    OBSensorType,
    OBFormat,
    OBError,
    FormatConvertFilter,
    VideoFrame,
    OBConvertFormat,
    transformation2dto3d,
    OBPoint2f,
)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "tools")))

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

from tools.sam_client import generate_mask

DEFAULT_DEPTH_API = os.getenv("DEPTH_FRAME_API", "http://127.0.0.1:8000/api/depth/frame")

@dataclass
class DepthSnapshot:
    depth: np.ndarray
    intrinsics: Any
    extrinsic: Any
    dtype: str = "uint16"


def _is_sdk_intrinsics(value: Any) -> bool:
    try:
        cls = value.__class__
    except AttributeError:
        return False
    module = getattr(cls, "__module__", "") or ""
    name = getattr(cls, "__name__", "") or ""
    return module.startswith("pyorbbecsdk") and "Intrinsic" in name


def _is_sdk_extrinsic(value: Any) -> bool:
    try:
        cls = value.__class__
    except AttributeError:
        return False
    module = getattr(cls, "__module__", "") or ""
    name = getattr(cls, "__name__", "") or ""
    return module.startswith("pyorbbecsdk") and "Extrinsic" in name


def _normalize_intrinsics(value: Any, width: int, height: int) -> Dict[str, float]:
    result: Dict[str, float] = {}
    keys = ("fx", "fy", "cx", "cy", "width", "height")
    if isinstance(value, dict):
        for key in keys:
            if key in value and value[key] is not None:
                try:
                    result[key] = float(value[key])
                except (TypeError, ValueError):
                    pass
    else:
        for key in keys:
            if hasattr(value, key):
                try:
                    result[key] = float(getattr(value, key))
                except (TypeError, ValueError):
                    pass
    result.setdefault("width", float(width))
    result.setdefault("height", float(height))
    return result


def _compose_extrinsic(rotation: Any, translation: Any) -> Optional[np.ndarray]:
    if rotation is None or translation is None:
        return None
    try:
        rot = np.asarray(rotation, dtype=float).reshape(3, 3)
        trans = np.asarray(translation, dtype=float).reshape(3)
    except Exception:
        return None
    matrix = np.eye(4, dtype=float)
    matrix[:3, :3] = rot
    matrix[:3, 3] = trans
    return matrix


def _extract_extrinsic_matrix(extrinsic: Any) -> Optional[np.ndarray]:
    if extrinsic is None:
        return None
    if isinstance(extrinsic, dict):
        matrix = extrinsic.get("matrix")
        if matrix is not None:
            try:
                arr = np.asarray(matrix, dtype=float)
                if arr.shape == (4, 4):
                    return arr
            except Exception:
                pass
        rotation = extrinsic.get("rotation")
        translation = extrinsic.get("translation")
        return _compose_extrinsic(rotation, translation)
    rotation = getattr(extrinsic, "rotation", None)
    translation = getattr(extrinsic, "translation", None)
    return _compose_extrinsic(rotation, translation)

def decode_snapshot(payload: Dict[str, Any]) -> DepthSnapshot:
    depth_b64 = payload.get("depth_b64")
    if not depth_b64:
        raise ValueError("depth_b64 missing from payload")
    depth_bytes = base64.b64decode(depth_b64)
    dtype = payload.get("dtype", "uint16")
    width = int(payload.get("width", 0))
    height = int(payload.get("height", 0))
    if width <= 0 or height <= 0:
        raise ValueError("invalid depth dimensions")
    depth = np.frombuffer(depth_bytes, dtype=np.dtype(dtype)).reshape(height, width)
    intrinsics = payload.get("depth_intrinsics", {})
    extrinsic = payload.get("extrinsic", {})
    return DepthSnapshot(
        depth=depth,
        intrinsics=intrinsics,
        extrinsic=extrinsic,
        dtype=dtype,
    )


def fetch_snapshot(api_url: str = DEFAULT_DEPTH_API, *, timeout: float = 1.2) -> DepthSnapshot:
    response = requests.get(api_url, params={"align": "1"}, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    return decode_snapshot(data)


class TargetLocalizer:
    """Process depth frames to estimate target pose and supporting plane orientation."""

    def __init__(
        self,
        *,
        depth_api: str = DEFAULT_DEPTH_API,
    ) -> None:
        self.depth_api = depth_api

    # ------------------------------------------------------------------
    # Public entry points
    # ---------------------------------------------------------------
    def localize_from_service(
        self,
        bbox: Iterable[float],
        *,
        surface_points_hint: Optional[List[List[float]]] = None,
        range_estimate: Optional[float] = None,
        rgb_frame: Optional[np.ndarray] = None,
        timeout: float = 1.2,
        surface_mask: Optional[np.ndarray] = None,
        include_transform: bool = False,
    ) -> Optional[Dict[str, Any]]:
        snapshot = fetch_snapshot(self.depth_api, timeout=timeout)
        return self.localize_object(
            bbox=bbox,
            snapshot=snapshot,
            surface_points_hint=surface_points_hint,
            range_estimate=range_estimate,
            rgb_frame=rgb_frame,
            surface_mask=surface_mask,
            include_transform=include_transform,
        )

    def localize_object(
        self,
        *,
        bbox: Iterable[float],
        snapshot: DepthSnapshot,
        surface_points_hint: Optional[List[List[float]]] = None,
        range_estimate: Optional[float] = None,
        rgb_frame: Optional[np.ndarray] = None,
        surface_mask: Optional[np.ndarray] = None,
        include_transform: bool = False,
    ) -> Optional[Dict[str, Any]]:
        bbox_int = list(map(int, bbox))
        if len(bbox_int) != 4:
            raise ValueError("bbox must contain four numbers")

        transform = self._transform_points(
            snapshot=snapshot,
            bbox=bbox_int,
            rgb_frame=rgb_frame,
            surface_points_hint=surface_points_hint,
            surface_mask_override=surface_mask,
        )
        if transform is None:
            return None

        support_points = transform["support_points"]
        edge_info = self._fit_edge_orientation(support_points)
        tune_angle = float(edge_info.get("angle") or 0.0)

        payload: Dict[str, Any] = {
            "obj_center_3d": transform["center"].tolist(),
            "tune_angle": tune_angle,
            "edge_confidence": edge_info.get("confidence"),
            "edge_method": edge_info.get("method"),
            "edge_inliers": edge_info.get("inlier_count"),
            "edge_total_points": edge_info.get("total_points"),
            "edge_residual": edge_info.get("residual"),
            "surface_points": transform.get("surface_points"),
            "surface_mask_available": transform.get("surface_mask_available", False),
            "bbox": transform["bbox"],
        }

        if include_transform:
            payload["transform_result"] = transform

        return payload

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _transform_points(
        self,
        *,
        snapshot: DepthSnapshot,
        bbox: List[int],
        rgb_frame: Optional[np.ndarray],
        surface_points_hint: Optional[List[List[float]]],
        surface_mask_override: Optional[np.ndarray] = None,
    ) -> Optional[Dict[str, Any]]:
        depth = snapshot.depth
        height, width = depth.shape
        projector = self._build_projector(snapshot, (height, width))
        if projector is None:
            return None
        x1, y1, x2, y2 = bbox
        x_min, x_max = sorted((max(0, x1), min(width, x2)))
        y_min, y_max = sorted((max(0, y1), min(height, y2)))
        if x_min >= x_max or y_min >= y_max:
            return None

        def project_pixels(pixels: Iterable[Tuple[int, int]]) -> List[List[float]]:
            out: List[List[float]] = []
            for ix, iy in pixels:
                if 0 <= ix < width and 0 <= iy < height:
                    depth_val = float(depth[iy, ix])
                    if depth_val <= 0:
                        continue
                    point = projector(ix, iy, depth_val)
                    if point is not None:
                        out.append(point)
            return out

        bbox_pixels = [
            (ix, iy) for ix in range(x_min, x_max) for iy in range(y_min, y_max)
        ]
        object_points = project_pixels(bbox_pixels)
        if not object_points:
            return None

        cx = (x_min + x_max) // 2
        cy = (y_min + y_max) // 2
        object_points_np = np.asarray(object_points)
        coordinate_3d_array = np.mean(object_points_np, axis=0)
        obj_center_depth = float(coordinate_3d_array[2])

        surface_mask = (
            self._build_surface_mask(
                rgb_frame=rgb_frame,
                surface_points_hint=surface_points_hint,
                depth_width=width,
                depth_height=height,
            )
            if surface_mask_override is None
            else self._resize_mask_to_depth(surface_mask_override, width, height)
        )
        mask_available = surface_mask is not None
        camera_params = self._camera_params_from_snapshot(snapshot, width, height)

        support_points = self._collect_support_points(
            depth_data=depth,
            depth_width=width,
            depth_height=height,
            bbox=(x_min, x_max, y_min, y_max),
            center_px=(cx, cy),
            center_depth=obj_center_depth,
            project_pixels=project_pixels,
            surface_mask=surface_mask,
        )

        return {
            "center": coordinate_3d_array,
            "object_points": object_points_np,
            "support_points": support_points,
            "surface_points": surface_points_hint,
            "surface_mask_available": mask_available,
            "camera_params": camera_params,
            "intrinsics_raw": snapshot.intrinsics,
            "extrinsic_raw": snapshot.extrinsic,
            "bbox": (x_min, x_max, y_min, y_max),
            "depth_data": depth,
            "depth_width": width,
            "depth_height": height,
        }

    def _build_projector(
        self,
        snapshot: DepthSnapshot,
        frame_shape: Tuple[int, int],
    ):
        intr = snapshot.intrinsics
        extrinsic = snapshot.extrinsic
        sdk_projector = None
        if (
            transformation2dto3d is not None
            and OBPoint2f is not None
            and _is_sdk_intrinsics(intr)
            and (extrinsic is None or _is_sdk_extrinsic(extrinsic))
        ):
            sdk_projector = self._sdk_projector(intr, extrinsic)
        numeric_projector = self._make_numeric_projector(intr, extrinsic, frame_shape)
        if sdk_projector is None and numeric_projector is None:
            return None
        if sdk_projector is None:
            return numeric_projector
        if numeric_projector is None:
            return sdk_projector

        def projector(ix: int, iy: int, depth_value: float):
            point = sdk_projector(ix, iy, depth_value)
            if point is None:
                point = numeric_projector(ix, iy, depth_value)
            return point

        return projector

    def _sdk_projector(self, intr, extr):
        def projector(ix: int, iy: int, depth_value: float):
            if depth_value <= 0:
                return None
            try:
                point = transformation2dto3d(
                    OBPoint2f(float(ix), float(iy)),
                    float(depth_value),
                    intr,
                    extr,
                )
            except Exception:
                return None
            if point is None:
                return None
            return [float(point.x), float(point.y), float(point.z)]

        return projector

    def _make_numeric_projector(
        self,
        intrinsics: Any,
        extrinsic: Any,
        frame_shape: Tuple[int, int],
    ):
        height, width = frame_shape
        params = _normalize_intrinsics(intrinsics, width, height)
        fx = float(params.get("fx") or 0.0)
        fy = float(params.get("fy") or 0.0)
        if fx == 0.0 or fy == 0.0:
            return None
        cx = float(params.get("cx", params.get("width", width) / 2.0))
        cy = float(params.get("cy", params.get("height", height) / 2.0))
        extr_matrix = _extract_extrinsic_matrix(extrinsic)
        rot = None
        trans = None
        if extr_matrix is not None:
            try:
                mat = np.asarray(extr_matrix, dtype=float)
                if mat.shape == (4, 4):
                    rot = mat[:3, :3]
                    trans = mat[:3, 3]
            except Exception:
                rot = None
                trans = None

        def projector(ix: int, iy: int, depth_value: float):
            if depth_value <= 0:
                return None
            z = float(depth_value)
            x = (float(ix) - cx) * z / fx
            y = (float(iy) - cy) * z / fy
            if rot is not None and trans is not None:
                vec = rot @ np.array([x, y, z], dtype=float) + trans
                x, y, z = vec.tolist()
            return [float(x), float(y), float(z)]

        return projector

    def _build_surface_mask(
        self,
        *,
        rgb_frame: Optional[np.ndarray],
        surface_points_hint: Optional[List[List[float]]],
        depth_width: int,
        depth_height: int,
    ) -> Optional[np.ndarray]:
        if not surface_points_hint or rgb_frame is None:
            return None
        try:
            mask, _ = generate_mask(rgb_frame, surface_points_hint)
        except Exception as exc:
            print(f"[SAM] 生成mask失败: {exc}")
            return None
        if mask is None:
            return None
        return self._resize_mask_to_depth(mask, depth_width, depth_height)

    def _resize_mask_to_depth(
        self,
        mask: np.ndarray,
        depth_width: int,
        depth_height: int,
    ) -> np.ndarray:
        if mask.shape[0] == depth_height and mask.shape[1] == depth_width:
            return mask
        if cv2 is not None:
            return cv2.resize(mask, (depth_width, depth_height), interpolation=cv2.INTER_NEAREST)
        return np.array(
            Image.fromarray(mask).resize((depth_width, depth_height), resample=Image.NEAREST)
        )

    def _collect_support_points(
        self,
        *,
        depth_data: np.ndarray,
        depth_width: int,
        depth_height: int,
        bbox: Tuple[int, int, int, int],
        center_px: Tuple[int, int],
        center_depth: float,
        project_pixels: Any,
        surface_mask: Optional[np.ndarray],
    ) -> np.ndarray:
        sampled_pixels: List[Tuple[int, int]] = []
        if surface_mask is not None:
            indices = np.argwhere(surface_mask > 0)
            if indices.size:
                max_samples = 2000
                step = max(1, len(indices) // max_samples)
                for iy, ix in indices[::step]:
                    depth_val = float(depth_data[iy, ix])
                    if depth_val > 0:
                        sampled_pixels.append((ix, iy))

        max_samples = 1200
        if len(sampled_pixels) > max_samples:
            sampled_pixels = random.sample(sampled_pixels, max_samples)

        support_points = project_pixels(sampled_pixels)
        if not support_points:
            return np.empty((0, 3))
        return np.asarray(support_points)

    @staticmethod
    def _camera_params_from_snapshot(snapshot: DepthSnapshot, width: int, height: int) -> Dict[str, Any]:
        """
        Build the camera payload expected by the ZeroGrasp service.
        """
        params = _normalize_intrinsics(snapshot.intrinsics, width, height)
        width_i = int(width)
        height_i = int(height)
        intrinsics = {
            "fx": float(params.get("fx", 0.0)),
            "fy": float(params.get("fy", 0.0)),
            "cx": float(params.get("cx", width / 2.0)),
            "cy": float(params.get("cy", height / 2.0)),
        }
        return {
            "width": width_i,
            "height": height_i,
            "camera_width": width_i,
            "camera_height": height_i,
            "rectified_width": width_i,
            "rectified_height": height_i,
            "color_intrinsics": intrinsics.copy(),
            "depth_intrinsics": intrinsics.copy(),
        }

    @staticmethod
    def _structured_point_cloud(depth_data: np.ndarray, intrinsics: Any, extrinsic: Any) -> Optional[np.ndarray]:
        height, width = depth_data.shape
        params = _normalize_intrinsics(intrinsics, width, height)
        fx = float(params.get("fx") or 0.0)
        fy = float(params.get("fy") or 0.0)
        if fx == 0.0 or fy == 0.0:
            return None
        cx = float(params.get("cx", width / 2.0))
        cy = float(params.get("cy", height / 2.0))
        u, v = np.meshgrid(np.arange(width), np.arange(height))
        z = depth_data.astype(np.float32)
        x = (u.astype(np.float32) - cx) * z / fx
        y = (v.astype(np.float32) - cy) * z / fy
        points = np.stack([x, y, z], axis=-1)
        extr_matrix = _extract_extrinsic_matrix(extrinsic)
        if extr_matrix is not None:
            rot = extr_matrix[:3, :3]
            trans = extr_matrix[:3, 3]
            pts = points.reshape(-1, 3)
            pts = (rot @ pts.T).T + trans
            points = pts.reshape(height, width, 3)
        return points

    @staticmethod
    def _encode_png(image: np.ndarray) -> str:
        pil = Image.fromarray(image.astype(np.uint8, copy=False))
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")

    @staticmethod
    def _encode_npy(array: np.ndarray) -> str:
        buffer = io.BytesIO()
        np.save(buffer, array)
        return base64.b64encode(buffer.getvalue()).decode("ascii")

    def _fit_edge_orientation(self, points: np.ndarray) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "slope": None,
            "angle": None,
            "confidence": 0.0,
            "method": None,
            "inlier_count": 0,
            "total_points": int(points.shape[0]) if points is not None else 0,
            "residual": None,
        }
        if points is None or len(points) < 8:
            return result

        x = points[:, 0]
        z = points[:, 2]
        pts = np.column_stack([x, z])
        total = len(pts)

        best_inliers = 0
        best_model = None
        rng = random.Random(42)
        residual_threshold = 0.02
        max_trials = 120
        min_inliers = max(10, total // 6)

        for _ in range(max_trials):
            if total < 2:
                break
            i1, i2 = rng.sample(range(total), 2)
            (x1, z1), (x2, z2) = pts[i1], pts[i2]
            denom = (x2 - x1)
            if abs(denom) < 1e-6:
                slope = math.copysign(1e6, z2 - z1 or 1.0)
                intercept = z1 - slope * x1
            else:
                slope = (z2 - z1) / denom
                intercept = z1 - slope * x1

            residuals = np.abs(z - (slope * x + intercept))
            inlier_mask = residuals < residual_threshold
            inlier_count = int(inlier_mask.sum())
            if inlier_count > best_inliers:
                best_inliers = inlier_count
                best_model = (slope, intercept, inlier_mask)
                if best_inliers > total * 0.85:
                    break

        if best_model and best_inliers >= min_inliers:
            slope_init, _, inlier_mask = best_model
            inlier_pts = pts[inlier_mask]
            slope, intercept = np.polyfit(inlier_pts[:, 0], inlier_pts[:, 1], 1)
            residuals = np.abs(inlier_pts[:, 1] - (slope * inlier_pts[:, 0] + intercept))
            rms = float(np.sqrt(np.mean(residuals**2))) if len(residuals) else None
            angle = math.atan(slope)
            result.update(
                {
                    "slope": float(slope),
                    "angle": float(angle),
                    "confidence": best_inliers / total,
                    "method": "ransac",
                    "inlier_count": best_inliers,
                    "residual": rms,
                }
            )
            return result

        centered = pts - pts.mean(axis=0, keepdims=True)
        cov = centered.T @ centered / max(total - 1, 1)
        eig_vals, eig_vecs = np.linalg.eigh(cov)
        principal = eig_vecs[:, np.argmax(eig_vals)]
        dx, dz = principal
        slope = math.copysign(1e6, dz) if abs(dx) < 1e-6 else dz / dx
        angle = math.atan(slope)
        result.update(
            {
                "slope": float(slope),
                "angle": float(angle),
                "confidence": 0.2,
                "method": "pca",
                "inlier_count": total,
            }
        )
        return result

    def run_zero_grasp_inference(
        self,
        *,
        transform_result: Dict[str, Any],
        bbox: Tuple[int, int, int, int],
        rgb_frame: Optional[np.ndarray],
        timings: Optional[Dict[str, float]] = None,
        events: Optional[List[Tuple[str, str]]] = None,
    ) -> Optional[Dict[str, Any]]:
        zerograsp_url = os.getenv("ZEROGRASP_URL", "http://10.46.118.233:8000/predict")
        if not zerograsp_url:
            return None

        total_start = time.perf_counter()

        def _event(label: str) -> None:
            if events is not None:
                events.append((datetime.now().strftime("%H:%M:%S.%f")[:-3], label))

        try:
            _event("zerograsp.start")
            debug_flag = os.getenv("ZEROGRASP_DEBUG")
            debug_enabled = debug_flag not in (None, "", "0", "false", "False")

            def _debug(message: str) -> None:
                if debug_enabled:
                    print(f"[ZeroGrasp][debug] {message}")

            depth_data = transform_result.get("depth_data")
            if depth_data is None or depth_data.size == 0:
                _debug("missing depth data")
                return None
            depth_height, depth_width = depth_data.shape

            if rgb_frame is None:
                _debug("missing rgb frame")
                return None
            if rgb_frame.shape[0] != depth_height or rgb_frame.shape[1] != depth_width:
                if cv2 is None:
                    _debug("cv2 unavailable for rgb resize")
                    return None
                rgb_frame = cv2.resize(rgb_frame, (depth_width, depth_height), interpolation=cv2.INTER_AREA)

            camera_params = transform_result.get("camera_params")
            intr_raw = transform_result.get("intrinsics_raw")
            extr_raw = transform_result.get("extrinsic_raw")
            if not camera_params or not intr_raw:
                _debug("camera parameters or intrinsics missing")
                return None

            _event("zerograsp.point_cloud.start")
            pc_start = time.perf_counter()
            points3d = self._structured_point_cloud(depth_data, intr_raw, extr_raw)
            if timings is not None:
                timings["point_cloud"] = time.perf_counter() - pc_start
            if points3d is None:
                _debug("unable to build structured point cloud")
                return None
            _event("zerograsp.point_cloud.end")

            _event("zerograsp.encode_rgb.start")
            rgb_encode_start = time.perf_counter()
            rgb_b64 = self._encode_png(rgb_frame)
            if timings is not None:
                timings["encode_rgb"] = time.perf_counter() - rgb_encode_start
            _event("zerograsp.encode_rgb.end")

            points32 = points3d.astype(np.float32, copy=False)
            _event("zerograsp.encode_points.start")
            points_encode_start = time.perf_counter()
            points_b64 = self._encode_npy(points32)
            if timings is not None:
                timings["encode_points"] = time.perf_counter() - points_encode_start
            _event("zerograsp.encode_points.end")

            bbox_dict = {
                "x_min": int(bbox[0]),
                "y_min": int(bbox[2]),
                "x_max": int(bbox[1]),
                "y_max": int(bbox[3]),
            }

            payload: Dict[str, Any] = {
                "rgb": rgb_b64,
                "points3d": points_b64,
                "bbox": bbox_dict,
                "camera": camera_params,
                "frame_id": "frame-0",
                "object_id": 0,
                "grasp_limit": 1,
            }
            sam2_url = os.getenv("SAM2_URL", "http://10.46.118.233:5000")
            if sam2_url:
                payload["sam2_url"] = sam2_url

            json_payload = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
            payload_bytes = len(json_payload.encode("utf-8"))
            payload_mb = payload_bytes / (1024 * 1024)
            if timings is not None:
                timings["payload_size_mb"] = payload_mb

            try:
                timeout = float(os.getenv("ZEROGRASP_TIMEOUT", "60"))
            except ValueError:
                timeout = 60.0

            request_start = None
            try:
                _debug(
                    f"posting payload: rgb={rgb_frame.shape}, points3d={points3d.shape}, bbox={bbox_dict}, "
                    f"camera=({camera_params.get('width')}, {camera_params.get('height')})",
                )
                _event("zerograsp.request.start")
                request_start = time.perf_counter()
                resp = requests.post(
                    zerograsp_url,
                    data=json_payload,
                    headers={"Content-Type": "application/json"},
                    timeout=timeout,
                )
                resp.raise_for_status()
            except requests.Timeout:
                print("[ZeroGrasp] 请求超时")
                return None
            except Exception as exc:  # noqa: BLE001
                print(f"[ZeroGrasp] 请求失败: {exc}")
                return None
            finally:
                if timings is not None and request_start is not None:
                    request_duration = time.perf_counter() - request_start
                    timings["request"] = request_duration
                    if request_duration > 0:
                        timings["request_bandwidth_mb_s"] = payload_mb / request_duration
                        timings["request_bandwidth_mbit_s"] = (payload_mb * 8) / request_duration
                if request_start is not None:
                    _event("zerograsp.request.end")

            try:
                return self._to_serializable(resp.json())
            except Exception:
                return {"raw": resp.text}
        finally:
            if timings is not None:
                timings["total"] = time.perf_counter() - total_start
            _event("zerograsp.end")

    @staticmethod
    def _to_serializable(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.floating, np.integer)):
            return value.item()
        if isinstance(value, dict):
            return {k: TargetLocalizer._to_serializable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [TargetLocalizer._to_serializable(v) for v in value]
        return value


__all__ = [
    "Any",
    "DepthSnapshot",
    "decode_snapshot",
    "fetch_snapshot",
    "TargetLocalizer",
]
