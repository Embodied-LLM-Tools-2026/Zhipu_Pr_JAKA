import base64
import math
import os
import random
import sys
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
from tools.zerograsp_wrapper import get_shared_zerograsp_runner

DEFAULT_DEPTH_API = os.getenv("DEPTH_FRAME_API", "http://127.0.0.1:8000/api/depth/frame")

@dataclass
class DepthSnapshot:
    depth: np.ndarray
    intrinsics: Any
    extrinsic: Any
    dtype: str = "uint16"

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
    response = requests.get(api_url, timeout=timeout)
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
    ) -> Optional[Dict[str, Any]]:
        snapshot = fetch_snapshot(self.depth_api, timeout=timeout)
        return self.localize_object(
            bbox=bbox,
            snapshot=snapshot,
            surface_points_hint=surface_points_hint,
            range_estimate=range_estimate,
            rgb_frame=rgb_frame,
        )

    def localize_object(
        self,
        *,
        bbox: Iterable[float],
        snapshot: DepthSnapshot,
        surface_points_hint: Optional[List[List[float]]] = None,
        range_estimate: Optional[float] = None,
        rgb_frame: Optional[np.ndarray] = None,
    ) -> Optional[Dict[str, Any]]:
        bbox_int = list(map(int, bbox))
        if len(bbox_int) != 4:
            raise ValueError("bbox must contain four numbers")

        transform = self._transform_points(
            snapshot=snapshot,
            bbox=bbox_int,
            rgb_frame=rgb_frame,
            surface_points_hint=surface_points_hint,
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
            "bbox": transform["bbox"],
        }

        # using absolute threshold for close-range objects
        # if range_estimate is not None and range_estimate <= 0.5:
        #     zg = self._run_zero_grasp_inference(
        #         transform_result=transform,
        #         bbox=transform["bbox"],
        #         rgb_frame=rgb_frame,
        #     )
        #     if zg is not None:
        #         payload["zerograsp"] = zg

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
    ) -> Optional[Dict[str, Any]]:
        intr = snapshot.intrinsics
        depth = snapshot.depth
        height, width = depth.shape
        extrinsic = snapshot.extrinsic
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
                    point = transformation2dto3d(
                            OBPoint2f(float(ix), float(iy)), depth_val, intr, extrinsic
                        )
                    if point is not None:
                        out.append([float(point.x), float(point.y), float(point.z)])
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

        surface_mask = self._build_surface_mask(
            rgb_frame=rgb_frame,
            surface_points_hint=surface_points_hint,
            depth_width=width,
            depth_height=height,
        )

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
            "bbox": (x_min, x_max, y_min, y_max),
            "depth_data": depth,
            "depth_width": width,
            "depth_height": height,
        }

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
            mask = generate_mask(rgb_frame, surface_points_hint)
        except Exception as exc:
            print(f"[SAM] 生成mask失败: {exc}")
            return None
        if mask is None:
            return None
        if mask.shape[0] != depth_height or mask.shape[1] != depth_width:
            if cv2 is not None:
                mask = cv2.resize(mask, (depth_width, depth_height), interpolation=cv2.INTER_NEAREST)
            else:
                mask = np.array(Image.fromarray(mask).resize((depth_width, depth_height)))
        return mask

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

    def _run_zero_grasp_inference(
        self,
        *,
        transform_result: Dict[str, Any],
        bbox: Tuple[int, int, int, int],
        rgb_frame: Optional[np.ndarray],
    ) -> Optional[Dict[str, Any]]:
        ws_url = os.getenv("ZEROGRASP_WS_URL")
        if not ws_url:
            return None
        runner = get_shared_zerograsp_runner(
            ws_url=ws_url,
            camera_cfg_path=os.getenv("ZEROGRASP_CAMERA_CFG"),
        )
        if runner is None:
            return None

        depth_data = transform_result.get("depth_data")
        if depth_data is None or depth_data.size == 0:
            return None

        if rgb_frame is None:
            return None

        depth_height, depth_width = depth_data.shape
        if rgb_frame.shape[0] != depth_height or rgb_frame.shape[1] != depth_width:
            if cv2 is None:
                return None
            rgb_frame = cv2.resize(
                rgb_frame, (depth_width, depth_height), interpolation=cv2.INTER_AREA
            )

        mask = np.zeros((depth_height, depth_width), dtype=np.uint8)
        x_min, x_max, y_min, y_max = bbox
        x_min = max(0, int(x_min))
        x_max = min(depth_width, int(x_max))
        y_min = max(0, int(y_min))
        y_max = min(depth_height, int(y_max))
        if x_min >= x_max or y_min >= y_max:
            return None
        mask[y_min:y_max, x_min:x_max] = 255

        try:
            result = runner.infer(
                rgb_image=rgb_frame.astype(np.uint8, copy=False),
                depth_image=depth_data.astype(np.uint16, copy=False),
                mask_image=mask,
            )
        except Exception as exc:  # pragma: no cover
            print(f"[ZeroGrasp] Runtime error: {exc}")
            return None

        if result is None:
            return None
        if isinstance(result, dict):
            return self._to_serializable(result)
        return {"raw": self._to_serializable(result)}

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
