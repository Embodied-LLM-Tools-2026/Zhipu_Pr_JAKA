"""
Visual observer that queries the VLM for structured detections.

The observer is responsible for capturing camera frames, preparing them
for the VLM, parsing the response into an ObservationResult and
forwarding lightweight payloads to the UI for visualisation.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import threading

import dashscope  # type: ignore
import numpy as np
import requests
from PIL import Image, ImageDraw

try:
    import cv2  # type: ignore

    CV2_AVAILABLE = True
except Exception:  # pragma: no cover
    cv2 = None
    CV2_AVAILABLE = False

from tools.logging.task_logger import log_error, log_info, log_success, log_warning  # type: ignore
from tools.vision.upload_image import upload_file_and_get_url  # type: ignore  # noqa: E402
from .sam_worker import sam_mask_worker  # type: ignore

from ..utils.config import Config
from ..control.task_structures import ObservationPhase, ObservationResult
from .localize_target import fetch_snapshot
from action_sequence.navigate import Navigate
@dataclass
class ObservationContext:
    step: int
    max_steps: int


def _create_csrt_tracker():
    if not CV2_AVAILABLE:
        return None
    if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerCSRT_create"):
        return cv2.legacy.TrackerCSRT_create()
    if hasattr(cv2, "TrackerCSRT_create"):
        return cv2.TrackerCSRT_create()
    return None


def _xyxy_to_xywh(bbox: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = bbox
    return (x1, y1, max(1.0, x2 - x1), max(1.0, y2 - y1))


def _xywh_to_xyxy(bbox: Tuple[float, float, float, float]) -> Tuple[int, int, int, int]:
    x, y, w, h = bbox
    return (
        int(round(x)),
        int(round(y)),
        int(round(x + w)),
        int(round(y + h)),
    )


def _clamp_bbox(bbox: Tuple[int, int, int, int], width: int, height: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(width - 1, x1))
    x2 = max(0, min(width - 1, x2))
    y1 = max(0, min(height - 1, y1))
    y2 = max(0, min(height - 1, y2))
    if x2 <= x1:
        x2 = min(width - 1, x1 + 1)
    if y2 <= y1:
        y2 = min(height - 1, y1 + 1)
    return (x1, y1, x2, y2)


class TrackerManager:
    """Lightweight CSRT tracker wrapper to reuse VLM bbox between observations."""

    def __init__(self, max_age: float = 2.5) -> None:
        self.max_age = max_age
        self._tracker = None
        self._last_update = 0.0
        self._last_bbox: Optional[Tuple[int, int, int, int]] = None
        self._frame_size: Optional[Tuple[int, int]] = None
        self._lock = threading.RLock()

    def reset(self, frame: Optional[np.ndarray], bbox_xyxy: List[float]) -> bool:
        if not CV2_AVAILABLE or frame is None or frame.size == 0:
            self.invalidate()
            return False
        tracker = _create_csrt_tracker()
        if tracker is None:
            self.invalidate()
            return False
        x1, y1, x2, y2 = map(float, bbox_xyxy)
        xywh = _xyxy_to_xywh((x1, y1, x2, y2))
        ok = tracker.init(frame, xywh)
        if not ok:
            self.invalidate()
            return False
        h, w = frame.shape[:2]
        with self._lock:
            self._tracker = tracker
            self._frame_size = (w, h)
            self._last_bbox = _clamp_bbox(_xywh_to_xyxy(xywh), w, h)
            self._last_update = time.time()
        return True

    def update(self, frame: Optional[np.ndarray]) -> Optional[Tuple[int, int, int, int]]:
        if frame is None or frame.size == 0:
            return None
        with self._lock:
            if not self._tracker or (time.time() - self._last_update) > self.max_age:
                self.invalidate()
                return None
            ok, bbox = self._tracker.update(frame)
            if not ok:
                self.invalidate()
                return None
            h, w = frame.shape[:2]
            xyxy = _clamp_bbox(_xywh_to_xyxy(bbox), w, h)
            self._last_bbox = xyxy
            self._last_update = time.time()
            self._frame_size = (w, h)
            return xyxy

    def is_active(self) -> bool:
        # 🚫 临时禁用 CSRT 追踪器，强制使用 VLM
        return False
        # === 原始代码（已禁用）===
        # with self._lock:
        #     if self._tracker is None:
        #         return False
        #     if (time.time() - self._last_update) > self.max_age:
        #         self.invalidate()
        #         return False
        #     return True

    def invalidate(self) -> None:
        with self._lock:
            self._tracker = None
            self._last_bbox = None
            self._frame_size = None
            self._last_update = 0.0

    @property
    def last_bbox(self) -> Optional[Tuple[int, int, int, int]]:
        with self._lock:
            return self._last_bbox


class VLMObserver:
    """Encapsulates VLM interaction for image understanding."""

    def __init__(self, cam_name: str = "front") -> None:
        self.cam_name = cam_name
        self.vlm_api_key = os.getenv("Zhipu_real_demo_API_KEY")
        if not self.vlm_api_key:
            raise ValueError("请设置Zhipu_real_demo_API_KEY环境变量用于DashScope调用")
        self.vlm_model = Config.VLM_NAME
        self.target_resolution = (1000, 1000)
        # self.tracker = TrackerManager(max_age=float(os.getenv("TRACKER_MAX_AGE", "2.5"))) # Removed CSRT
        self._frame_lock = threading.RLock()
        self._latest_capture: Optional[VLMObserver.CaptureInfo] = None
        self._latest_frame: Optional[np.ndarray] = None
        self._frame_thread: Optional[threading.Thread] = None
        self._frame_stop = threading.Event()
        self._frame_interval = float(os.getenv("TRACKER_FRAME_INTERVAL", "0.25"))

    def _start_frame_loop(self) -> None:
        pass # Disabled frame loop as tracker is removed
        # if self._frame_thread and self._frame_thread.is_alive():
        #     return
        # self._frame_stop.clear()
        # self._frame_thread = threading.Thread(target=self._frame_loop, name="vlm_frame_loop", daemon=True)
        # self._frame_thread.start()

    def _frame_loop(self) -> None:
        pass # Disabled
        # while not self._frame_stop.is_set():
        #     capture = self._capture_image(self.cam_name)
        #     if capture.path:
        #         frame = self._load_frame_bgr(capture.path)
        #         if frame is not None:
        #             capture.width = frame.shape[1]
        #             capture.height = frame.shape[0]
        #         self._update_cached_frame(capture, frame)
        #         if frame is not None:
        #             self.tracker.update(frame)
        #     time.sleep(max(0.05, self._frame_interval))

    def _update_cached_frame(self, capture: "VLMObserver.CaptureInfo", frame: Optional[np.ndarray]) -> None:
        with self._frame_lock:
            self._latest_capture = capture
            self._latest_frame = frame

    def _get_latest_frame(self) -> Tuple[Optional["VLMObserver.CaptureInfo"], Optional[np.ndarray]]:
        with self._frame_lock:
            return self._latest_capture, self._latest_frame

    # ------------------------------------------------------------------
    def observe(
        self,
        target_name: str,
        phase: ObservationPhase,
        context: ObservationContext,
        navigator: Navigate,
        *,
        force_vlm: bool = False,
        analysis_request: Optional[str] = None,
    ) -> Tuple[ObservationResult, Dict[str, Any]]:
        """
        Capture image and perform observation.
        If force_vlm=True (default for detection), call VLM.
        If force_vlm=False, return RGB image only (no detection).
        """
        # 1. Capture Image
        image_capture = self._capture_image(self.cam_name)
        if not image_capture.path:
            raise RuntimeError("采集图片失败")
        
        # 2. If force_vlm=False, return "Capture Only" result (RGB only)
        if not force_vlm:
            # Return a minimal result with the image path, but no detection info
            observation = ObservationResult(
                found=False, # Not checked
                confidence=0.0,
                bbox=None,
                source="capture_only",
                original_image_path=image_capture.path,
                processed_image_path=image_capture.path,
                timestamp=time.time(),
                target_id=target_name
            )
            # Try to get depth snapshot if available, but don't fail hard
            try:
                observation.depth_snapshot = fetch_snapshot()
            except Exception:
                pass
                
            observation.robot_pose = navigator.get_current_pose()
            return observation, {}

        # 3. Full VLM Detection (force_vlm=True)
        prep_info = self._prepare_image_for_vlm(image_capture.path)
        processed_path = prep_info["path"]
        original_size = prep_info["original_size"]
        processed_size = prep_info["processed_size"]
        
        image_url = upload_file_and_get_url(
            api_key=self.vlm_api_key,
            model_name=self.vlm_model,
            file_path=processed_path,
        )

        prompt = self._build_prompt(target_name, phase, context, analysis_request=analysis_request)
        response = self._call_vlm(image_url, prompt)
        observation, parsed_payload = self._parse_response(
            response,
            processed_size,
            original_size,
            processed_path,
            image_capture.path,
        )
        observation.source = "vlm"
        
        # Get depth snapshot for 3D localization
        depth_snapshot = fetch_snapshot()
        if not depth_snapshot:
            log_warning("⚠️ [Observer] 采集深度快照失败，无法进行3D定位")
        observation.depth_snapshot = depth_snapshot
        observation.robot_pose = navigator.get_current_pose()

        payload = self._build_frontend_payload(
            observation, parsed_payload, processed_size
        )

        if payload:
            self._push_detection_to_frontend(payload)
            
        return observation, payload

    # ------------------------------------------------------------------
    # Camera helpers
    # ------------------------------------------------------------------
    @dataclass
    class CaptureInfo:
        path: str
        width: int
        height: int
        url: str

    def _capture_image(self, cam_name: str) -> "VLMObserver.CaptureInfo":
        try:
            resp = requests.get(
                f"http://127.0.0.1:8000/api/capture?cam={cam_name}", timeout=3
            )
            data = resp.json()
            path = data["url"]
            width = data.get("w", 0)
            height = data.get("h", 0)
            return VLMObserver.CaptureInfo(
                path=path,
                width=width,
                height=height,
                url=str(path),
            )
        except Exception as exc:
            log_error(f"❌ 采集图片失败: {exc}")
            return VLMObserver.CaptureInfo(path="", width=0, height=0, url="")

    @staticmethod
    def _load_frame_bgr(image_path: str) -> Optional[np.ndarray]:
        if not CV2_AVAILABLE or not image_path:
            return None
        try:
            frame = cv2.imread(image_path)
            if frame is None or frame.size == 0:
                return None
            return frame
        except Exception:
            return None

    def _build_tracker_observation(
        self,
        bbox: Tuple[int, int, int, int],
        image_path: str,
        frame_shape: Tuple[int, int],
    ) -> Tuple[ObservationResult, Dict[str, Any], List[int]]:
        width, height = int(frame_shape[0]), int(frame_shape[1])
        bbox_list = [int(b) for b in bbox]
        observation = ObservationResult(
            found=True,
            bbox=bbox_list,
            confidence=0.5,
            range_estimate=None,
            raw_response={"source": "tracker", "bbox": bbox_list},
            processed_image_path=image_path,
            original_image_path=image_path,
            source="tracker",
        )
        annotated_url: Optional[str] = None
        try:
            annotated = self._save_annotated_image(image_path, [bbox_list])
            annotated_url = annotated.get("url")
            observation.annotated_url = annotated_url
        except Exception as exc:
            log_warning(f"⚠️ 保存跟踪标注图失败: {exc}")
        payload = {
            "found": True,
            "mapped_bbox": bbox_list,
            "original_bbox": bbox_list,
            "confidence": observation.confidence,
            "source": "tracker",
            "annotated_url": annotated_url,
        }
        processed_size = [width, height]
        return observation, payload, processed_size

    def _prepare_image_for_vlm(self, image_path: str) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "path": image_path,
            "original_size": [0, 0],
            "processed_size": [0, 0],
            "resized": False,
        }
        if not os.path.isfile(image_path):
            log_warning(f"[处理] 图片路径不存在: {image_path}")
            return info
        try:
            with Image.open(image_path) as img:
                original_size = list(img.size)
                info["original_size"] = original_size
                target = self.target_resolution
                if target and tuple(original_size) != target:
                    resized_img = img.resize(target, Image.BILINEAR)
                    base_dir = os.path.dirname(image_path)
                    ts = int(time.time() * 1000)
                    new_name = f"vlm_{ts}_{target[0]}x{target[1]}.jpg"
                    processed_path = os.path.join(base_dir, new_name)
                    resized_img.save(processed_path, format="JPEG", quality=90)
                    info.update(
                        {
                            "path": processed_path,
                            "processed_size": [target[0], target[1]],
                            "resized": True,
                        }
                    )
                    log_info(f"[处理] 图片尺寸 {original_size} -> {info['processed_size']}")
                else:
                    info["processed_size"] = original_size
        except Exception as exc:
            log_warning(f"[处理] 图片尺寸调整失败: {exc}")

        if info["processed_size"] == [0, 0]:
            try:
                with Image.open(info["path"]) as fallback_img:
                    info["processed_size"] = list(fallback_img.size)
            except Exception:
                pass
        return info

    # ------------------------------------------------------------------
    # Prompt & VLM interaction
    # ------------------------------------------------------------------
    def _build_prompt(
        self,
        target_name: str,
        phase: ObservationPhase,
        context: ObservationContext,
        *,
        analysis_request: Optional[str] = None,
    ) -> str:
        base = [
            f"你是部署在服务机器人上的视觉助手。目标物体是“{target_name}”。",
            "请分析图像并返回结构化JSON，仅包含视觉信息，不要给出动作建议。",
            "如果未发现目标，请保持found=false并给出简要中文分析。",
            "字段说明：",
            "{",
            '  "found": true/false,',
            '  "bbox": [x_min, y_max, x_max, y_min],',
            '  "confidence": number (0-1),',
            '  "surface_points": [[x, y], ...] // 1-2个背景平面点，一般是指承载目标物体的平面，注意，点一定要打在平面上，不要打在平面之外',
            '  "description": "string" // 简要描述物体的外观、材质（如soft, rigid, metal, plastic等）',
            "}",
        ]
        if analysis_request:
            base.append(
                '如果需要额外说明，请在JSON中加入字段 "analysis": "使用中文描述，' + analysis_request + '"。'
            )
        base.append("禁止返回其他键。确保JSON合法。")
        return "\n".join(base)

    def _call_vlm(self, image_url: str, prompt: str) -> Dict[str, Any]:
        response = dashscope.MultiModalConversation.call(
            api_key=self.vlm_api_key,
            model=self.vlm_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"image": image_url},
                        {"text": prompt},
                    ],
                }
            ],
            max_tokens=400,
            temperature=0.01,
            response_format={"type": "json_object"},
        )
        content = response.output.choices[0].message.content  # type: ignore[index]
        if isinstance(content, list):
            if content and isinstance(content[0], dict) and "text" in content[0]:
                result_text = content[0]["text"]
            else:
                result_text = json.dumps(content, ensure_ascii=False)
        elif isinstance(content, dict):
            result_text = content.get("text") or json.dumps(content, ensure_ascii=False)
        else:
            result_text = str(content)
        return json.loads(result_text)

    # ------------------------------------------------------------------
    # Response parsing & UI payload
    # ------------------------------------------------------------------
    def _parse_response(
        self,
        result: Dict[str, Any],
        processed_size: List[int],
        original_size: List[int],
        processed_path: str,
        original_path: str,
    ) -> Tuple[ObservationResult, Dict[str, Any]]:
        found = bool(result.get("found"))
        bbox = result.get("bbox") or []
        confidence = float(result.get("confidence", 0.0) or 0.0)
        range_estimate = result.get("range_estimate", None)
        if isinstance(range_estimate, (str, int, float)):
            try:
                range_estimate = float(range_estimate)
                if range_estimate <= 0:
                    range_estimate = None
            except (TypeError, ValueError):
                range_estimate = None
        surface_points = None
        if isinstance(result.get("surface_points"), list):
            surface_points = []
            for pt in result["surface_points"]:
                if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                    try:
                        surface_points.append([int(pt[0]), int(pt[1])])
                    except (TypeError, ValueError):
                        continue
            if not surface_points:
                surface_points = None

        remapped_bbox = self._remap_bbox_to_original(
            bbox, processed_size, original_size
        )
        remapped_points = self._remap_points_to_original(surface_points, processed_size, original_size)
        final_bbox = remapped_bbox if remapped_bbox else (bbox if isinstance(bbox, list) else [])
        analysis_text = result.get("analysis")
        if isinstance(analysis_text, str):
            analysis_text = analysis_text.strip()
        else:
            analysis_text = None
        
        description = result.get("description")
        if isinstance(description, str):
            description = description.strip()
        else:
            description = None

        observation = ObservationResult(
            found=found,
            bbox=final_bbox,
            confidence=confidence,
            range_estimate=range_estimate,
            surface_points=remapped_points,
            raw_response=result,
            processed_image_path=processed_path,
            original_image_path=original_path,
            analysis=analysis_text,
            description=description,
        )
        observation.source = "vlm"
        boxes_for_ui = [final_bbox] if final_bbox else []
        annotated = None
        if boxes_for_ui:
            annotated = self._save_annotated_image(
                original_path,
                boxes_for_ui,
                surface_points=observation.surface_points,
            )
            if annotated:
                observation.annotated_url = annotated.get("url")

        if observation.surface_points:
            job_id = sam_mask_worker.submit(
                image_path=observation.original_image_path,
                surface_points=observation.surface_points,
            )
            observation.surface_mask_task_id = job_id

        payload = {
            "found": found,
            "original_bbox": bbox,
            "mapped_bbox": remapped_bbox,
            "confidence": confidence,
            "range_estimate": range_estimate,
            "annotated_url": observation.annotated_url,
            "surface_points": observation.surface_points,
        }
        return observation, payload

    def _build_frontend_payload(
        self,
        observation: ObservationResult,
        parsed_payload: Dict[str, Any],
        processed_size: List[int],
    ) -> Dict[str, Any]:
        """
        Prepare a compact payload that the FastAPI UI endpoint can consume.

        The UI expects xyxy boxes along with basic metadata so we normalise
        everything here before posting.
        """
        if observation is None:
            return {}
        payload_src = parsed_payload if isinstance(parsed_payload, dict) else {}
        original_bbox = payload_src.get("original_bbox") or payload_src.get("bbox") or []
        if not isinstance(original_bbox, list):
            original_bbox = []
        mapped_bbox = payload_src.get("mapped_bbox")
        if not mapped_bbox and observation.bbox:
            mapped_bbox = observation.bbox
        boxes = [observation.bbox] if observation.bbox else []
        payload: Dict[str, Any] = {
            "found": observation.found,
            "bbox": observation.bbox,
            "boxes": boxes,
            "bbox_mode": "xyxy",
            "mapped_bbox": mapped_bbox,
            "original_bbox": original_bbox,
            "processed_size": processed_size,
            "confidence": observation.confidence,
            "range_estimate": observation.range_estimate,
            "surface_points": observation.surface_points,
            "annotated_url": observation.annotated_url,
            "processed_image_path": observation.processed_image_path,
            "original_image_path": observation.original_image_path,
            "raw_result": parsed_payload,
            "mask_url": observation.surface_mask_url,
            "mask_score": observation.surface_mask_score,
            "analysis": observation.analysis,
        }
        payload["source"] = getattr(observation, "source", payload_src.get("source") or "vlm")
        return payload

    def _push_detection_to_frontend(self, payload: Dict[str, Any]) -> None:
        try:
            response = requests.post(
                "http://127.0.0.1:8000/api/vlm/result",
                json=payload,
                timeout=3,
            )
            if response.status_code != 200:
                log_warning(f"⚠️ 推送VLM结果到前端失败: {response.status_code}")
            else:
                log_success("✓ 标注结果已推送前端")
        except Exception as exc:
            log_error(f"❌ 推送VLM结果失败: {exc}")

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _remap_bbox_to_original(
        bbox: List[float],
        processed_size: List[int],
        original_size: List[int],
    ) -> Optional[List[int]]:
        if not bbox or len(bbox) != 4:
            return None
        proc_w, proc_h = processed_size
        orig_w, orig_h = original_size
        if proc_w == 0 or proc_h == 0:
            return None
        sx = orig_w / proc_w
        sy = orig_h / proc_h
        x_min, y_max, x_max, y_min = bbox
        x_min *= sx
        x_max *= sx
        y_min *= sy
        y_max *= sy
        top = y_max
        bottom = y_min
        remapped = [int(x_min), int(top), int(x_max), int(bottom)]
        return remapped

    @staticmethod
    @staticmethod
    def _remap_points_to_original(
        points: Optional[List[List[int]]],
        processed_size: List[int],
        original_size: List[int],
    ) -> Optional[List[List[int]]]:
        if not points:
            return None
        proc_w, proc_h = processed_size
        orig_w, orig_h = original_size
        if proc_w == 0 or proc_h == 0:
            return None
        sx = orig_w / proc_w
        sy = orig_h / proc_h
        mapped: List[List[int]] = []
        for pt in points:
            if len(pt) < 2:
                continue
            x = int(max(0, min(orig_w - 1, pt[0] * sx)))
            y = int(max(0, min(orig_h - 1, pt[1] * sy)))
            mapped.append([x, y])
        return mapped or None

    def _save_annotated_image(
        self,
        image_path: str,
        boxes: List[List[int]],
        surface_points: Optional[List[List[int]]] = None,
    ) -> Dict[str, str]:
        base_dir = os.path.dirname(image_path)
        with Image.open(image_path) as img:
            draw = ImageDraw.Draw(img)
            for x1, y1, x2, y2 in boxes:
                draw.rectangle([(x1, y1), (x2, y2)], outline="red", width=4)
            if surface_points:
                for px, py in surface_points:
                    r = 6
                    draw.ellipse(
                        [(px - r, py - r), (px + r, py + r)],
                        outline="blue",
                        fill="blue",
                    )
            ts = int(time.time() * 1000)
            annotated_name = f"annotated_{ts}.jpg"
            annotated_path = os.path.join(base_dir, annotated_name)
            img.save(annotated_path, format="JPEG", quality=90)
        return {
            "path": annotated_path,
            "url": self._path_to_static_url(annotated_path),
        }

    @staticmethod
    def _path_to_static_url(abs_path: str) -> str:
        normalized = abs_path.replace("\\", "/")
        idx = normalized.find("/static/")
        return normalized[idx:] if idx != -1 else normalized
