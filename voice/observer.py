"""
Visual observer that queries the VLM for structured detections.

The observer is responsible for capturing camera frames, preparing them
for the VLM, parsing the response into an ObservationResult and
forwarding lightweight payloads to the UI for visualisation.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import dashscope  # type: ignore
import numpy as np
import requests
from PIL import Image, ImageDraw

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "tools")))

from task_logger import log_error, log_info, log_success, log_warning  # type: ignore
from upload_image import upload_file_and_get_url  # type: ignore
from sam_client import generate_mask  # type: ignore

from config import Config
from task_structures import ObservationPhase, ObservationResult
from localize_target import fetch_snapshot
from action_sequence.navigate import Navigate
@dataclass
class ObservationContext:
    step: int
    max_steps: int
    last_analysis: Optional[str] = None
    surface_region: Optional[List[int]] = None
    surface_points: Optional[List[List[int]]] = None


class VLMObserver:
    """Encapsulates VLM interaction for image understanding."""

    def __init__(self, cam_name: str = "front") -> None:
        self.cam_name = cam_name
        self.vlm_api_key = os.getenv("Zhipu_real_demo_API_KEY")
        if not self.vlm_api_key:
            raise ValueError("请设置Zhipu_real_demo_API_KEY环境变量用于DashScope调用")
        self.vlm_model = Config.VLM_NAME
        self.target_resolution = (1000, 1000)

    # ------------------------------------------------------------------
    def observe(
        self,
        target_name: str,
        phase: ObservationPhase,
        context: ObservationContext,
        navigator: Navigate,
    ) -> Tuple[ObservationResult, Dict[str, Any]]:
        """Capture, query VLM and return structured observation."""
        image_capture = self._capture_image(self.cam_name)
        if not image_capture.path:
            raise RuntimeError("采集图片失败")
        depth_snapshot = fetch_snapshot()
        if not depth_snapshot:
            raise RuntimeError("采集深度快照失败")
        prep_info = self._prepare_image_for_vlm(image_capture.path)
        processed_path = prep_info["path"]
        original_size = prep_info["original_size"]
        processed_size = prep_info["processed_size"]
        image_url = upload_file_and_get_url(
            api_key=self.vlm_api_key,
            model_name=self.vlm_model,
            file_path=processed_path,
        )

        prompt = self._build_prompt(target_name, phase, context, processed_size)
        response = self._call_vlm(image_url, prompt)
        observation, parsed_payload = self._parse_response(
            response,
            processed_size,
            original_size,
            processed_path,
            image_capture.path,
        )

        payload = self._build_frontend_payload(
            observation, parsed_payload, original_size, processed_size
        )

        if payload:
            self._push_detection_to_frontend(payload)
        observation.depth_snapshot = depth_snapshot
        observation.robot_pose = navigator.get_current_pose()
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
        image_size: List[int],
    ) -> str:
        base = [
            f"你是部署在服务机器人上的视觉助手。目标物体是“{target_name}”。",
            "请分析图像并返回结构化JSON，仅包含视觉信息，不要给出动作建议。",
            "如果未发现目标，请保持found=false并给出简要中文分析。",
            "字段说明：",
            "{",
            '  "found": true/false,',
            '  "bbox": [x_min, y_max, x_max, y_min],',
            '  "image_size": [width, height],',
            '  "confidence": number (0-1),',
            '  "range_estimate": number // 估计距离（米），无法估计用 -1,',
            '  "analysis": "<简短中文说明>",',
            '  "surface_roi": [x_min, y_min, x_max, y_max], // 背景平面区域，承载目标物体的平面',
            '  "surface_points": [[x, y], ...] // 1-2个背景平面点，一般是指承载目标物体的平面，注意，点一定要打在平面上，不要打在平面之外',
            "}",
            "禁止返回其他键。确保JSON合法。",
        ]
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
        analysis = result.get("analysis", "")
        range_estimate = result.get("range_estimate", None)
        if isinstance(range_estimate, (str, int, float)):
            try:
                range_estimate = float(range_estimate)
                if range_estimate <= 0:
                    range_estimate = None
            except (TypeError, ValueError):
                range_estimate = None
        surface_roi = result.get("surface_roi")
        if not isinstance(surface_roi, list) or len(surface_roi) != 4:
            surface_roi = None
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
        annotated = None
        if remapped_bbox:
            boxes_for_ui = [remapped_bbox]
        else:
            boxes_for_ui = []
        if boxes_for_ui:
            annotated = self._save_annotated_image(
                original_path,
                boxes_for_ui,
                surface_region=self._remap_roi_to_original(
                    surface_roi, processed_size, original_size
                ),
                surface_points=self._remap_points_to_original(
                surface_points, processed_size, original_size
                ),
            )

        observation = ObservationResult(
            found=found,
            bbox=remapped_bbox or [],
            image_size=original_size,
            confidence=confidence,
            range_estimate=range_estimate,
            analysis=analysis,
            surface_roi=self._remap_roi_to_original(
                surface_roi, processed_size, original_size
            )
            if surface_roi
            else None,
            surface_points=self._remap_points_to_original(
                surface_points, processed_size, original_size
            ),
            annotated_url=annotated["url"] if annotated else None,
            raw_response=result,
            processed_image_path=processed_path,
            original_image_path=original_path,
        )
        mask_preview = self._maybe_generate_surface_mask(
            original_path=original_path,
            surface_points=observation.surface_points,
        )
        if mask_preview:
            observation.surface_mask_path = mask_preview.get("path")
            observation.surface_mask_url = mask_preview.get("url")
            observation.surface_mask_score = mask_preview.get("score")

        payload = {
            "found": found,
            "original_bbox": bbox,
            "mapped_bbox": remapped_bbox,
            "image_size": original_size,
            "analysis": analysis,
            "confidence": confidence,
            "range_estimate": range_estimate,
            "annotated_url": observation.annotated_url,
            "surface_region": observation.surface_roi,
            "surface_points": observation.surface_points,
        }
        return observation, payload

    def _build_frontend_payload(
        self,
        observation: ObservationResult,
        parsed_payload: Dict[str, Any],
        original_size: List[int],
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
            "image_size": observation.image_size or original_size,
            "processed_size": processed_size,
            "confidence": observation.confidence,
            "analysis": observation.analysis,
            "range_estimate": observation.range_estimate,
            "surface_region": observation.surface_roi,
            "surface_points": observation.surface_points,
            "annotated_url": observation.annotated_url,
            "processed_image_path": observation.processed_image_path,
            "original_image_path": observation.original_image_path,
            "raw_result": parsed_payload,
            "mask_url": observation.surface_mask_url,
            "mask_score": observation.surface_mask_score,
        }
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
    def _remap_roi_to_original(
        roi: Optional[List[int]],
        processed_size: List[int],
        original_size: List[int],
    ) -> Optional[List[int]]:
        if not roi or len(roi) != 4:
            return None
        proc_w, proc_h = processed_size
        orig_w, orig_h = original_size
        if proc_w == 0 or proc_h == 0:
            return None
        sx = orig_w / proc_w
        sy = orig_h / proc_h
        x_min, y_min, x_max, y_max = roi
        x_min = max(0, min(orig_w, x_min * sx))
        x_max = max(0, min(orig_w, x_max * sx))
        y_min = max(0, min(orig_h, y_min * sy))
        y_max = max(0, min(orig_h, y_max * sy))
        if x_min >= x_max or y_min >= y_max:
            return None
        return [int(x_min), int(y_min), int(x_max), int(y_max)]

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
        surface_region: Optional[List[int]] = None,
        surface_points: Optional[List[List[int]]] = None,
    ) -> Dict[str, str]:
        base_dir = os.path.dirname(image_path)
        with Image.open(image_path) as img:
            draw = ImageDraw.Draw(img)
            for x1, y1, x2, y2 in boxes:
                draw.rectangle([(x1, y1), (x2, y2)], outline="red", width=4)
            if surface_region:
                sx1, sy1, sx2, sy2 = surface_region
                draw.rectangle([(sx1, sy1), (sx2, sy2)], outline="green", width=3)
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

    def _maybe_generate_surface_mask(
        self,
        original_path: str,
        surface_points: Optional[List[List[int]]],
    ) -> Optional[Dict[str, Any]]:
        if not surface_points or not original_path:
            return None
        try:
            with Image.open(original_path) as img:
                rgb = np.array(img.convert("RGB"))
        except Exception as exc:
            log_warning(f"[SAM] 无法读取原图生成mask: {exc}")
            return None
        mask, score = generate_mask(rgb, surface_points)
        if mask is None:
            return None
        mask = (mask > 0).astype(np.uint8) * 255
        base_dir = os.path.dirname(original_path)
        ts = int(time.time() * 1000)
        mask_name = f"sam_mask_{ts}.png"
        mask_path = os.path.join(base_dir, mask_name)
        try:
            Image.fromarray(mask, mode="L").save(mask_path, format="PNG")
        except Exception as exc:
            log_warning(f"[SAM] 保存mask失败: {exc}")
            return None
        return {
            "path": mask_path,
            "url": self._path_to_static_url(mask_path),
            "score": score,
        }

    @staticmethod
    def _path_to_static_url(abs_path: str) -> str:
        normalized = abs_path.replace("\\", "/")
        idx = normalized.find("/static/")
        return normalized[idx:] if idx != -1 else normalized
