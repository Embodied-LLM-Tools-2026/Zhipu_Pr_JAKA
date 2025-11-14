"""
Background worker that accumulates scene-level object detections.

After each primary observation the TaskProcessor can enqueue a job
containing the captured RGB image, depth map and robot pose. The worker
queries the VLM for dense detections, fuses them with depth data to
estimate world coordinates, and registers high-confidence objects into
the world model.
"""

from __future__ import annotations

import json
import os
import threading
import queue
import time
from typing import Any, Dict, List, Optional

import dashscope  # type: ignore
import numpy as np

from config import Config
from executor import SkillExecutor
from localize_target import TargetLocalizer, DepthSnapshot
from task_logger import log_error, log_warning  # type: ignore
from tools.upload_image import upload_file_and_get_url  # type: ignore


class SceneCatalogWorker:
    """Asynchronous scene understanding worker."""

    def __init__(
        self,
        world_model,
        vlm_api_key: Optional[str] = None,
        vlm_model: Optional[str] = None,
        queue_size: int = 20,
    ) -> None:
        self.world = world_model
        self.vlm_api_key = (
            vlm_api_key
            or os.getenv("Zhipu_real_demo_API_KEY")
            or getattr(Config, "Zhipu_real_demo_API_KEY", "")
        )
        if not self.vlm_api_key:
            raise ValueError("SceneCatalogWorker 需要提供 VLM API Key")
        self.vlm_model = vlm_model or Config.VLM_NAME
        self.localizer = TargetLocalizer()
        self.queue: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=queue_size)
        self._thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._thread.start()

    # ------------------------------------------------------------------
    def submit(self, job: Dict[str, Any]) -> None:
        """Enqueue a job consisting of rgb/depth frame and robot pose."""
        try:
            self.queue.put_nowait(job)
        except queue.Full:
            log_warning("⚠️ 场景建模队列已满，跳过一帧")

    # ------------------------------------------------------------------
    def _worker_loop(self) -> None:
        while True:
            job = self.queue.get()
            try:
                self._process_job(job)
            except Exception as exc:
                log_error(f"❌ 场景建模任务失败: {exc}")
            finally:
                self.queue.task_done()
# job = {
#     "image_path": observation.original_image_path,
#     "depth_map": depth_bundle.depth,
#     "depth_intrinsics": depth_bundle.intrinsics,
#     "extrinsic": depth_bundle.extrinsic,
#     "robot_pose": observation.robot_pose,
# }
    def _process_job(self, job: Dict[str, Any]) -> None:
        image_path = job.get("image_path")
        snapshot = DepthSnapshot(
            depth=job.get("depth_map"),
            intrinsics=job.get("depth_intrinsics"),
            extrinsic=job.get("extrinsic"),
        )
        robot_pose = job.get("robot_pose")
        if image_path is None or snapshot is None or robot_pose is None:
            log_warning("⚠️ 场景建模任务缺少必要字段")
            return

        detections = self._detect_objects(image_path)
        if not detections:
            return

        for det in detections:
            label = det.get("label") or det.get("class") or "object"
            confidence = float(det.get("confidence", 0.0) or 0.0)
            bbox = det.get("bbox")
            if not bbox or confidence < 0.6:
                continue
            pose = self._localize_detection(bbox, snapshot, robot_pose)
            if pose is None:
                continue
            camera_center, robot_center, world_center = pose
            attrs = {
                "source": "catalog",
                "bbox": bbox,
                "timestamp": job.get("timestamp", time.time()),
                "image_path": image_path,
            }
            self.world.register_catalog_detection(
                label=label,
                world_center=world_center.tolist(),
                confidence=confidence,
                camera_center=camera_center.tolist(),
                robot_center=robot_center.tolist(),
                attrs=attrs,
            )

    # ------------------------------------------------------------------
    def _detect_objects(self, image_path: str) -> List[Dict[str, Any]]:
        try:
            image_url = upload_file_and_get_url(
                api_key=self.vlm_api_key, model_name=self.vlm_model, file_path=image_path
            )
        except Exception as exc:
            log_warning(f"⚠️ 上传图片失败: {exc}")
            return []

        prompt = """
        你是部署在机器人上的视觉助手，需要列出图像中所有主要的可感知物体，注意是主要的物体，一些杂乱的或者不太会让机器人抓取的物体就不用标了。
        请输出JSON，格式如下：
        {
            "objects": [
                {"label": "red cup", "bbox": [x_min, y_min, x_max, y_max], "confidence": 0.82}
            ]
        }
        坐标使用像素，confidence 范围0-1。禁止输出多余文本。
        """
        try:
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
                temperature=0.2,
                response_format={"type": "json_object"},
            )
            content = response.output.choices[0].message.content  # type: ignore[index]
            if isinstance(content, list):
                payload = content[0].get("text") if content else "{}"
            elif isinstance(content, dict):
                payload = content.get("text", "{}")
            else:
                payload = str(content)
        except Exception as exc:
            log_warning(f"⚠️ 请求场景检测失败: {exc}")
            return []
        try:
            data = json.loads(payload)
        except Exception as exc:
            log_warning(f"⚠️ 解析场景检测结果失败: {exc}")
            return []
        objects = data.get("objects") if isinstance(data, dict) else None
        if not isinstance(objects, list):
            return []
        cleaned: List[Dict[str, Any]] = []
        for obj in objects:
            if not isinstance(obj, dict):
                continue
            bbox = obj.get("bbox")
            if not isinstance(bbox, list) or len(bbox) != 4:
                continue
            try:
                bbox = [float(v) for v in bbox]
                confidence = float(obj.get("confidence", 0.0) or 0.0)
            except (TypeError, ValueError):
                continue
            cleaned.append(
                {
                    "label": obj.get("label") or obj.get("class") or "object",
                    "bbox": bbox,
                    "confidence": confidence,
                }
            )
        return cleaned

    # ------------------------------------------------------------------
    def _localize_detection(
        self,
        bbox: List[float],
        snapshot: DepthSnapshot,
        robot_pose: Dict[str, float],
    ) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        bbox_xyxy: List[int] = []
        try:
            bbox_xyxy = [int(float(v)) for v in bbox]
        except (TypeError, ValueError):
            return None
        try:
            localization = self.localizer.localize_object(
                bbox=bbox_xyxy,
                snapshot=snapshot,
                surface_points_hint=None,
                range_estimate=None,
                rgb_frame=None,
            )
        except Exception as exc:
            log_warning(f"⚠️ 场景对象定位失败: {exc}")
            return None
        if not localization:
            return None
        cam_center = localization.get("obj_center_3d")
        if cam_center is None:
            return None
        cam_point_mm = np.array(cam_center, dtype=float)
        robot_point_mm = SkillExecutor.transform_camera_to_robot(cam_point_mm)
        world_point_m = SkillExecutor.transform_robot_to_world(robot_point_mm, robot_pose)
        return cam_point_mm, robot_point_mm, world_point_m
