import os
import queue
import threading
import time
import uuid
from typing import Any, Dict, List, Optional

import numpy as np
import requests
from PIL import Image

from tools.logging.task_logger import log_info, log_warning  # type: ignore
from tools.vision.sam_client import generate_mask  # type: ignore


class SamMaskWorker:
    """Asynchronous worker that generates SAM masks without blocking the main thread."""

    def __init__(self, max_queue_size: int = 10) -> None:
        self.queue: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=max_queue_size)
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._thread.start()
        self._frontend_endpoint = os.getenv(
            "SAM_MASK_RESULT_ENDPOINT",
            "http://127.0.0.1:8000/api/vlm/mask",
        )

    def submit(self, image_path: str, surface_points: Optional[List[List[int]]]) -> Optional[str]:
        """Enqueue a mask generation job; return job_id for later retrieval."""
        if not image_path or not surface_points:
            return None
        job_id = str(uuid.uuid4())
        job = {
            "id": job_id,
            "image_path": image_path,
            "surface_points": surface_points,
            "event": threading.Event(),
            "timestamp": time.time(),
        }
        with self._lock:
            self._jobs[job_id] = job
        try:
            self.queue.put_nowait(job)
        except queue.Full:
            log_warning("⚠️ SAM掩码生成队列已满，跳过本次任务")
            job["error"] = "queue_full"
            job["event"].set()
            with self._lock:
                self._jobs.pop(job_id, None)
            return None
        return job_id

    def wait_for_result(self, job_id: Optional[str], timeout: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """Wait for the specified job to finish and return its result."""
        if not job_id:
            return None
        job = self._jobs.get(job_id)
        if not job:
            return None
        job["event"].wait(timeout)
        if not job["event"].is_set():
            return None
        result = job.get("result")
        if result:
            result = dict(result)
        with self._lock:
            self._jobs.pop(job_id, None)
        return result

    def _worker_loop(self) -> None:
        while True:
            job = self.queue.get()
            try:
                result = self._process_job(job)
                if result:
                    job["result"] = result
            except Exception as exc:  # noqa: BLE001
                log_warning(f"⚠️ SAM掩码生成失败: {exc}")
                job["error"] = str(exc)
            finally:
                job["event"].set()
                self.queue.task_done()

    def _process_job(self, job: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        image_path = job.get("image_path")
        surface_points = job.get("surface_points") or []
        timestamp = job.get("timestamp", time.time())
        job_id = job.get("id")
        if not image_path or not surface_points:
            return None
        try:
            with Image.open(image_path) as img:
                rgb = np.array(img.convert("RGB"))
        except Exception as exc:
            log_warning(f"[SAM] 无法读取图像: {exc}")
            return None
        mask, score = generate_mask(rgb, surface_points)
        if mask is None:
            return None
        mask = (mask > 0).astype(np.uint8) * 255
        base_dir = os.path.dirname(image_path)
        ts = int(time.time() * 1000)
        mask_name = f"sam_mask_{ts}.png"
        mask_path = os.path.join(base_dir, mask_name)
        try:
            Image.fromarray(mask, mode="L").save(mask_path, format="PNG")
        except Exception as exc:
            log_warning(f"[SAM] 保存掩码失败: {exc}")
            return None
        url = self._path_to_static_url(mask_path)
        log_info(f"[SAM] 掩码生成完成: {mask_path}")
        payload = {
            "path": mask_path,
            "url": url,
            "score": score,
            "image_path": image_path,
            "timestamp": timestamp,
            "job_id": job_id,
        }
        self._notify_frontend(payload)
        return payload

    def _notify_frontend(self, payload: Dict[str, Any]) -> None:
        """Post mask metadata to the FastAPI UI so它可以刷新预览."""
        endpoint = getattr(self, "_frontend_endpoint", None)
        if not endpoint:
            return
        try:
            requests.post(endpoint, json=payload, timeout=2)
        except Exception as exc:
            log_warning(f"[SAM] 无法将掩码信息推送到前端: {exc}")

    @staticmethod
    def _path_to_static_url(abs_path: str) -> str:
        normalized = abs_path.replace("\\", "/")
        idx = normalized.find("/static/")
        return normalized[idx:] if idx != -1 else normalized


sam_mask_worker = SamMaskWorker()

__all__ = ["SamMaskWorker", "sam_mask_worker"]
