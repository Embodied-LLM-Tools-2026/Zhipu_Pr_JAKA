import asyncio
import base64
import io
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from PIL import Image

try:
    import websockets  # type: ignore
except Exception:  # noqa: B902
    websockets = None

_SHARED_RUNNER: Optional["ZeroGraspWrapper"] = None


class ZeroGraspWrapper:
    """Wrapper that communicates with a remote ZeroGrasp WebSocket service."""

    def __init__(
        self,
        *,
        ws_url: str,
        camera_cfg_path: Optional[str] = None,
    ) -> None:
        if websockets is None:
            raise ImportError(
                "The 'websockets' package is required for ZeroGrasp integration. "
                "Please install it (e.g., pip install websockets)."
            )
        self.ws_url = ws_url
        self.camera_cfg_path = None
        self.camera_cfg_payload = None
        if camera_cfg_path:
            cfg_path = Path(camera_cfg_path).expanduser().resolve()
            if cfg_path.exists():
                self.camera_cfg_path = str(cfg_path)
                self.camera_cfg_payload = cfg_path.read_text(encoding="utf-8")
            else:
                print(f"[ZeroGrasp] Camera config file not found: {cfg_path}")

    @staticmethod
    def _encode_png(array: np.ndarray, mode: str) -> str:
        img = Image.fromarray(array, mode=mode)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")

    async def _send_payload(self, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        headers = {}
        if self.auth_token:
            headers["Authorization"] = self.auth_token
        try:
            async with websockets.connect(self.ws_url, extra_headers=headers) as ws:
                await ws.send(json.dumps(payload))
                response = await ws.recv()
        except Exception as exc:  # noqa: B902
            print(f"[ZeroGrasp] WebSocket call failed: {exc}")
            return None
        try:
            data = json.loads(response)
        except json.JSONDecodeError:
            return {"raw": response}
        return data

    def infer(
        self,
        *,
        rgb_image: np.ndarray,
        depth_image: np.ndarray,
        mask_image: np.ndarray,
    ) -> Optional[Dict[str, Any]]:
        if rgb_image is None or depth_image is None or mask_image is None:
            return None

        payload: Dict[str, Any] = {
            "rgb_png": self._encode_png(rgb_image.astype(np.uint8, copy=False), "RGB"),
            "depth_png": self._encode_png(depth_image.astype(np.uint16, copy=False), "I;16"),
            "mask_png": self._encode_png(mask_image.astype(np.uint8, copy=False), "L"),
        }
        if self.camera_cfg_payload:
            payload["camera_cfg"] = self.camera_cfg_payload

        async def _run() -> Optional[Dict[str, Any]]:
            return await self._send_payload(payload)

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(_run())
        else:
            if loop.is_running():
                new_loop = asyncio.new_event_loop()
                try:
                    return new_loop.run_until_complete(_run())
                finally:
                    new_loop.close()
            return loop.run_until_complete(_run())


def get_shared_zerograsp_runner(
    *,
    ws_url: Optional[str],
    camera_cfg_path: Optional[str],
) -> Optional[ZeroGraspWrapper]:
    global _SHARED_RUNNER
    if not ws_url:
        return None
    needs_new = False
    cfg_resolved = (
        str(Path(camera_cfg_path).expanduser().resolve())
        if camera_cfg_path
        else None
    )
    if _SHARED_RUNNER is None:
        needs_new = True
    else:
        if (
            _SHARED_RUNNER.ws_url != ws_url
            or _SHARED_RUNNER.camera_cfg_path != cfg_resolved
        ):
            needs_new = True
    if needs_new:
        try:
            _SHARED_RUNNER = ZeroGraspWrapper(
                ws_url=ws_url,
                camera_cfg_path=cfg_resolved,
            )
        except Exception as exc:  # noqa: B902
            print(f"[ZeroGrasp] Unable to initialize runner: {exc}")
            _SHARED_RUNNER = None
    return _SHARED_RUNNER
