#!/usr/bin/env python3
"""
Utility helpers for interacting with an external SAM (Segment Anything) service.

The client expects the environment variable `SAM_API_URL` to point to an HTTP
endpoint that accepts POST requests with a JSON body:

{
  "image": "<base64 PNG>",
  "points": [[x, y], ...]
}

and returns

{
  "mask": "<base64 PNG mask>"
}

If the environment variable is absent or the request fails, the client falls
back to generating a simple circular mask around the provided points so that
the rest of the pipeline keeps functioning during development.
"""

import base64
import io
import json
import os
from typing import Any, List, Optional

import numpy as np
import requests
from PIL import Image


SAM_API_URL = os.getenv("SAM_API_URL")
SAM_API_TIMEOUT = float(os.getenv("SAM_API_TIMEOUT", "3.0"))


def _encode_png(image: np.ndarray, mode: str) -> str:
    img = Image.fromarray(image, mode=mode)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _decode_mask(data: str) -> np.ndarray:
    raw = base64.b64decode(data)
    image = Image.open(io.BytesIO(raw))
    mask = np.array(image)
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    return mask


def _fallback_mask(shape: tuple, points: List[List[float]], radius: int = 40) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    if not points:
        return mask
    h, w = shape
    for px, py in points:
        cx = int(round(px))
        cy = int(round(py))
        yy, xx = np.ogrid[:h, :w]
        distance = (xx - cx) ** 2 + (yy - cy) ** 2
        mask[distance <= radius * radius] = 255
    return mask


def generate_mask(
    rgb_image: np.ndarray,
    points: List[List[float]],
) -> Optional[np.ndarray]:
    """Generate a binary mask using the SAM service or a local fallback."""
    if rgb_image is None or not points:
        return None

    if SAM_API_URL:
        try:
            payload = {
                "image": _encode_png(rgb_image.astype(np.uint8, copy=False), "RGB"),
                "points": points,
            }
            resp = requests.post(
                SAM_API_URL,
                data=json.dumps(payload),
                timeout=SAM_API_TIMEOUT,
                headers={"Content-Type": "application/json"},
            )
            resp.raise_for_status()
            data = resp.json()
            mask_data = data.get("mask")
            if not mask_data:
                raise ValueError("SAM response missing 'mask'")
            mask = _decode_mask(mask_data)
            return mask
        except Exception as exc:  # noqa: B902
            print(f"[SAM] Remote mask generation failed: {exc}")

    # Fallback: simple circular mask around the provided points
    height, width = rgb_image.shape[:2]
    return _fallback_mask((height, width), points)
