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
from typing import Any, List, Optional, Tuple

import numpy as np
import requests
from PIL import Image


_DEFAULT_SAM_ENDPOINT = "http://10.46.147.25:5000/predict"
SAM_API_URL = os.getenv("SAM_API_URL") or _DEFAULT_SAM_ENDPOINT
SAM_API_TIMEOUT = float(os.getenv("SAM_API_TIMEOUT", "8.0"))


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


def _resolve_sam_endpoint() -> str:
    url = SAM_API_URL.rstrip("/") if SAM_API_URL else _DEFAULT_SAM_ENDPOINT
    if url.endswith("/predict"):
        return url
    return url + "/predict"


def generate_mask(
    rgb_image: np.ndarray,
    points: List[List[float]],
    point_labels: Optional[List[int]] = None,
) -> Tuple[Optional[np.ndarray], Optional[float]]:
    """
    Generate a binary mask using the SAM service or a local fallback.

    Returns:
        mask: np.ndarray or None
        score: float or None
    """
    if rgb_image is None or not points:
        return None, None

    endpoint = _resolve_sam_endpoint() if SAM_API_URL else None
    if endpoint:
        try:
            pil = Image.fromarray(rgb_image.astype(np.uint8, copy=False))
            buf = io.BytesIO()
            pil.save(buf, format="PNG")
            img_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            payload: dict[str, Any] = {
                "image": f"data:image/png;base64,{img_b64}",
                "point_coords": points,
                "point_labels": point_labels or [1] * len(points),
            }
            resp = requests.post(
                endpoint,
                json=payload,
                timeout=SAM_API_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
            mask_entries = data.get("masks") or []
            mask_payload: Optional[str] = None
            if isinstance(mask_entries, list) and mask_entries:
                mask_payload = mask_entries[0]
            elif isinstance(data.get("mask"), str):
                mask_payload = data["mask"]
            if not mask_payload:
                raise ValueError("SAM response missing mask data")
            if isinstance(mask_payload, str) and mask_payload.startswith("data:"):
                _, _, encoded = mask_payload.partition(",")
                mask_payload = encoded
            mask = _decode_mask(mask_payload)
            scores = data.get("scores")
            score_val: Optional[float] = None
            if isinstance(scores, list) and scores:
                try:
                    score_val = float(scores[0])
                except (TypeError, ValueError):
                    score_val = None
            return mask, score_val
        except Exception as exc:  # noqa: B902
            print(f"[SAM] Remote mask generation failed: {exc}")

    height, width = rgb_image.shape[:2]
    fallback = _fallback_mask((height, width), points)
    return fallback, None
