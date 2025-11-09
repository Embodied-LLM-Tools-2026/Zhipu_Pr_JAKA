#!/usr/bin/env python3
"""
Validate that the numeric projector introduced in TargetLocalizer matches
the official pyorbbecsdk transformation2dto3d results.

This script captures a synchronized color/depth frame from the default
Orbbec device, samples random valid depth pixels, and compares the 3D
points produced by:
  1) pyorbbecsdk.transformation2dto3d (SDK projector)
  2) TargetLocalizer._make_numeric_projector (pure numpy projector)

Run this on the host that has the Orbbec camera connected.
"""

from __future__ import annotations

import argparse
import random
import sys
from typing import List, Tuple

import numpy as np

try:
    from pyorbbecsdk import Config, Pipeline, OBSensorType  # type: ignore
except Exception as exc:  # pragma: no cover
    print(f"[compare_projectors] pyorbbecsdk not available: {exc}")
    sys.exit(1)

from voice.localize_target import (
    TargetLocalizer,
)


def _enable_default_streams(pipeline: Pipeline, config: Config) -> None:
    for sensor_type in (OBSensorType.DEPTH_SENSOR, OBSensorType.COLOR_SENSOR):
        profile_list = pipeline.get_stream_profile_list(sensor_type)
        if profile_list is None:
            raise RuntimeError(f"No stream profile list for sensor {sensor_type}")
        profile = profile_list.get_default_video_stream_profile()
        if profile is None:
            raise RuntimeError(f"No default profile for sensor {sensor_type}")
        config.enable_stream(profile)


def _serialize_intrinsics(intrinsic) -> dict:
    return {
        "fx": float(getattr(intrinsic, "fx", 0.0)),
        "fy": float(getattr(intrinsic, "fy", 0.0)),
        "cx": float(getattr(intrinsic, "cx", 0.0)),
        "cy": float(getattr(intrinsic, "cy", 0.0)),
    }


def _serialize_extrinsic(extrinsic) -> dict:
    rotation = np.asarray(getattr(extrinsic, "rotation", []), dtype=float).reshape(3, 3)
    translation = np.asarray(getattr(extrinsic, "translation", []), dtype=float).reshape(3)
    matrix = np.eye(4, dtype=float)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = translation
    return {"rotation": rotation.tolist(), "translation": translation.tolist(), "matrix": matrix.tolist()}


def _collect_random_pixels(depth: np.ndarray, sample_count: int) -> List[Tuple[int, int]]:
    coords = np.argwhere(depth > 0)
    if coords.size == 0:
        return []
    if len(coords) <= sample_count:
        return [(int(c[1]), int(c[0])) for c in coords]
    picks = random.sample(range(len(coords)), sample_count)
    return [(int(coords[i][1]), int(coords[i][0])) for i in picks]


def compare_projectors(sample_count: int) -> None:
    pipeline = Pipeline()
    config = Config()
    _enable_default_streams(pipeline, config)
    pipeline.start(config)

    try:
        frames = None
        for _ in range(60):
            frames = pipeline.wait_for_frames(1000)
            if frames and frames.get_depth_frame() and frames.get_color_frame():
                break
        if frames is None:
            raise RuntimeError("Unable to fetch synchronized frames")

        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if depth_frame is None or color_frame is None:
            raise RuntimeError("Incomplete frame set")

        depth_profile = depth_frame.get_stream_profile()
        color_profile = color_frame.get_stream_profile()
        depth_video = depth_profile.as_video_stream_profile()

        depth_data = np.frombuffer(
            depth_frame.get_data(), dtype=np.uint16
        ).reshape(depth_video.get_height(), depth_video.get_width())

        sdk_intrinsics = depth_video.get_intrinsic()
        sdk_extrinsic = depth_profile.get_extrinsic_to(color_profile)
        intr_dict = _serialize_intrinsics(sdk_intrinsics)
        extr_dict = _serialize_extrinsic(sdk_extrinsic)

        localizer = TargetLocalizer()
        sdk_projector = localizer._sdk_projector(sdk_intrinsics, sdk_extrinsic)
        numeric_projector = localizer._make_numeric_projector(
            intr_dict, extr_dict, (depth_video.get_height(), depth_video.get_width())
        )
        if sdk_projector is None or numeric_projector is None:
            raise RuntimeError("Failed to build projectors for comparison")

        pixels = _collect_random_pixels(depth_data, sample_count)
        if not pixels:
            raise RuntimeError("No valid depth pixels to sample")

        deltas = []
        for ix, iy in pixels:
            depth_val = float(depth_data[iy, ix])
            p_sdk = sdk_projector(ix, iy, depth_val)
            p_num = numeric_projector(ix, iy, depth_val)
            if p_sdk is None or p_num is None:
                continue
            diff = np.linalg.norm(np.array(p_sdk) - np.array(p_num))
            deltas.append(diff)

        if not deltas:
            raise RuntimeError("No comparable points collected")

        deltas_np = np.array(deltas, dtype=float)
        print(f"[compare_projectors] compared {len(deltas)} samples")
        print(f"  mean delta: {deltas_np.mean():.6f}")
        print(f"  max delta : {deltas_np.max():.6f}")
        print(f"  min delta : {deltas_np.min():.6f}")
    finally:
        try:
            pipeline.stop()
        except Exception:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare SDK projector vs numeric projector.")
    parser.add_argument(
        "--samples",
        type=int,
        default=200,
        help="Number of valid depth pixels to compare.",
    )
    args = parser.parse_args()
    compare_projectors(args.samples)


if __name__ == "__main__":
    main()
