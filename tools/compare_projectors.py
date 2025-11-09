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
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
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


def _serialize_extrinsic(extrinsic, *, debug: bool = False) -> dict:
    if extrinsic is None:
        if debug:
            print("[compare_projectors] extrinsic object is None")
        return {}

    if debug:
        public_attrs = [attr for attr in dir(extrinsic) if not attr.startswith("_")]
        print("[compare_projectors] extrinsic public attrs:", public_attrs)

    def _value(attr_name: str):
        attr = getattr(extrinsic, attr_name, None)
        if debug:
            print(f"[compare_projectors] extrinsic.{attr_name} -> {type(attr)}")
        if attr is None:
            return None
        return attr() if callable(attr) else attr

    rotation_candidates = (
        "rotation",
        "get_rotation",
        "rot",
        "get_rot",
        "rotation_matrix",
        "get_rotation_matrix",
    )
    translation_candidates = (
        "translation",
        "get_translation",
        "trans",
        "get_trans",
        "t",
    )
    transform_candidates = (
        "transform",
        "get_transform",
        "matrix",
        "get_matrix",
    )

    rotation = None
    for name in rotation_candidates:
        rotation = _value(name)
        if rotation is not None:
            if debug:
                print(f"[compare_projectors] rotation source: {name}")
            break
    translation = None
    for name in translation_candidates:
        translation = _value(name)
        if translation is not None:
            if debug:
                print(f"[compare_projectors] translation source: {name}")
            break
    transform = None
    if rotation is None or translation is None:
        for name in transform_candidates:
            transform = _value(name)
            if transform is not None:
                if debug:
                    print(f"[compare_projectors] transform source: {name}")
                break

    rot_arr = None
    trans_arr = None
    if rotation is not None:
        try:
            rot_arr = np.asarray(rotation, dtype=float)
            if debug:
                print(f"[compare_projectors] rotation raw shape {rot_arr.shape}")
            if rot_arr.size == 9:
                rot_arr = rot_arr.reshape(3, 3)
            else:
                rot_arr = None
        except Exception as exc:
            if debug:
                print(f"[compare_projectors] rotation reshape failed: {exc}")
            rot_arr = None
    if translation is not None:
        try:
            trans_arr = np.asarray(translation, dtype=float)
            if debug:
                print(f"[compare_projectors] translation raw shape {trans_arr.shape}")
            if trans_arr.size == 3:
                trans_arr = trans_arr.reshape(3)
            else:
                trans_arr = None
        except Exception as exc:
            if debug:
                print(f"[compare_projectors] translation reshape failed: {exc}")
            trans_arr = None

    if rot_arr is None or trans_arr is None:
        if transform is not None:
            try:
                tf_arr = np.asarray(transform, dtype=float)
                if debug:
                    print(f"[compare_projectors] transform raw shape {tf_arr.shape}")
                flat = tf_arr.ravel()
                size = flat.size
                if size == 3:
                    if trans_arr is None:
                        trans_arr = flat.reshape(3)
                elif size == 9:
                    mat3 = flat.reshape(3, 3)
                    if rot_arr is None:
                        rot_arr = mat3
                elif size == 12:
                    mat34 = flat.reshape(3, 4)
                    if rot_arr is None:
                        rot_arr = mat34[:, :3]
                    if trans_arr is None:
                        trans_arr = mat34[:, 3]
                elif size == 16:
                    mat4 = flat.reshape(4, 4)
                    if rot_arr is None:
                        rot_arr = mat4[:3, :3]
                    if trans_arr is None:
                        trans_arr = mat4[:3, 3]
            except Exception as exc:
                if debug:
                    print(f"[compare_projectors] transform reshape failed: {exc}")

    if rot_arr is None or trans_arr is None:
        if debug:
            print("[compare_projectors] missing rotation/translation, returning empty extrinsic dict")
        return {}

    matrix = np.eye(4, dtype=float)
    matrix[:3, :3] = rot_arr
    matrix[:3, 3] = trans_arr
    return {
        "rotation": rot_arr.tolist(),
        "translation": trans_arr.tolist(),
        "matrix": matrix.tolist(),
    }


def _collect_random_pixels(depth: np.ndarray, sample_count: int) -> List[Tuple[int, int]]:
    coords = np.argwhere(depth > 0)
    if coords.size == 0:
        return []
    if len(coords) <= sample_count:
        return [(int(c[1]), int(c[0])) for c in coords]
    picks = random.sample(range(len(coords)), sample_count)
    return [(int(coords[i][1]), int(coords[i][0])) for i in picks]


def compare_projectors(sample_count: int, debug_extr: bool) -> None:
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
        extr_dict = _serialize_extrinsic(sdk_extrinsic, debug=debug_extr)
        if not extr_dict and debug_extr:
            print("[compare_projectors] extrinsic dict empty; numeric projector will stay in depth frame")

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
    parser.add_argument(
        "--debug-extr",
        action="store_true",
        help="Print diagnostic information about the extrinsic struct.",
    )
    args = parser.parse_args()
    compare_projectors(args.samples, args.debug_extr)


if __name__ == "__main__":
    main()
