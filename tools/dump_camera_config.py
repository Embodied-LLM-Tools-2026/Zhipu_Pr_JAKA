#!/usr/bin/env python3
"""
Quick utility to dump intrinsic/extrinsic parameters from an Orbbec depth camera.

The script mirrors the stream setup used in the UI service and produces a YAML
payload compatible with the ZeroGrasp config requirements.

Usage:
    python tools/dump_camera_config.py --output camera_config.yml

Environment variables:
    ORBBEC_SERIAL (optional): restricts capture to a specific device serial.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import yaml

try:
    from pyorbbecsdk import (  # type: ignore
        Config,
        OBSensorType,
        Pipeline,
    )
except Exception as exc:  # noqa: B902
    print(f"[dump_camera_config] pyorbbecsdk not available: {exc}")
    sys.exit(1)


def _video_profile_dict(video_profile) -> Dict[str, Any]:
    intr = video_profile.get_intrinsic()
    result = {
        "width": video_profile.get_width(),
        "height": video_profile.get_height(),
        "fps": video_profile.get_fps(),
        "fx": intr.fx,
        "fy": intr.fy,
        "cx": intr.cx,
        "cy": intr.cy,
        "model": getattr(intr, "distortion_model", None),
        "coeffs": getattr(intr, "distortion_coeffs", None),
    }
    return result


def _matrix_from_extrinsic(extrinsic) -> Optional[list]:
    if extrinsic is None:
        return None
    try:
        rotation = np.array(extrinsic.rotation).reshape(3, 3)
        translation = np.array(extrinsic.translation).reshape(3)
        matrix = np.eye(4, dtype=float)
        matrix[:3, :3] = rotation
        matrix[:3, 3] = translation
        return matrix.tolist()
    except Exception:  # noqa: B902
        return None


def _try_depth_range(depth_profile) -> Dict[str, Optional[float]]:
    result = {"depth_min": None, "depth_max": None}
    try:
        if hasattr(depth_profile, "get_min_distance"):
            result["depth_min"] = float(depth_profile.get_min_distance())
        if hasattr(depth_profile, "get_max_distance"):
            result["depth_max"] = float(depth_profile.get_max_distance())
    except Exception:
        pass
    return result


def _try_depth_scale(depth_frame) -> Optional[float]:
    try:
        if hasattr(depth_frame, "get_device"):
            device = depth_frame.get_device()
            depth_sensor = device.get_sensor(OBSensorType.DEPTH_SENSOR)
            if hasattr(depth_sensor, "get_depth_scale"):
                return float(depth_sensor.get_depth_scale())
    except Exception:
        pass
    return None


def dump_camera_config(output_path: Path, serial_filter: Optional[str]) -> Dict[str, Any]:
    cfg = Config()
    pipeline = Pipeline()

    for sensor_type in (OBSensorType.DEPTH_SENSOR, OBSensorType.COLOR_SENSOR):
        profile_list = pipeline.get_stream_profile_list(sensor_type)
        if profile_list is None:
            raise RuntimeError(f"No stream profile for sensor type {sensor_type}")
        profile = profile_list.get_default_video_stream_profile()
        if profile is None:
            raise RuntimeError(f"Unable to get default profile for {sensor_type}")
        cfg.enable_stream(profile)

    if serial_filter:
        device_list = pipeline.get_connected_devices()
        if device_list:
            for idx in range(device_list.get_count()):
                device = device_list.get_device(idx)
                if device.get_device_info().serial_number == serial_filter:
                    prev_cfg = cfg
                    cfg = Config()
                    cfg.enable_streams_from_device(device, prev_cfg)
                    break

    pipeline.start(cfg)

    try:
        frames = None
        for _ in range(50):
            frames = pipeline.wait_for_frames(1000)
            if frames and frames.get_depth_frame() and frames.get_color_frame():
                break
            time.sleep(0.05)
        if frames is None:
            raise RuntimeError("Unable to acquire synchronized frames")

        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if color_frame is None or depth_frame is None:
            raise RuntimeError("Incomplete frame set received")

        color_profile = color_frame.get_stream_profile().as_video_stream_profile()
        depth_profile = depth_frame.get_stream_profile().as_video_stream_profile()

        color_info = _video_profile_dict(color_profile)
        depth_info = _video_profile_dict(depth_profile)
        extrinsic = depth_profile.get_extrinsic_to(color_profile)

        payload: Dict[str, Any] = {
            "width": color_info["width"],
            "height": color_info["height"],
            "camera_width": color_info["width"],
            "camera_height": color_info["height"],
            "rectified_width": depth_info["width"],
            "rectified_height": depth_info["height"],
            "color_intrinsics": {
                "fx": color_info["fx"],
                "fy": color_info["fy"],
                "cx": color_info["cx"],
                "cy": color_info["cy"],
                "model": color_info["model"],
                "coeffs": color_info["coeffs"],
            },
            "depth_intrinsics": {
                "fx": depth_info["fx"],
                "fy": depth_info["fy"],
                "cx": depth_info["cx"],
                "cy": depth_info["cy"],
                "model": depth_info["model"],
                "coeffs": depth_info["coeffs"],
            },
            "extrinsic_color_to_depth": _matrix_from_extrinsic(extrinsic),
            "timestamp_system_us": int(time.time() * 1e6),
            "depth_scale": _try_depth_scale(depth_frame),
        }

        depth_range = _try_depth_range(depth_profile)
        payload.update(depth_range)

        device = pipeline.get_device()
        if device is not None:
            device_info = device.get_device_info()
            payload["device"] = {
                # "name": device_info.name,
                # "serial_number": device_info.serial_number,
                # "firmware_version": device_info.firmware_version,
                # "hardware_version": device_info.hardware_version,
            }

        output_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        print(f"[dump_camera_config] Wrote configuration to {output_path}")
        print(json.dumps(payload, indent=2, default=str))
        return payload
    finally:
        try:
            pipeline.stop()
        except Exception:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Dump Orbbec camera configuration.")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to the YAML file that will contain the configuration.",
    )
    parser.add_argument(
        "--serial",
        type=str,
        default=None,
        help="Optional device serial number if multiple cameras are connected.",
    )
    args = parser.parse_args()
    dump_camera_config(args.output, args.serial)


if __name__ == "__main__":
    main()
