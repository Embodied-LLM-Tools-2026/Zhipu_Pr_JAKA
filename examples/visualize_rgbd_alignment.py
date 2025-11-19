#!/usr/bin/env python3
"""
Capture a single RGB + depth frame and dump visualizations that help verify alignment.

This script reuses the Orbbec capture pipeline from the ZeroGrasp VLM demo and writes:
  - color.png: raw BGR frame from the RGB camera
  - depth_colormap.png: depth visualized with a jet colormap
  - overlay.png: color and depth colormap blended together
  - edge_overlay.png: Canny edges of color (red) vs. depth (green) for easier misalignment spotting

Usage:
    python examples/visualize_rgbd_alignment.py [--serial SERIAL] [--output-dir DIR] [--disable-align]
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import zerograsp_vlm_demo as zg_demo  # type: ignore

OrbbecRGBDCapture = zg_demo.OrbbecRGBDCapture


def _colorize_depth(depth: np.ndarray) -> np.ndarray:
    """Convert raw depth (uint16) to a jet colormap for visualization."""
    depth_u8 = np.zeros_like(depth, dtype=np.uint8)
    valid = depth > 0
    if np.any(valid):
        min_val = int(depth[valid].min())
        max_val = int(depth[valid].max())
        if max_val == min_val:
            max_val = min_val + 1
        depth_u8 = cv2.normalize(
            depth,
            None,
            alpha=0,
            beta=255,
            norm_type=cv2.NORM_MINMAX,
            dtype=cv2.CV_8U,
        )
    depth_color = cv2.applyColorMap(depth_u8, cv2.COLORMAP_JET)
    return depth_color


def _blend_images(color_bgr: np.ndarray, depth_color: np.ndarray) -> np.ndarray:
    """Resize depth visualization to RGB size if needed and blend the two."""
    h, w = color_bgr.shape[:2]
    depth_vis = depth_color
    if depth_vis.shape[:2] != (h, w):
        depth_vis = cv2.resize(depth_vis, (w, h), interpolation=cv2.INTER_NEAREST)
    overlay = cv2.addWeighted(color_bgr, 0.6, depth_vis, 0.4, 0.0)
    return overlay


def _edge_overlay(color_bgr: np.ndarray, depth_color: np.ndarray) -> np.ndarray:
    """Create a 2-channel edge overlay: color edges in red, depth edges in green."""
    color_gray = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2GRAY)
    color_edges = cv2.Canny(color_gray, 100, 200)

    depth_gray = cv2.cvtColor(depth_color, cv2.COLOR_BGR2GRAY)
    depth_edges = cv2.Canny(depth_gray, 50, 150)
    if depth_edges.shape != color_edges.shape:
        depth_edges = cv2.resize(depth_edges, (color_edges.shape[1], color_edges.shape[0]), interpolation=cv2.INTER_NEAREST)

    edge_vis = np.zeros_like(color_bgr)
    edge_vis[..., 2] = color_edges  # Red channel
    edge_vis[..., 1] = depth_edges  # Green channel
    # Highlight overlapping edges in yellow for easier inspection
    overlap = cv2.bitwise_and(color_edges, depth_edges)
    edge_vis[..., 0] = overlap
    return edge_vis


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize RGB/Depth alignment for Orbbec capture")
    parser.add_argument("--serial", default=os.getenv("ORBBEC_SERIAL"), help="Optional: lock to a specific Orbbec device serial")
    parser.add_argument("--disable-align", action="store_true", help="Disable depth-to-color alignment (enabled by default)")
    parser.add_argument("--output-dir", help="Directory to save visualization images; defaults to a temp dir")
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else Path(tempfile.mkdtemp(prefix="rgbd_alignment_"))
    output_dir.mkdir(parents=True, exist_ok=True)

    align_enabled = not args.disable_align
    print(f"[RGBD Alignment] Saving outputs to {output_dir}")
    print(f"[RGBD Alignment] Depth-to-color align filter: {'ON' if align_enabled else 'OFF'}")

    capture = OrbbecRGBDCapture(serial_filter=args.serial, align_to_color=align_enabled)
    try:
        color_bgr, depth_snapshot = capture.capture()
    finally:
        capture.stop()

    target_size = (1280, 800)  # width, height for RGB-space visualizations

    depth = depth_snapshot.depth
    depth_color = _colorize_depth(depth)
    color_resized = color_bgr
    if (color_bgr.shape[1], color_bgr.shape[0]) != target_size:
        color_resized = cv2.resize(color_bgr, target_size, interpolation=cv2.INTER_LINEAR)
    overlay = _blend_images(color_resized, depth_color)
    edges = _edge_overlay(color_resized, depth_color)

    color_path = output_dir / "color.png"
    depth_path = output_dir / "depth_colormap.png"
    overlay_path = output_dir / "overlay.png"
    edges_path = output_dir / "edge_overlay.png"

    cv2.imwrite(str(color_path), color_resized)
    cv2.imwrite(str(depth_path), depth_color)
    cv2.imwrite(str(overlay_path), overlay)
    cv2.imwrite(str(edges_path), edges)

    print(f"[RGBD Alignment] Saved color frame: {color_path}")
    print(f"[RGBD Alignment] Saved depth colormap: {depth_path}")
    print(f"[RGBD Alignment] Saved color-depth overlay: {overlay_path}")
    print(f"[RGBD Alignment] Saved edge overlay: {edges_path}")


if __name__ == "__main__":
    main()
