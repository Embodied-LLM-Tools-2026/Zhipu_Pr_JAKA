import sys
import os
import time
import json
import math
import numpy as np
import cv2
from PIL import Image, ImageDraw

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from voice.perception.observer import VLMObserver, ObservationContext
from voice.perception.localize_target import TargetLocalizer
from voice.control.task_structures import ObservationPhase

# Mock Navigate class
class MockNavigate:
    def get_current_pose(self):
        # Return a dummy pose (e.g., robot at origin)
        return {"x": 0.0, "y": 0.0, "theta": 0.0}

def transform_camera_to_robot(cam_point_mm):
    """
    Convert camera-frame coordinates (mm) to robot base coordinates (mm).
    Copied from SkillExecutor.transform_camera_to_robot
    """
    cam_x, cam_y, cam_z = cam_point_mm
    x_ab = cam_y + 50.0
    y_ab = -cam_x + 180.0
    z_ab = cam_z
    return np.array([x_ab, y_ab, z_ab], dtype=float)

def _transform_with_matrix(cam_point_mm):
    """Reference transform using 4x4 homogeneous matrix (camera->robot)."""
    vec = np.array(list(cam_point_mm) + [1.0], dtype=float).reshape(4, 1)
    # Translation matches transform_camera_to_robot: x += 50, y += 180 after 90° rotation.
    T_mat = np.array(
        [
            [0, 1, 0, 50],
            [-1, 0, 0, 180],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        dtype=float,
    )
    jaka_vec = T_mat @ vec
    return jaka_vec[:3].ravel()

def _apply_pose_to_robot_point(robot_point_mm, pose):
    """
    Convert robot-frame coords (mm) to world frame using navigator pose (m, rad).
    Mirrors the snippet provided for testing consistency.
    """
    X_OA = pose["x"] * 1000.0
    Y_OA = pose["y"] * 1000.0
    theta_OA = pose["theta"]
    X_AB, Y_AB, Z_AB = robot_point_mm
    X_OB = X_OA + (X_AB * math.cos(theta_OA) - Y_AB * math.sin(theta_OA))
    Y_OB = Y_OA + (X_AB * math.sin(theta_OA) + Y_AB * math.cos(theta_OA))
    return np.array([X_OB, Y_OB, Z_AB], dtype=float)

def test_transform_camera_to_robot_matches_homogeneous():
    """Ensure the explicit matrix transform matches the helper function."""
    samples = [
        [0.0, 0.0, 0.0],
        [120.0, -30.0, 500.0],
        [-80.0, 140.0, 250.0],
    ]
    for cam_point in samples:
        direct = transform_camera_to_robot(cam_point)
        matrix = _transform_with_matrix(cam_point)
        assert np.allclose(direct, matrix), f"Mismatch for {cam_point}: {direct} vs {matrix}"

def test_transform_world_frame_consistency():
    """Check world-frame results stay consistent across both transform methods."""
    cam_point = [-60.0, 90.0, 300.0]
    pose = {"x": 1.2, "y": -0.4, "theta": math.radians(30)}
    direct_robot = transform_camera_to_robot(cam_point)
    matrix_robot = _transform_with_matrix(cam_point)
    direct_world = _apply_pose_to_robot_point(direct_robot, pose)
    matrix_world = _apply_pose_to_robot_point(matrix_robot, pose)
    assert np.allclose(direct_world, matrix_world), f"World frame mismatch: {direct_world} vs {matrix_world}"

def main():
    target_name = input("请输入目标物体名称 (默认: bottle): ") or "bottle"
    
    print(f"Initializing VLMObserver and TargetLocalizer for target: {target_name}...")
    
    # Initialize components
    observer = VLMObserver(cam_name="front")
    localizer = TargetLocalizer()
    navigator = MockNavigate()
    
    # Context for observation
    context = ObservationContext(step=1, max_steps=1)
    
    print("Capturing observation...")
    # 1. Observe scene to get bbox
    observation, payload = observer.observe(
        target_name=target_name,
        phase=ObservationPhase.SEARCH,
        context=context,
        navigator=navigator,
        force_vlm=True 
    )
    
    if not observation.found:
        print(f"Target '{target_name}' not found.")
        return

    print(f"Target found! BBox: {observation.bbox}")
    print(f"Confidence: {observation.confidence}")
    
    # 2. Localize target to get 3D coordinates
    print("Localizing target...")
    # We need rgb_frame for localization if we want to use surface mask, 
    # but localize_from_service can handle it if we pass the frame.
    # Let's load the frame from the observation path.
    rgb_frame = None
    if observation.original_image_path and os.path.exists(observation.original_image_path):
        rgb_frame = cv2.imread(observation.original_image_path)
    
    depth_info = localizer.localize_from_service(
        bbox=observation.bbox,
        rgb_frame=rgb_frame,
        include_transform=True
    )
    
    if not depth_info:
        print("Depth localization failed.")
        return

    obj_center_3d = depth_info.get("obj_center_3d")
    if not obj_center_3d:
        print("Could not determine 3D center.")
        return
        
    print(f"Camera Frame Coordinates (mm): {obj_center_3d}")
    
    # 3. Transform to Robot Frame
    cam_point_mm = np.array(obj_center_3d, dtype=float)
    
    robot_point_mm = transform_camera_to_robot(cam_point_mm)
    print(f"Robot Frame Coordinates (mm): {robot_point_mm}")

    # Consistency check: helper vs homogeneous transform and world projection.
    matrix_robot_mm = _transform_with_matrix(cam_point_mm)
    robot_match = np.allclose(robot_point_mm, matrix_robot_mm, atol=1e-6)
    print(f"[Check] Robot frame (helper): {robot_point_mm}")
    print(f"[Check] Robot frame (matrix): {matrix_robot_mm}")
    print(f"[Check] Robot frame consistency: {robot_match}")

    pose = navigator.get_current_pose()
    world_helper_mm = _apply_pose_to_robot_point(robot_point_mm, pose)
    world_matrix_mm = _apply_pose_to_robot_point(matrix_robot_mm, pose)
    world_match = np.allclose(world_helper_mm, world_matrix_mm, atol=1e-6)
    print(f"[Check] World frame (helper, mm): {world_helper_mm}")
    print(f"[Check] World frame (matrix, mm): {world_matrix_mm}")
    print(f"[Check] World frame consistency: {world_match}")
    
    # 4. Draw bbox and save image
    if observation.original_image_path:
        image_path = observation.original_image_path
        print(f"Processing image: {image_path}")
        
        try:
            with Image.open(image_path) as img:
                draw = ImageDraw.Draw(img)
                x1, y1, x2, y2 = observation.bbox
                draw.rectangle([(x1, y1), (x2, y2)], outline="red", width=5)
                
                # Draw text
                text = f"{target_name}: {observation.confidence:.2f}\nCam: {obj_center_3d}\nRobot: {robot_point_mm}"
                draw.text((x1, y1 - 10), text, fill="red")
                
                output_path = os.path.join(os.path.dirname(image_path), "debug_localize_target.jpg")
                img.save(output_path)
                print(f"Saved debug image to: {output_path}")
                
        except Exception as e:
            print(f"Failed to draw image: {e}")

if __name__ == "__main__":
    main()
