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
