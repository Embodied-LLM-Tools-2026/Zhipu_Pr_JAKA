import base64, io, json, numpy as np, requests
from PIL import Image

def to_base64_image(path):
    buf = io.BytesIO()
    Image.open(path).convert("RGB").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

def to_base64_points(path):
    arr = np.load(path).astype(np.float32)
    buf = io.BytesIO()
    np.save(buf, arr)
    return base64.b64encode(buf.getvalue()).decode()

payload = {
    "rgb": to_base64_image("sample_rgb.png"),
    "points3d": to_base64_points("sample_points.npy"),
    "bbox": {"x_min": 420, "x_max": 560, "y_min": 180, "y_max": 380},
    "camera": {
        "width": 1280, "height": 720,
        "camera_width": 1280, "camera_height": 720,
        "rectified_width": 1280, "rectified_height": 800,
        "color_intrinsics": {"fx": 609.93835, "fy": 609.43658, "cx": 633.95282, "cy": 351.65677},
        "depth_intrinsics": {"fx": 614.6842, "fy": 614.6842, "cx": 636.27179, "cy": 400.97281}
    },
    "frame_id": "can-001",
    "object_id": 0,
    "grasp_limit": 1
}

resp = requests.post("http://127.0.0.1:8000/predict", json=payload)
resp.raise_for_status()
print(json.dumps(resp.json(), indent=2))