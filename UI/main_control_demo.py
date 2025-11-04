
"""
Minimal "main control" demo showing how to call the server:
- Start a VLM call (server captures + resizes to 1000x750)
- Pretend to run VLM and post back a couple of boxes
"""
import time, requests, random

BASE = "http://127.0.0.1:8000"

def start_call(prompt="find coke", cam="front"):
    r = requests.post(f"{BASE}/api/vlm/start_call", json={
        "prompt": prompt,
        "cam": cam,
        "return_base64": False  # set True if you want to get the JPEG bytes to feed into VLM directly
    }, timeout=10)
    r.raise_for_status()
    return r.json()

def finish_call(call_id, boxes):
    r = requests.post(f"{BASE}/api/vlm/finish_call", json={
        "call_id": call_id,
        "boxes": boxes
    }, timeout=10)
    r.raise_for_status()
    return r.json()

def set_status(current):
    requests.post(f"{BASE}/api/status", json={"current": current}, timeout=5)

def set_telemetry(yaw, pitch, roll, vlin, vang, action):
    requests.post(f"{BASE}/api/telemetry", json={
        "orientation": {"yaw": yaw, "pitch": pitch, "roll": roll},
        "velocity": {"linear": vlin, "angular": vang},
        "chassis_action": action
    }, timeout=5)

if __name__ == "__main__":
    for step in ["listening", "searching", "waiting_api", "navigating", "grasping"]:
        set_status(step)
        set_telemetry(yaw=random.random()*360, pitch=0.0, roll=0.0,
                      vlin=random.uniform(-0.2, 0.4), vang=random.uniform(-0.5, 0.5),
                      action=random.choice(["idle","forward","turn_left","turn_right","stop"]))
        time.sleep(1.0)

    # Start a VLM call: server grabs current frame and resizes it to 1000x750
    resp = start_call(prompt="find coke", cam="front")
    print("Payload URL:", resp["payload_url"], "Size:", resp["payload_size"])
    call_id = resp["call_id"]

    # ... here you would call your real VLM with the resized image ...
    # For demo, we just return two random boxes inside 1000x750
    boxes = []
    for _ in range(2):
        x1 = random.randint(0, 800)
        y1 = random.randint(0, 600)
        x2 = min(999, x1 + random.randint(60, 200))
        y2 = min(749, y1 + random.randint(40, 160))
        boxes.append([x1,y1,x2,y2])

    fin = finish_call(call_id, boxes)
    print("Finish response:", fin)