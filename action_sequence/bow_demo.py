import requests
import time

API_BASE_URL = "http://192.168.10.90:5000/api/extaxis"
ENABLE_URL = f"{API_BASE_URL}/enable"
MOVETO_URL = f"{API_BASE_URL}/moveto"
STATUS_URL = f"{API_BASE_URL}/status"

HEAD_PITCH_MAX = -20.0   # 鞠躬最大俯仰角（度）
HEAD_PITCH_MIN = -5    # 恢复初始俯仰角（度）

# 1. 使能
try:
    resp = requests.post(ENABLE_URL, json={"enable": 1}, timeout=1.0)
    if resp.status_code == 200:
        print("升降电机已使能")
    else:
        print(f"使能失败: {resp.status_code}")
except Exception as e:
    print(f"使能请求异常: {e}")
    exit(1)

# move_cmd = {
#     "pos": [30, 0, -20, -5],
#     "vel": 100,
#     "acc": 100
# }
# try:
#     resp = requests.post(MOVETO_URL, json=move_cmd, timeout=2.0)
#     if resp.status_code == 200:
#         print(f"鞠躬：头部低头到 {HEAD_PITCH_MAX}°")
#     else:
#         print(f"鞠躬动作失败: {resp.status_code}")
# except Exception as e:
#     print(f"鞠躬动作异常: {e}")
#     exit(1)
# 2. 获取当前状态
try:
    resp = requests.get(STATUS_URL, timeout=2.0)
    state = resp.json()
    current_height = state[0]["pos"]
    current_rotation = state[2]["pos"]
    print(f"当前高度: {current_height}, 当前头部旋转: {current_rotation}")
except Exception as e:
    print(f"获取状态异常: {e}")
    exit(1)

# 3. 鞠躬动作：低头
move_cmd = {
    "pos": [100, 0, -20, 0],
    "vel": 100,
    "acc": 100
}
try:
    resp = requests.post(MOVETO_URL, json=move_cmd, timeout=5.0)
    if resp.status_code == 200:
        print(f"鞠躬：头部低头到 {HEAD_PITCH_MAX}°")
    else:
        print(f"鞠躬动作失败: {resp.status_code}")
except Exception as e:
    print(f"鞠躬动作异常: {e}")
    exit(1)

time.sleep(0.5)  # 保持鞠躬动作

# # 4. 恢复抬头
# move_cmd = {
#     "pos": [100, 0, 40, 0],
#     "vel": 100,
#     "acc": 100
# }
# try:
#     resp = requests.post(MOVETO_URL, json=move_cmd, timeout=5.0)
#     if resp.status_code == 200:
#         print(f"恢复：头部抬起到 {HEAD_PITCH_MIN}°")
#     else:
#         print(f"恢复动作失败: {resp.status_code}")
# except Exception as e:
#     print(f"恢复动作异常: {e}")
#     exit(1)


# time.sleep(0.5)  # 保持鞠躬动作

# move_cmd = {
#     "pos": [100, 0, 0, 0],
#     "vel": 100,
#     "acc": 100
# }
# try:
#     resp = requests.post(MOVETO_URL, json=move_cmd, timeout=5.0)
#     if resp.status_code == 200:
#         print(f"恢复：头部抬起到 {HEAD_PITCH_MIN}°")
#     else:
#         print(f"恢复动作失败: {resp.status_code}")
# except Exception as e:
#     print(f"恢复动作异常: {e}")
#     exit(1)
