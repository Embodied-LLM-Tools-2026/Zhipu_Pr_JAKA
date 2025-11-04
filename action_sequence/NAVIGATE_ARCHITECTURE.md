# Navigate 类架构升级说明

## 🎯 设计目标

**问题**：AGV status 包含丰富信息（位置、导航状态等），之前重复查询浪费资源
**方案**：改用 **socket 长连接 + 一个后台循环**，统一获取并更新多个全局变量

## 📊 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                    AGV 硬件 (192.168.10.10:31001)           │
└─────────────────────────────────────────────────────────────┘
                              ▲
                              │ [长连接 socket]
                              │ [每 0.1s 查询一次]
                              ▼
┌─────────────────────────────────────────────────────────────┐
│         Navigate._status_monitoring_loop()                  │
│    (后台线程，维护 socket 长连接，定期查询状态)             │
└─────────────────────────────────────────────────────────────┘
            │
            ├─ _update_global_state()
            │   ├─ current_pose = {theta, x, y}
            │   ├─ navigation_state = {move_status, is_navigating}
            │   └─ last_status_response = {完整响应}
            │
            ▼
┌─────────────────────────────────────────────────────────────┐
│                    全局状态变量                              │
│  • current_pose: 实时位置 {theta, x, y}                    │
│  • navigation_state: 导航状态 {move_status, is_navigating} │
│  • last_status_response: 完整响应（扩展字段）              │
└─────────────────────────────────────────────────────────────┘
            ▲
            │ [其他模块直接读取]
            │ [无需重复查询]
            │
    ┌───────┴───────┬──────────────┬─────────────────┐
    │               │              │                 │
    ▼               ▼              ▼                 ▼
robot_ui_demo.py  main_hand.py  VLM.py          其他模块
```

## 🔧 核心改动

### 1. 全局状态变量初始化（__init__）
```python
# 位置信息
self.current_pose = {"theta": 0.0, "x": 0.0, "y": 0.0}

# 导航状态
self.navigation_state = {
    "move_status": None,      # "success", "running", "failed"
    "is_navigating": False,   # 是否正在导航
    "last_target": None
}

# 完整响应（用于扩展字段）
self.last_status_response = {}

# Socket 长连接
self._socket = None
```

### 2. 后台监控循环（_status_monitoring_loop）
```
建立 socket 长连接 →  定期发送查询  →  接收解析 →  更新全局变量 →  循环
                    │
                    └─ 连接断开时自动重连
```

**关键特性**：
- ✅ 长连接（一次 connect，多次 query）
- ✅ 自动重连（连接失败后 exponential backoff）
- ✅ 异常处理（socket 超时、连接拒绝）
- ✅ 线程安全（使用 threading.Lock）

### 3. 全局状态更新（_update_global_state）
```python
def _update_global_state(self, status_data):
    with self._lock:
        # 保存完整响应
        self.last_status_response = status_data
        
        # 解析位置信息
        move_status = status_data.get('result', {}).get('move_status')
        if move_status and len(move_status) >= 3:
            self.current_pose["theta"] = float(move_status[0])
            self.current_pose["x"] = float(move_status[1])
            self.current_pose["y"] = float(move_status[2])
        
        # 解析导航状态
        nav_status = status_data.get('result', {}).get('navigation_status')
        if nav_status:
            self.navigation_state["move_status"] = nav_status
            self.navigation_state["is_navigating"] = (nav_status == "running")
```

## 📚 API 接口

### 启动/停止监控
```python
navigator = Navigate()

# 启动后台监控（0.1s 轮询间隔）
navigator.start_pose_monitoring(poll_interval=0.1)

# 停止监控（关闭 socket）
navigator.stop_pose_monitoring()
```

### 查询全局状态（无需网络查询）
```python
# 获取位置
pose = navigator.get_current_pose()
# → {"theta": 45.23, "x": 10.5, "y": 20.3}

# 获取导航状态
state = navigator.get_navigation_state()
# → {"move_status": "running", "is_navigating": True, ...}

# 检查是否正在导航
if navigator.is_navigating():
    print("正在导航中...")

# 获取完整响应（用于扩展字段）
resp = navigator.get_last_status_response()
```

### 导航指令
```python
# 发送导航目标
navigator.navigate_to('bar')

# 等待导航完成（使用全局状态变量，高效）
navigator.wait_until_navigation_complete(timeout=60)
```

## 🚀 使用示例

### 场景1：实时监控（FastAPI 前端）
```python
# robot_ui_demo.py 中
from action_sequence.navigate import Navigate

NAVIGATOR = Navigate()

@APP.on_event("startup")
def startup():
    NAVIGATOR.start_pose_monitoring(poll_interval=0.1)

@APP.on_event("shutdown")
def shutdown():
    NAVIGATOR.stop_pose_monitoring()

@APP.get("/api/agv/pose")
def api_agv_pose():
    pose = NAVIGATOR.get_current_pose()
    return {"status": "ok", "pose": pose}
```

### 场景2：导航流程（VLM 主逻辑）
```python
# main_hand.py 中
navigator.start_pose_monitoring(0.1)

# 发送导航指令
navigator.navigate_to('target_location')

# 等待完成（从全局状态读取，不重复查询）
if navigator.wait_until_navigation_complete(timeout=60):
    print("导航成功！")
else:
    print("导航失败！")

navigator.stop_pose_monitoring()
```

### 场景3：多模块共享状态
```python
# 模块A：获取位置
pose = navigator.get_current_pose()
log_position(pose)

# 模块B：检查导航状态
if navigator.is_navigating():
    wait_for_navigation()

# 所有查询都来自同一个后台循环的全局变量
# 无需重复连接和查询
```

## 📈 性能对比

| 指标 | 旧方案（HTTP 重复查询） | 新方案（Socket 长连接） |
|------|----------------------|----------------------|
| 连接方式 | 每次查询建立新连接 | 一条长连接，重复使用 |
| 轮询方式 | HTTP 请求 | Socket 直接收发 |
| 效率 | ❌ 低（连接开销大） | ✅ 高（连接复用） |
| 资源占用 | ❌ 高（频繁建立断开） | ✅ 低（单一长连接） |
| 扩展性 | ❌ 多个模块重复查询 | ✅ 单一数据源，多模块读取 |
| 数据同步 | ❌ 各自查询，可能不同步 | ✅ 统一后台循环，保证同步 |

## ⚠️ 注意事项

### 1. 线程安全
所有全局状态访问都通过 `threading.Lock` 保护：
```python
with self._lock:
    data = self.current_pose.copy()  # ✅ 正确
```

### 2. 重连机制
连接失败会自动重试，最多 3 次后等待 2 倍轮询间隔后重试。

### 3. Socket 关闭
确保程序退出时调用 `stop_pose_monitoring()`，否则 socket 不会关闭。

### 4. AGV 响应格式
期望的响应格式：
```json
{
  "result": {
    "move_status": [theta, x, y],
    "navigation_status": "running|success|failed|..."
  }
}
```

## 🔍 调试

### 启用日志
后台循环会自动打印日志：
```
✅ 已连接到 AGV
📍 AGV 位置: θ=45.00°, x=10.50m, y=20.30m
🔄 导航状态: running
⚠️ Socket 超时
🔌 尝试连接 AGV (192.168.10.10:31001)...
```

### 测试脚本
运行 `test_navigate_socket.py` 查看实时更新：
```bash
python test_navigate_socket.py
```

---

**更新时间**：2025-10-22  
**版本**：2.0（Socket 长连接架构）
