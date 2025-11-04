import requests

# 推送一条info日志
requests.post("http://127.0.0.1:8000/api/task/log", json={
    "message": "开始采集图片...",
    "level": "info"
})

# 推送一条success日志
requests.post("http://127.0.0.1:8000/api/task/log", json={
    "message": "图片采集成功",
    "level": "success"
})

# 推送一条warning日志
requests.post("http://127.0.0.1:8000/api/task/log", json={
    "message": "VLM返回坐标范围异常",
    "level": "warning"
})

# 推送一条error日志
requests.post("http://127.0.0.1:8000/api/task/log", json={
    "message": "图片上传失败",
    "level": "error"
})