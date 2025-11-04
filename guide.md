## 摄像头
fuser -v /dev/video*   # 查看哪些进程占用摄像头
sudo fuser -vk /dev/video* # 强制杀掉占用摄像头的进程