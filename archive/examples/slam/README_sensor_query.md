# 传感器数据查询工具

这个脚本用于查询机器人的IMU数据和激光点云数据。

## 功能特性

1. **IMU数据查询**（消息类型1014）
   - 偏航角、滚转角、俯仰角
   - 加速度计数据（X、Y、Z轴）
   - 陀螺仪数据（X、Y、Z轴）
   - 四元数数据
   - 时间戳信息

2. **激光点云数据查询**（消息类型1009）
   - 激光传感器设备信息
   - 光束数据（角度、距离、有效性等）
   - 安装位置信息
   - 统计信息

3. **其他功能**
   - 数据保存到JSON文件
   - 连续查询模式
   - 详细数据显示
   - 命令行参数控制

## 使用方法

### 基本用法

```bash
# 查询所有传感器数据
python sensor_data_query.py

# 指定AGV服务器IP
python sensor_data_query.py --ip 192.168.1.51

# 保存数据到文件
python sensor_data_query.py --save

# 显示详细的激光数据
python sensor_data_query.py --detailed-laser
```

### 单独查询特定传感器

```bash
# 仅查询IMU数据
python sensor_data_query.py --imu-only

# 仅查询激光数据
python sensor_data_query.py --laser-only
```

### 连续查询模式

```bash
# 每秒查询一次，持续10秒
python sensor_data_query.py --continuous --interval 1.0 --duration 10

# 每2秒查询一次，无限制（按Ctrl+C停止）
python sensor_data_query.py --continuous --interval 2.0

# 连续查询并保存数据
python sensor_data_query.py --continuous --save --interval 0.5
```

### 参数说明

- `--ip`: AGV服务器IP地址（默认：192.168.192.5）
- `--save`: 保存数据到JSON文件
- `--detailed-laser`: 显示详细的激光光束数据
- `--continuous`: 启用连续查询模式
- `--interval`: 连续查询间隔（秒，默认：1.0）
- `--duration`: 连续查询持续时间（秒，不指定则无限制）
- `--imu-only`: 仅查询IMU数据
- `--laser-only`: 仅查询激光数据

## 数据格式

### IMU数据示例
```json
{
  "yaw": -3.128697633743291,
  "roll": 0,
  "pitch": 0,
  "acc_x": 0,
  "acc_y": 0,
  "acc_z": 0,
  "rot_x": 0,
  "rot_y": 0,
  "rot_z": 0,
  "qx": 0,
  "qy": 0,
  "qz": 0,
  "qw": 0,
  "imu_header": {
    "data_nsec": "16704707855595",
    "frame_id": "/imu",
    "pub_nsec": "16704707855637",
    "seq": "0"
  }
}
```

### 激光数据示例
```json
{
  "lasers": [
    {
      "beams": [
        {
          "angle": -90,
          "dist": 2.762,
          "rssi": 44.5273,
          "valid": true,
          "is_obstacle": false,
          "is_virtual": false
        }
      ],
      "device_info": {
        "device_name": "laser",
        "max_angle": 90,
        "max_range": 30,
        "min_angle": -90,
        "scan_freq": 30
      },
      "install_info": {
        "x": 0.3386,
        "y": 0,
        "z": 0.229,
        "yaw": 0,
        "upside": true
      }
    }
  ]
}
```

## 作为Python模块使用

```python
from sensor_data_query import SensorDataQuery

# 使用上下文管理器
with SensorDataQuery(ip='192.168.192.5') as sensor:
    # 查询IMU数据
    imu_data = sensor.get_imu_data()
    
    # 查询激光数据
    laser_data = sensor.get_laser_data()
    
    # 显示数据
    sensor.display_imu_data(imu_data)
    sensor.display_laser_data(laser_data)
    
    # 保存数据
    sensor.save_data_to_file(imu_data, laser_data)
```

## 注意事项

1. 确保AGV服务器已启动并可访问
2. 检查网络连接和IP地址设置
3. 某些数据字段可能为可选，实际数据可能与示例不同
4. 连续查询时注意数据量，避免磁盘空间不足