import cv2
import numpy as np



print(cv2.__version__)

# 打开相机
cap = cv2.VideoCapture(0)  # 使用默认相机，如果有多个相机可以尝试1,2等

# 检查相机是否成功打开
if not cap.isOpened():
    print("Cannot open camera")
    exit()

# 指定字典类型
dict_type = cv2.aruco.DICT_6X6_1000

# 加载预定义字典
dictionary = cv2.aruco.getPredefinedDictionary(dict_type)

# 创建Aruco参数
aruco_params = cv2.aruco.DetectorParameters()
aruco_params.adaptiveThreshWinSizeMin = 3
aruco_params.adaptiveThreshWinSizeMax = 80
aruco_params.adaptiveThreshWinSizeStep = 10
aruco_params.adaptiveThreshConstant = 7
aruco_params.minMarkerPerimeterRate = 0.1
aruco_params.maxMarkerPerimeterRate = 10.0
aruco_params.polygonalApproxAccuracyRate = 0.1
aruco_params.minCornerDistanceRate = 0.05
aruco_params.minDistanceToBorder = 0

# 创建Aruco检测器
detector = cv2.aruco.ArucoDetector(dictionary, aruco_params)

# 添加相机参数（实际校准值）
# camera_matrix = np.array([[547.5700824698947, 0.0, 314.70719414532533], 
#                           [0.0, 546.8945958211962, 242.9467184746241], 
#                           [0.0, 0.0, 1.0]], dtype=np.float32)
import json

# 从camera_new.json读取相机参数
with open('camera_new.json', 'r', encoding='utf-8') as f:
    camera_data = json.load(f)
    camera_matrix = np.array(camera_data["camera"], dtype=np.float32)
dist_coeffs = np.zeros(5, dtype=np.float32)  # 畸变系数

# ArUco标记的实际大小（米），用户提供的值
marker_length = 0.1  # 实际大小：100mm x 100mm

# 添加帧计数器和位置数据收集
frame_count = 0
position_data = {}  # 存储每个marker ID的位置数据
start_frame = 50
end_frame = 150

print("Press 'q' to quit")

while True:
    # 读取一帧
    ret, frame = cap.read()
    if not ret:
        print("Cannot receive frame")
        break
    
    frame_count += 1
    
    # 转换为灰度图
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 检测Aruco markers
    corners, ids, rejected = detector.detectMarkers(gray)
    
    # 在左上角显示检测信息
    info_text = f"Frame: {frame_count}, Detected: {len(corners) if ids is not None else 0} markers"
    cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    if ids is not None:
        cv2.putText(frame, f"Rejected: {len(rejected)} candidates", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # 绘制检测到的markers
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        
        # 处理检测到的markers
        detected_markers = []
        for i in range(len(ids)):
            c = corners[i][0]
            marker_id = ids[i][0]
            
            # 计算marker的中心
            cx = int((c[0][0] + c[2][0]) / 2)
            cy = int((c[0][1] + c[2][1]) / 2)
            
            # 计算marker的旋转角度
            dx = c[0][0] - c[1][0]
            dy = c[0][1] - c[1][1]
            angle = np.arctan2(dy, dx)
            angle = np.degrees(angle)
            
            # 计算marker的尺寸
            size = np.sqrt(dx * dx + dy * dy)
            
            # 存储marker信息
            detected_markers.append({
                'id': marker_id,
                'center': (cx, cy),
                'angle': angle,
                'size': size
            })
            
            # 在marker上显示信息
            cv2.putText(frame, f"ID: {marker_id}", (int(c[0][0]), int(c[0][1]) - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(frame, f"Angle: {angle:.1f}°", (int(c[0][0]), int(c[0][1]) + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(frame, f"Size: {size:.1f}", (int(c[0][0]), int(c[0][1]) + 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # 绘制中心点和方向线
            cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
            cv2.line(frame, (cx, cy), (int(c[0][0]), int(c[0][1])), (255, 0, 0), 2)
            cv2.line(frame, (cx, cy), (int(c[1][0]), int(c[1][1])), (0, 255, 0), 2)
        
        # 定义ArUco标记的3D对象点（假设标记是平面的，Z=0）
        half_len = marker_length / 2.0
        obj_points = np.array([
            [-half_len, half_len, 0],
            [half_len, half_len, 0],
            [half_len, -half_len, 0],
            [-half_len, -half_len, 0]
        ], dtype=np.float32)

        rvecs = []
        tvecs = []
        for i in range(len(ids)):
            img_points = corners[i][0]  # 当前标记的四个角点
            success, rvec, tvec = cv2.solvePnP(obj_points, img_points, camera_matrix, dist_coeffs)
            if success:
                rvecs.append(rvec)
                tvecs.append(tvec)
                
                # 在第50-150帧之间收集位置数据
                if start_frame <= frame_count <= end_frame:
                    marker_id = ids[i][0]
                    if marker_id not in position_data:
                        position_data[marker_id] = []
                    position_data[marker_id].append({
                        'frame': frame_count,
                        'position': (tvec[0][0], tvec[1][0], tvec[2][0]),
                        'rotation': (rvec[0][0], rvec[1][0], rvec[2][0])
                    })
            else:
                print(f"位姿估计失败 for marker ID: {ids[i][0]}")
                continue

        # 为每个标记绘制坐标轴并显示位姿
        for i in range(len(rvecs)):
            # 绘制坐标轴
            cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvecs[i], tvecs[i], marker_length * 0.5)
            
            # 计算并显示位姿信息
            pos_text = f"Pos: {tvecs[i][0][0]:.5f}, {tvecs[i][1][0]:.5f}, {tvecs[i][2][0]:.5f}"
            rot_text = f"Rot: {rvecs[i][0][0]:.5f}, {rvecs[i][1][0]:.5f}, {rvecs[i][2][0]:.5f}"
            # print(f"{i},pos_text: {pos_text}")
            # 在图像上显示（调整位置以避免重叠）
            cv2.putText(frame, pos_text, (int(corners[i][0][0][0]), int(corners[i][0][0][1]) + 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
            cv2.putText(frame, rot_text, (int(corners[i][0][0][0]), int(corners[i][0][0][1]) + 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
        
        # 在右上角显示详细信息
        if len(detected_markers) > 0:
            # 获取图像尺寸
            height, width = frame.shape[:2]
            
            # 显示第一个marker的详细信息
            marker1 = detected_markers[0]
            right_x = width - 300
            y_start = 30
            line_height = 25
            
            # cv2.putText(frame, f"Marker 1 (ID: {marker1['id']}):", (right_x, y_start), 
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            # cv2.putText(frame, f"  Center: ({marker1['center'][0]}, {marker1['center'][1]})", 
            #             (right_x, y_start + line_height), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            # cv2.putText(frame, f"  Angle: {marker1['angle']:.1f}°", 
            #             (right_x, y_start + line_height * 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            # cv2.putText(frame, f"  Size: {marker1['size']:.1f}", 
            #             (right_x, y_start + line_height * 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            # 如果有第二个marker，显示其信息
            if len(detected_markers) > 1:
                marker2 = detected_markers[1]
                y_start2 = y_start + line_height * 5
                
                # cv2.putText(frame, f"Marker 2 (ID: {marker2['id']}):", (right_x, y_start2), 
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                # cv2.putText(frame, f"  Center: ({marker2['center'][0]}, {marker2['center'][1]})", 
                #             (right_x, y_start2 + line_height), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                # cv2.putText(frame, f"  Angle: {marker2['angle']:.1f}°", 
                #             (right_x, y_start2 + line_height * 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                # cv2.putText(frame, f"  Size: {marker2['size']:.1f}", 
                #             (right_x, y_start2 + line_height * 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    
    # 显示帧
    cv2.imshow("Aruco Real-time Detection", frame)
    if frame_count > end_frame+10:
        break
    # 检查按键
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()

# 计算并显示第50-150帧的位置平均值
print("\n" + "="*50)
print("第50帧到第150帧的ArUco码位置平均值:")
print("="*50)

# 准备JSON数据
json_data = {
    "collection_info": {
        "start_frame": start_frame,
        "end_frame": end_frame,
        "total_frames": end_frame - start_frame + 1
    },
    "markers": {}
}

if position_data:
    for marker_id, data_list in position_data.items():
        if data_list:
            # 计算位置平均值
            positions = np.array([data['position'] for data in data_list])
            rotations = np.array([data['rotation'] for data in data_list])
            
            avg_position = np.mean(positions, axis=0)
            avg_rotation = np.mean(rotations, axis=0)
            
            # 计算标准差
            pos_std = np.std(positions, axis=0)
            rot_std = np.std(rotations, axis=0)
            
            # 存储到JSON数据中（只保存平均值）
            json_data["markers"][str(marker_id)] = {
                "data_frames": len(data_list),
                "average_position": {
                    "x": float(avg_position[0]),
                    "y": float(avg_position[1]),
                    "z": float(avg_position[2])
                },
                "average_rotation": {
                    "rx": float(avg_rotation[0]),
                    "ry": float(avg_rotation[1]),
                    "rz": float(avg_rotation[2])
                }
            }
            
            # 同时打印到控制台（可选）
            print(f"\nMarker ID {marker_id}:")
            print(f"  数据帧数: {len(data_list)}")
            print(f"  平均位置 (X, Y, Z): ({avg_position[0]:.5f}, {avg_position[1]:.5f}, {avg_position[2]:.5f})")
            print(f"  平均旋转 (Rx, Ry, Rz): ({avg_rotation[0]:.5f}, {avg_rotation[1]:.5f}, {avg_rotation[2]:.5f})")
            print(f"  位置标准差: ({pos_std[0]:.5f}, {pos_std[1]:.5f}, {pos_std[2]:.5f})")
            print(f"  旋转标准差: ({rot_std[0]:.5f}, {rot_std[1]:.5f}, {rot_std[2]:.5f})")
else:
    print("在第50-150帧之间没有检测到ArUco码")
    json_data["error"] = "在第50-150帧之间没有检测到ArUco码"

# 写入JSON文件
import json
from datetime import datetime

# 生成文件名（包含时间戳）
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"aruco_average_positions.json"

try:
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    print(f"\n数据已保存到文件: {filename}")
except Exception as e:
    print(f"保存文件时出错: {e}")



