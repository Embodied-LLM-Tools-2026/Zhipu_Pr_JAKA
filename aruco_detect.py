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
camera_matrix = np.array([[547.5700824698947, 0.0, 314.70719414532533], 
                          [0.0, 546.8945958211962, 242.9467184746241], 
                          [0.0, 0.0, 1.0]], dtype=np.float32)
                          
dist_coeffs = np.zeros(5, dtype=np.float32)  # 畸变系数

# ArUco标记的实际大小（米），用户提供的值
marker_length = 0.1  # 实际大小：100mm x 100mm

print("Press 'q' to quit")

while True:
    # 读取一帧
    ret, frame = cap.read()
    if not ret:
        print("Cannot receive frame")
        break
    
    # 转换为灰度图
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 检测Aruco markers
    corners, ids, rejected = detector.detectMarkers(gray)
    
    # 在左上角显示检测信息
    info_text = f"Detected: {len(corners) if ids is not None else 0} markers"
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
            else:
                print(f"位姿估计失败 for marker ID: {ids[i][0]}")
                continue

        # 为每个标记绘制坐标轴并显示位姿
        for i in range(len(rvecs)):
            # 绘制坐标轴
            cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvecs[i], tvecs[i], marker_length * 0.5)
            
            # 计算并显示位姿信息
            pos_text = f"Pos: {tvecs[i][0][0]:.2f}, {tvecs[i][1][0]:.2f}, {tvecs[i][2][0]:.2f}"
            rot_text = f"Rot: {rvecs[i][0][0]:.2f}, {rvecs[i][1][0]:.2f}, {rvecs[i][2][0]:.2f}"
            
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
            
            cv2.putText(frame, f"Marker 1 (ID: {marker1['id']}):", (right_x, y_start), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(frame, f"  Center: ({marker1['center'][0]}, {marker1['center'][1]})", 
                        (right_x, y_start + line_height), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            cv2.putText(frame, f"  Angle: {marker1['angle']:.1f}°", 
                        (right_x, y_start + line_height * 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            cv2.putText(frame, f"  Size: {marker1['size']:.1f}", 
                        (right_x, y_start + line_height * 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            # 如果有第二个marker，显示其信息
            if len(detected_markers) > 1:
                marker2 = detected_markers[1]
                y_start2 = y_start + line_height * 5
                
                cv2.putText(frame, f"Marker 2 (ID: {marker2['id']}):", (right_x, y_start2), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.putText(frame, f"  Center: ({marker2['center'][0]}, {marker2['center'][1]})", 
                            (right_x, y_start2 + line_height), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                cv2.putText(frame, f"  Angle: {marker2['angle']:.1f}°", 
                            (right_x, y_start2 + line_height * 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                cv2.putText(frame, f"  Size: {marker2['size']:.1f}", 
                            (right_x, y_start2 + line_height * 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    
    # 显示帧
    cv2.imshow("Aruco Real-time Detection", frame)
    
    # 检查按键
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()

