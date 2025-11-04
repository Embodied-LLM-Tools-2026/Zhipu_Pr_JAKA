import cv2
import time
import numpy as np
from typing import Tuple, Dict, Optional

class CameraManager:
    """
    管理多个摄像头（RGB 和深度），支持同时显示、参数读取、点云生成
    支持的功能：
    - 多摄像头管理
    - 深度图左右翻转
    - 真实深度值读取（mm）
    - 根据内外参生成点云
    """
    
    def __init__(self):
        self.cameras = {}
        self.camera_info = {}
        self.camera_calibration = {}  # 摄像头标定参数（内参、外参）
        
    def add_camera(self, name, index, camera_type='rgb', flip_lr=False, camera_matrix=None, dist_coeffs=None, depth_scale=1.0):
        """
        添加摄像头
        Args:
            name: 摄像头名称 (如 'rgb', 'depth', 'left', 'right')
            index: 摄像头索引
            camera_type: 'rgb' 或 'depth'
            flip_lr: 是否左右翻转（深度图常需要）
            camera_matrix: 相机内参矩阵 3x3 (如果有标定数据)
            dist_coeffs: 畸变系数 (如果有标定数据)
            depth_scale: 深度缩放因子（从像素值转换到实际距离，单位mm）
                        例如：灰度值 100 -> 100 * depth_scale mm
        """
        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            print(f"❌ Camera '{name}' (index {index}) not available.")
            return False
        
        # 设置摄像头属性（可选）
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 最小缓冲
        
        self.cameras[name] = cap
        self.camera_info[name] = {
            'index': index,
            'type': camera_type,
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'frames': 0,
            'last_frame': None,
            'last_frame_raw': None,  # 原始帧（未处理）
            'depth_min': None,
            'depth_max': None,
            'flip_lr': flip_lr,
            'depth_scale': depth_scale,
        }
        
        # 保存标定参数
        if camera_matrix is not None:
            self.camera_calibration[name] = {
                'camera_matrix': camera_matrix,
                'dist_coeffs': dist_coeffs if dist_coeffs is not None else np.zeros(5),
            }
        else:
            # 默认内参（可用于测试，不是真实标定值）
            fx = self.camera_info[name]['width']
            fy = self.camera_info[name]['height']
            cx = self.camera_info[name]['width'] / 2
            cy = self.camera_info[name]['height'] / 2
            self.camera_calibration[name] = {
                'camera_matrix': np.array([[fx, 0, cx],
                                          [0, fy, cy],
                                          [0, 0, 1]], dtype=np.float32),
                'dist_coeffs': np.zeros(5, dtype=np.float32),
            }
        
        print(f"✅ Added camera '{name}' (index {index}): {self.camera_info[name]['width']}x{self.camera_info[name]['height']} @ {self.camera_info[name]['fps']:.1f} fps")
        if flip_lr:
            print(f"   ↔️  左右翻转已启用")
        if camera_matrix is not None:
            print(f"   📐 已设置相机内参")
        return True
    
    def read_frame(self, name):
        """读取单个摄像头的一帧"""
        if name not in self.cameras:
            return None
        
        cap = self.cameras[name]
        ret, frame = cap.read()
        
        if ret:
            self.camera_info[name]['frames'] += 1
            self.camera_info[name]['last_frame_raw'] = frame.copy()
            
            # 应用左右翻转（如果需要）
            if self.camera_info[name]['flip_lr']:
                frame = cv2.flip(frame, 1)  # 1 = 左右翻转
            
            self.camera_info[name]['last_frame'] = frame
            return frame
        return None
    
    def get_depth_stats(self, name):
        """获取深度图的统计信息（返回真实深度值 mm）"""
        if name not in self.cameras or self.camera_info[name]['type'] != 'depth':
            return None
        
        frame = self.camera_info[name]['last_frame']
        if frame is None:
            return None
        
        # 如果是彩色图（转为灰度作为深度）
        if len(frame.shape) == 3:
            depth_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            depth_frame = frame
        
        # 转换为真实深度值（mm）
        depth_scale = self.camera_info[name]['depth_scale']
        depth_real = depth_frame.astype(np.float32) * depth_scale
        
        h, w = depth_frame.shape
        depth_min = depth_real.min()
        depth_max = depth_real.max()
        depth_mean = depth_real.mean()
        
        self.camera_info[name]['depth_min'] = depth_min
        self.camera_info[name]['depth_max'] = depth_max
        
        return {
            'size': (w, h),
            'depth_min_mm': depth_min,
            'depth_max_mm': depth_max,
            'depth_mean_mm': depth_mean,
            'center_pixel_mm': depth_real[h//2, w//2],
            'depth_raw': depth_frame,
            'depth_real': depth_real,
        }
    
    def get_pixel_depth(self, name, x, y, return_real=True):
        """
        获取特定像素的深度值
        Args:
            name: 摄像头名称
            x, y: 像素坐标
            return_real: 是否返回真实深度值（mm），否则返回灰度值 0-255
        Returns:
            深度值（mm 或灰度值 0-255）
        """
        if name not in self.cameras:
            return None
        
        frame = self.camera_info[name]['last_frame']
        if frame is None:
            return None
        
        if len(frame.shape) == 3:
            depth_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            depth_frame = frame
        
        h, w = depth_frame.shape
        if 0 <= y < h and 0 <= x < w:
            depth_pixel = depth_frame[y, x]
            
            if return_real:
                # 返回真实深度值（mm）
                depth_scale = self.camera_info[name]['depth_scale']
                return float(depth_pixel) * depth_scale
            else:
                # 返回灰度值
                return depth_pixel
        return None
    
    def generate_pointcloud(self, name, max_points=None, sample_step=1):
        """
        根据深度图和内参生成点云
        Args:
            name: 深度摄像头名称
            max_points: 最多点数（超过此数时采样）
            sample_step: 采样步长（>1 时跳过像素加快速度）
        Returns:
            点云 (N, 3) 的 numpy 数组，单位为 mm
        """
        if name not in self.cameras or self.camera_info[name]['type'] != 'depth':
            print(f"❌ Camera '{name}' is not a depth camera")
            return None
        
        frame = self.camera_info[name]['last_frame']
        if frame is None:
            return None
        
        # 获取深度数据
        if len(frame.shape) == 3:
            depth_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            depth_frame = frame
        
        # 转换为真实深度值（mm）
        depth_scale = self.camera_info[name]['depth_scale']
        depth_real = depth_frame.astype(np.float32) * depth_scale
        
        # 获取相机内参
        calib = self.camera_calibration[name]
        camera_matrix = calib['camera_matrix']
        fx = camera_matrix[0, 0]
        fy = camera_matrix[1, 1]
        cx = camera_matrix[0, 2]
        cy = camera_matrix[1, 2]
        
        h, w = depth_frame.shape
        
        # 生成点云
        points = []
        for v in range(0, h, sample_step):
            for u in range(0, w, sample_step):
                d = depth_real[v, u]
                
                # 跳过无效深度（0 或很小的值）
                if d < 1.0:
                    continue
                
                # 反投影到3D: 根据相机模型 x = (u - cx) * d / fx
                x = (u - cx) * d / fx
                y = (v - cy) * d / fy
                z = d
                
                points.append([x, y, z])
        
        if len(points) == 0:
            print(f"⚠️  No valid points in point cloud")
            return None
        
        pointcloud = np.array(points, dtype=np.float32)
        
        # 采样到指定点数
        if max_points is not None and len(pointcloud) > max_points:
            indices = np.random.choice(len(pointcloud), max_points, replace=False)
            pointcloud = pointcloud[indices]
        
        print(f"✅ Generated point cloud with {len(pointcloud)} points from '{name}'")
        return pointcloud
    
    def get_pointcloud_stats(self, pointcloud):
        """获取点云统计信息"""
        if pointcloud is None or len(pointcloud) == 0:
            return None
        
        return {
            'point_count': len(pointcloud),
            'x_range': (pointcloud[:, 0].min(), pointcloud[:, 0].max()),
            'y_range': (pointcloud[:, 1].min(), pointcloud[:, 1].max()),
            'z_range': (pointcloud[:, 2].min(), pointcloud[:, 2].max()),
            'x_mean': pointcloud[:, 0].mean(),
            'y_mean': pointcloud[:, 1].mean(),
            'z_mean': pointcloud[:, 2].mean(),
        }
    
    def save_pointcloud(self, pointcloud, filename):
        """保存点云为 PLY 文件（可用 CloudCompare 等工具查看）"""
        if pointcloud is None or len(pointcloud) == 0:
            print(f"❌ Point cloud is empty")
            return False
        
        try:
            with open(filename, 'w') as f:
                # PLY 文件头
                f.write("ply\n")
                f.write("format ascii 1.0\n")
                f.write(f"element vertex {len(pointcloud)}\n")
                f.write("property float x\n")
                f.write("property float y\n")
                f.write("property float z\n")
                f.write("end_header\n")
                
                # 写入点
                for p in pointcloud:
                    f.write(f"{p[0]:.3f} {p[1]:.3f} {p[2]:.3f}\n")
            
            print(f"✅ Point cloud saved to {filename}")
            return True
        except Exception as e:
            print(f"❌ Failed to save point cloud: {e}")
            return False
    

    def display_all(self, duration=None):
        """
        同时显示所有摄像头
        Args:
            duration: 显示时长（秒），None 表示无限
        """
        if not self.cameras:
            print("❌ No cameras added.")
            return
        
        print(f"\n📹 Showing {len(self.cameras)} camera(s). Press 'q' or 'ESC' to exit, 's' for stats.")
        start_time = time.time()
        
        try:
            while True:
                frames = {}
                
                # 读取所有摄像头
                for name in self.cameras:
                    frame = self.read_frame(name)
                    if frame is not None:
                        frames[name] = frame
                
                if not frames:
                    break
                
                # 显示所有摄像头
                for name, frame in frames.items():
                    cam_type = self.camera_info[name]['type']
                    title = f"{name} ({cam_type})"
                    
                    # 添加信息文本到帧
                    info_text = f"Frames: {self.camera_info[name]['frames']} | Size: {frame.shape[1]}x{frame.shape[0]}"
                    cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # 如果是深度摄像头，显示深度信息
                    if cam_type == 'depth':
                        stats = self.get_depth_stats(name)
                        if stats:
                            depth_text = f"Depth: {stats['depth_min_mm']:.1f}-{stats['depth_max_mm']:.1f}mm (mean: {stats['depth_mean_mm']:.1f}mm)"
                            cv2.putText(frame, depth_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    
                    cv2.imshow(title, frame)
                
                # 检查按键
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # 'q' 或 ESC
                    break
                elif key == ord('s'):  # 's' 显示统计信息
                    self.print_stats()
                
                # 检查时长
                if duration is not None and (time.time() - start_time) > duration:
                    break
        
        finally:
            self.release_all()
    
    def print_stats(self):
        """打印所有摄像头的统计信息"""
        print("\n" + "="*60)
        print("📊 Camera Statistics")
        print("="*60)
        for name, info in self.camera_info.items():
            print(f"\n📷 Camera: {name}")
            print(f"   Type: {info['type']}")
            print(f"   Index: {info['index']}")
            print(f"   Resolution: {info['width']}x{info['height']}")
            print(f"   FPS: {info['fps']:.1f}")
            print(f"   Frames captured: {info['frames']}")
            if info['flip_lr']:
                print(f"   ↔️  Left-Right Flip: ON")
            
            if info['type'] == 'depth':
                stats = self.get_depth_stats(name)
                if stats:
                    print(f"   📏 Depth range: {stats['depth_min_mm']:.1f} - {stats['depth_max_mm']:.1f} mm")
                    print(f"   📊 Depth mean: {stats['depth_mean_mm']:.1f} mm")
                    print(f"   🎯 Center pixel depth: {stats['center_pixel_mm']:.1f} mm")
        print("="*60 + "\n")
    
    def release_all(self):
        """释放所有摄像头"""
        for cap in self.cameras.values():
            cap.release()
        cv2.destroyAllWindows()
        print("✅ All cameras released.")

def show_camera(index, duration=100):
    """原函数，保留向后兼容性"""
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        print(f"Camera {index} not available.")
        return False
    print(f"Showing camera {index} for {duration} seconds. Press any key to continue.")
    start = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow(f"Camera {index}", frame)
        if cv2.waitKey(1) != -1 or (time.time() - start) > duration:
            break
    cap.release()
    cv2.destroyAllWindows()
    return True

def main():
    # ============ 示例 1: 同时打开 RGB 和深度摄像头 ============
    print("🎥 Camera Manager - Multi-camera support")
    print("="*60)
    
    manager = CameraManager()
    
    # 添加 RGB 摄像头（索引 3）
    manager.add_camera('rgb', 10, camera_type='rgb')

    # 添加深度摄像头（索引 10 - 你之前的深度摄像头）
    manager.add_camera('depth', 8, camera_type='depth', flip_lr=True)
    
    # 也可以添加更多摄像头
    # manager.add_camera('left', 2, camera_type='rgb')
    # manager.add_camera('right', 5, camera_type='rgb')
    
    print("\n💡 键盘快捷键:")
    print("   'q' or ESC: 退出")
    print("   's': 显示统计信息")
    print("="*60)
    
    # 同时显示所有摄像头（无时长限制）
    manager.display_all(duration=None)
    
    # 显示最终统计信息
    manager.print_stats()

if __name__ == "__main__":
    main()