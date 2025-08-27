import os
import cv2
import pickle
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import json

# 尝试导入sklearn，如果不可用则使用简单线性回归
try:
    from sklearn.linear_model import RANSACRegressor
    SKLEARN_AVAILABLE = True
except ImportError:
    print("Warning: scikit-learn 不可用，将使用简单线性回归替代RANSAC")
    SKLEARN_AVAILABLE = False
    RANSACRegressor = None  # 设置为None以避免未定义错误


class CameraController:
    """相机控制器 - 处理图像拍摄和加载"""
    
    def __init__(self, camera_id: int = 0):
        """
        初始化相机控制器
        Args:
            camera_id: 相机设备ID，默认为0
        """
        self.camera_id = camera_id
        
    def capture_image(self, save_path: Optional[str] = None) -> np.ndarray:
        """
        拍摄当前货架图像
        Args:
            save_path: 可选的保存路径
        Returns:
            拍摄的图像数组
        """
        cap = cv2.VideoCapture(self.camera_id)
        if not cap.isOpened():
            raise RuntimeError(f"无法打开相机 {self.camera_id}")
        
        # 拍摄图像
        ret, frame = cap.read()
        # 人为进行掩码，将货架的上下两边进行掩码，防止干扰
        frame[0:200,:,:] = 255
        frame[350:,:,:] = 255

        cv2.imwrite(r"C:\Users\Work\Documents\Pr_Stage1\archive\examples\images\images\image_1.jpg", frame)

        cap.release()
        
        if not ret:
            raise RuntimeError("拍摄图像失败")
        
        # 保存图像（如果指定了路径）
        # if save_path:
        #     cv2.imwrite(save_path, frame)
        
        return frame
    
    def load_reference_image(self, reference_dir: str, drink_type: str) -> str:
        """
        加载饮料参考图片路径
        Args:
            reference_dir: 参考图片目录
            drink_type: 饮料类型
        Returns:
            参考图片的完整路径
        """
        reference_path = os.path.join(reference_dir, f"{drink_type}_ref.jpg")
        if not os.path.exists(reference_path):
            raise FileNotFoundError(f"找不到参考图片: {reference_path}")
        return reference_path


class PositionCalibrator:
    """位置标定器 - 负责标定和存储各种饮料的可能分布位置"""
    
    def __init__(self, template_dir: str):
        """
        初始化位置标定器
        Args:
            template_dir: 位置模板存储目录
        """
        self.template_dir = template_dir
        os.makedirs(template_dir, exist_ok=True)
    
    def calibrate_drink_positions(self, drink_type: str, full_shelf_mask: np.ndarray) -> Dict:
        """
        标定饮料的可能分布位置
        Args:
            drink_type: 饮料类别
            full_shelf_mask: 放满饮料时的完整掩码
        Returns:
            标定结果信息
        """
        try:
            # 分析掩码，分解出各个独立位置，并提取参考线信息
            individual_positions, reference_line_info = self._decompose_mask_to_positions(full_shelf_mask)
            
            # 创建位置模板
            position_template = {
                "drink_type": drink_type,
                "calibration_date": datetime.now().isoformat(),
                "template_mask": full_shelf_mask,
                "individual_positions": individual_positions,
                "reference_line_info": reference_line_info,
                "total_positions": len(individual_positions)
            }
            
            # 保存模板
            self.save_position_template(drink_type, position_template)
            
            return {
                "success": True,
                "positions_found": len(individual_positions),
                "message": f"成功标定{drink_type}的{len(individual_positions)}个位置"
            }
        
        except Exception as e:
            return {
                "success": False,
                "positions_found": 0,
                "message": f"标定失败: {str(e)}"
            }
    
    def _decompose_mask_to_positions(self, mask: np.ndarray) -> Tuple[Dict[int, np.ndarray], Dict]:
        """
        将完整掩码分解为各个独立位置，并提取参考线信息
        Args:
            mask: 完整的掩码
        Returns:
            位置编号到位置掩码的映射，参考线信息
        """
        # 找到所有连通区域
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        individual_positions = {}
        position_centers = []
        
        # 为每个连通区域创建独立掩码
        for i, contour in enumerate(contours):
            # 过滤掉太小的区域
            area = cv2.contourArea(contour)
            # 晒除掉面积较小的残缺的识别到的掩码块（这些掩码块大部分都是错检部分） # TODO: 测试
            if area < 100:  
                continue
            
            # 创建该位置的独立掩码
            position_mask = np.zeros_like(mask)
            cv2.fillPoly(position_mask, [contour], 255)
            
            # 计算位置中心点
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                position_centers.append((cx, cy, i))
        
        # 提取参考线信息（所有位置都在一条线上）
        reference_line_info = self._extract_reference_line(position_centers)
        
        # 根据参考线方向排序位置
        sorted_centers = self._sort_positions_along_line(position_centers, reference_line_info)
        
        # 分配位置编号
        for idx, (cx, cy, contour_idx) in enumerate(sorted_centers):
            contour = contours[contour_idx]
            position_mask = np.zeros_like(mask)
            cv2.fillPoly(position_mask, [contour], 255)
            individual_positions[idx + 1] = position_mask
        
        return individual_positions, reference_line_info
    
    def _extract_reference_line(self, position_centers: List[Tuple[int, int, int]]) -> Dict:
        """
        提取参考线信息（所有位置的中心连成的直线）
        Args:
            position_centers: 位置中心点列表
        Returns:
            参考线信息字典
        """
        if len(position_centers) < 2:
            return {"type": "insufficient_points"}
        
        # 提取中心点坐标
        points = np.array([(cx, cy) for cx, cy, _ in position_centers])
        
        # 使用RANSAC拟合直线（如果可用），否则使用简单线性回归
        if SKLEARN_AVAILABLE and RANSACRegressor is not None and len(points) >= 3:
            try:
                X = points[:, 0].reshape(-1, 1)
                y = points[:, 1]
                
                ransac = RANSACRegressor(random_state=42)
                ransac.fit(X, y)
                
                # 获取直线参数 y = kx + b
                k = ransac.estimator_.coef_[0]
                b = ransac.estimator_.intercept_
                
                # 计算直线的起点和终点
                x_min, x_max = points[:, 0].min(), points[:, 0].max()
                y_min, y_max = k * x_min + b, k * x_max + b
                
                # 计算直线角度
                angle = np.arctan(k) * 180 / np.pi
                
                reference_line_info = {
                    "type": "line",
                    "slope": k,
                    "intercept": b,
                    "angle": angle,
                    "start_point": (int(x_min), int(y_min)),
                    "end_point": (int(x_max), int(y_max)),
                    "center_points": points.tolist()
                }
                
            except Exception as e:
                reference_line_info = self._simple_line_fit(points)
        else:
            # 使用简单线性回归
            reference_line_info = self._simple_line_fit(points)
        
        return reference_line_info
    
    def _simple_line_fit(self, points: np.ndarray) -> Dict:
        """
        使用简单最小二乘法拟合直线
        Args:
            points: 点坐标数组
        Returns:
            直线信息
        """
        if len(points) >= 2:
            # 使用最小二乘法拟合直线
            A = np.vstack([points[:, 0], np.ones(len(points))]).T
            k, b = np.linalg.lstsq(A, points[:, 1], rcond=None)[0]
            
            x_min, x_max = points[:, 0].min(), points[:, 0].max()
            y_min, y_max = k * x_min + b, k * x_max + b
            angle = np.arctan(k) * 180 / np.pi
            
            return {
                "type": "line",
                "slope": k,
                "intercept": b,
                "angle": angle,
                "start_point": (int(x_min), int(y_min)),
                "end_point": (int(x_max), int(y_max)),
                "center_points": points.tolist()
            }
        else:
            return {"type": "insufficient_points"}
    
    def _sort_positions_along_line(self, position_centers: List[Tuple[int, int, int]], 
                                   reference_line_info: Dict) -> List[Tuple[int, int, int]]:
        """
        沿着参考线方向对位置进行排序
        Args:
            position_centers: 位置中心点列表
            reference_line_info: 参考线信息
        Returns:
            排序后的位置中心点列表
        """
        if reference_line_info["type"] != "line":
            # 如果没有有效的参考线，使用默认排序
            return sorted(position_centers, key=lambda x: (x[1] // 100, x[0]))
        
        # 计算每个点在参考线方向上的投影距离
        slope = reference_line_info["slope"]
        
        # 归一化的方向向量
        if abs(slope) < 1e-6:  # 水平线
            direction_vector = np.array([1, 0])
        else:
            direction_vector = np.array([1, slope])
            direction_vector = direction_vector / np.linalg.norm(direction_vector)
        
        # 计算投影距离
        projections = []
        for cx, cy, idx in position_centers:
            point = np.array([cx, cy])
            # 使用参考线起点作为原点
            start_point = np.array(reference_line_info["start_point"])
            relative_point = point - start_point
            projection_distance = np.dot(relative_point, direction_vector)
            projections.append((projection_distance, cx, cy, idx))
        
        # 按投影距离排序
        projections.sort(key=lambda x: x[0])
        
        return [(cx, cy, idx) for _, cx, cy, idx in projections]
    
    def save_position_template(self, drink_type: str, template: Dict):
        """保存位置模板到文件"""
        template_path = os.path.join(self.template_dir, f"{drink_type}_template.pkl")
        with open(template_path, 'wb') as f:
            pickle.dump(template, f)
        print(f"位置模板已保存到: {template_path}")
    
    def load_position_template(self, drink_type: str) -> Dict:
        """从文件加载位置模板"""
        template_path = os.path.join(self.template_dir, f"{drink_type}_template.pkl")
        if not os.path.exists(template_path):
            raise FileNotFoundError(f"找不到位置模板: {template_path}")
        
        with open(template_path, 'rb') as f:
            template = pickle.load(f)
        return template


class DrinkFinder:
    """饮料查找器 - 基于标定的位置模板查找当前货架上的饮料"""
    
    def __init__(self, overlap_threshold: float = 0.05, alignment_enabled: bool = True):
        """
        初始化饮料查找器
        Args:
            overlap_threshold: 重叠度阈值，用于判断位置是否有饮料
            alignment_enabled: 是否启用掩码对齐功能
        """
        self.overlap_threshold = overlap_threshold
        self.alignment_enabled = alignment_enabled
    
    def find_drinks(self, current_mask: np.ndarray, position_template: Dict, quantity: int, grabbing_direction: Optional[str] = "left") -> Dict:
        """
        查找指定数量的饮料位置
        Args:
            current_mask: 当前检测到的掩码
            position_template: 位置模板
            quantity: 需要的数量
        Returns:
            查找结果
        """
        try:
            # 如果启用对齐功能，先对当前掩码进行仿射变换对齐
            aligned_mask = current_mask
            if self.alignment_enabled and "reference_line_info" in position_template:
                aligned_mask = self._align_mask_to_template(current_mask, position_template)

            # 将当前检测到的mask分解为独立的个体位置
            current_individual_positions, _ = self._decompose_current_mask(aligned_mask)
            
            # 找到当前占用的位置
            occupied_positions = self._map_current_to_template_individual(current_individual_positions, position_template)
            
            # 排序位置编号
            sorted_positions = sorted(occupied_positions)
            
            # 返回前N个位置
            M = len(occupied_positions)
            if quantity > M:
                return {
                    "success": False,
                    "positions": [],
                    "found_count": M,
                    "message": f"当前货架上只有{M}个饮料"
                }
            else:
                if grabbing_direction == "left":
                    selected_positions = sorted_positions[:quantity]
                else:
                    selected_positions = sorted_positions[M-quantity:]
            
            return {
                "success": True,
                "positions": selected_positions,
                "found_count": len(selected_positions),
                "total_available": len(occupied_positions),
                "message": f"找到{len(selected_positions)}个位置: {selected_positions}"
            }
        
        except Exception as e:
            return {
                "success": False,
                "positions": [],
                "found_count": 0,
                "total_available": 0,
                "message": f"查找失败: {str(e)}"
            }
    
    def _map_current_to_template(self, current_mask: np.ndarray, position_template: Dict) -> List[int]:
        """
        将当前检测结果映射到标定的位置编号
        Args:
            current_mask: 当前掩码
            position_template: 位置模板
        Returns:
            占用的位置编号列表
        """
        occupied_positions = []
        
        for pos_id, pos_mask in position_template["individual_positions"].items():
            # 计算重叠度
            overlap = self._calculate_overlap(current_mask, pos_mask)
            
            # 如果重叠度超过阈值，认为该位置有饮料
            if overlap > self.overlap_threshold:
                occupied_positions.append(pos_id)
        
        print(f"总共找到 {len(occupied_positions)} 个占用位置: {occupied_positions}")
        return occupied_positions
    
    def _decompose_current_mask(self, mask: np.ndarray) -> Tuple[Dict[int, np.ndarray], Dict]:
        """
        将当前检测到的mask分解为独立的个体位置
        Args:
            mask: 当前检测到的合并mask
        Returns:
            位置编号到位置mask的映射，参考线信息
        """
        # 找到所有连通区域
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        print(f"当前mask检测到 {len(contours)} 个轮廓")
        
        individual_positions = {}
        position_centers = []
        
        # 为每个连通区域创建独立mask
        for i, contour in enumerate(contours):
            # 过滤掉太小的区域
            area = cv2.contourArea(contour)
            if area < 100:  # 最小面积阈值
                continue
            
            
            # 创建该位置的独立mask
            position_mask = np.zeros_like(mask)
            cv2.fillPoly(position_mask, [contour], 255)
            
            # 计算位置中心点
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                position_centers.append((cx, cy, i))
        
        # 提取参考线信息
        reference_line_info = self._extract_reference_line(position_centers)
        
        # 根据参考线方向排序位置
        sorted_centers = self._sort_positions_along_line(position_centers, reference_line_info)
        
        # 分配位置编号
        for idx, (cx, cy, contour_idx) in enumerate(sorted_centers):
            contour = contours[contour_idx]
            position_mask = np.zeros_like(mask)
            cv2.fillPoly(position_mask, [contour], 255)
            individual_positions[idx + 1] = position_mask
        
        return individual_positions, reference_line_info
    
    def _map_current_to_template_individual(self, current_individual_positions: Dict[int, np.ndarray], 
                                          position_template: Dict) -> List[int]:
        """
        将当前分解后的个体位置映射到模板位置
        Args:
            current_individual_positions: 当前分解后的个体位置
            position_template: 位置模板
        Returns:
            占用的位置编号列表
        """
        occupied_positions = []

        # 对每个当前检测到的个体位置
        for current_pos_id, current_pos_mask in current_individual_positions.items():
            best_match_pos = None
            best_overlap = 0
            
            # 找到与模板位置重叠度最高的位置
            for template_pos_id, template_pos_mask in position_template["individual_positions"].items():
                overlap = self._calculate_overlap(current_pos_mask, template_pos_mask)
                
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_match_pos = template_pos_id
            
            # 如果最佳重叠度超过阈值，认为该模板位置被占用
            if best_overlap > self.overlap_threshold and best_match_pos is not None:
                if best_match_pos not in occupied_positions:
                    occupied_positions.append(best_match_pos)
        
        return occupied_positions
    
    def _extract_reference_line(self, position_centers: List[Tuple[int, int, int]]) -> Dict:
        """
        提取参考线信息（所有位置的中心连成的直线）
        Args:
            position_centers: 位置中心点列表
        Returns:
            参考线信息字典
        """
        if len(position_centers) < 2:
            return {"type": "insufficient_points"}
        
        # 提取中心点坐标
        points = np.array([(cx, cy) for cx, cy, _ in position_centers])
        
        # 使用RANSAC拟合直线（如果可用），否则使用简单线性回归
        if SKLEARN_AVAILABLE and RANSACRegressor is not None and len(points) >= 3:
            try:
                X = points[:, 0].reshape(-1, 1)
                y = points[:, 1]
                
                ransac = RANSACRegressor(random_state=42)
                ransac.fit(X, y)
                
                # 获取直线参数 y = kx + b
                k = ransac.estimator_.coef_[0]
                b = ransac.estimator_.intercept_
                
                # 计算直线的起点和终点
                x_min, x_max = points[:, 0].min(), points[:, 0].max()
                y_min, y_max = k * x_min + b, k * x_max + b
                
                # 计算直线角度
                angle = np.arctan(k) * 180 / np.pi
                
                reference_line_info = {
                    "type": "line",
                    "slope": k,
                    "intercept": b,
                    "angle": angle,
                    "start_point": (int(x_min), int(y_min)),
                    "end_point": (int(x_max), int(y_max)),
                    "center_points": points.tolist()
                }
                
            except Exception as e:
                reference_line_info = self._simple_line_fit(points)
        else:
            # 使用简单线性回归
            reference_line_info = self._simple_line_fit(points)
        
        return reference_line_info
    
    def _simple_line_fit(self, points: np.ndarray) -> Dict:
        """
        使用简单最小二乘法拟合直线
        Args:
            points: 点坐标数组
        Returns:
            直线信息
        """
        if len(points) >= 2:
            # 使用最小二乘法拟合直线
            A = np.vstack([points[:, 0], np.ones(len(points))]).T
            k, b = np.linalg.lstsq(A, points[:, 1], rcond=None)[0]
            
            x_min, x_max = points[:, 0].min(), points[:, 0].max()
            y_min, y_max = k * x_min + b, k * x_max + b
            angle = np.arctan(k) * 180 / np.pi
            
            return {
                "type": "line",
                "slope": k,
                "intercept": b,
                "angle": angle,
                "start_point": (int(x_min), int(y_min)),
                "end_point": (int(x_max), int(y_max)),
                "center_points": points.tolist()
            }
        else:
            return {"type": "insufficient_points"}
    
    def _sort_positions_along_line(self, position_centers: List[Tuple[int, int, int]], 
                                   reference_line_info: Dict) -> List[Tuple[int, int, int]]:
        """
        沿着参考线方向对位置进行排序
        Args:
            position_centers: 位置中心点列表
            reference_line_info: 参考线信息
        Returns:
            排序后的位置中心点列表
        """
        if reference_line_info["type"] != "line":
            # 如果没有有效的参考线，使用默认排序
            return sorted(position_centers, key=lambda x: (x[1] // 100, x[0]))
        
        # 计算每个点在参考线方向上的投影距离
        slope = reference_line_info["slope"]
        
        # 归一化的方向向量
        if abs(slope) < 1e-6:  # 水平线
            direction_vector = np.array([1, 0])
        else:
            direction_vector = np.array([1, slope])
            direction_vector = direction_vector / np.linalg.norm(direction_vector)
        
        # 计算投影距离
        projections = []
        for cx, cy, idx in position_centers:
            point = np.array([cx, cy])
            # 使用参考线起点作为原点
            start_point = np.array(reference_line_info["start_point"])
            relative_point = point - start_point
            projection_distance = np.dot(relative_point, direction_vector)
            projections.append((projection_distance, cx, cy, idx))
        
        # 按投影距离排序
        projections.sort(key=lambda x: x[0])
        
        return [(cx, cy, idx) for _, cx, cy, idx in projections]
    
    def _calculate_overlap(self, mask1: np.ndarray, mask2: np.ndarray) -> float:
        """
        计算两个掩码的重叠度
        Args:
            mask1: 掩码1
            mask2: 掩码2
        Returns:
            重叠度 (0-1)
        """
        # 将掩码二值化
        mask1_binary = (mask1 > 0).astype(np.uint8)
        mask2_binary = (mask2 > 0).astype(np.uint8)
        
        # 计算交集和并集
        intersection = np.logical_and(mask1_binary, mask2_binary).sum()
        union = np.logical_or(mask1_binary, mask2_binary).sum()
        
        # 避免除零
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def _align_mask_to_template(self, current_mask: np.ndarray, position_template: Dict) -> np.ndarray:
        """
        将当前掩码通过仿射变换对齐到模板掩码
        基于个体mask匹配度最大化而不是参考线对齐
        Args:
            current_mask: 当前检测的掩码
            position_template: 位置模板
        Returns:
            对齐后的掩码
        """
        try:
            # 分解当前mask为个体位置
            current_individual_positions, _ = self._decompose_current_mask(current_mask)
            
            if len(current_individual_positions) == 0:
                return current_mask
            
            # 获取模板个体位置
            template_individual_positions = position_template["individual_positions"]
            
            if len(template_individual_positions) == 0:
                return current_mask
            
            # 计算最优仿射变换矩阵
            optimal_transform = self._compute_optimal_affine_transform(
                current_individual_positions, 
                template_individual_positions
            )
            
            if optimal_transform is None:
                return current_mask
            
            # 应用仿射变换
            aligned_mask = self._apply_affine_transform(current_mask, optimal_transform)
            
            return aligned_mask
            
        except Exception as e:
            return current_mask
    
    def _compute_optimal_affine_transform(self, current_individual_positions: Dict[int, np.ndarray], 
                                         template_individual_positions: Dict[int, np.ndarray]) -> Optional[np.ndarray]:
        """
        计算最优仿射变换矩阵，使得变换后的个体mask与模板mask匹配度最大化
        Args:
            current_individual_positions: 当前个体位置字典
            template_individual_positions: 模板个体位置字典
        Returns:
            最优的2x3仿射变换矩阵，如果无法计算则返回None
        """
        try:
            # 提取当前和模板个体mask的中心点
            current_centers = self._extract_mask_centers(current_individual_positions)
            template_centers = self._extract_mask_centers(template_individual_positions)
            
            if len(current_centers) == 0 or len(template_centers) == 0:
                return None
            
            # 使用蒙特卡洛方法找到最优变换参数
            optimal_transform = self._monte_carlo_optimal_transform(
                current_individual_positions,
                template_individual_positions,
                current_centers,
                template_centers
            )
            
            return optimal_transform
            
        except Exception as e:
            return None
    
    def _extract_mask_centers(self, individual_positions: Dict[int, np.ndarray]) -> List[Tuple[float, float]]:
        """
        提取个体mask的中心点
        Args:
            individual_positions: 个体位置字典
        Returns:
            中心点列表
        """
        centers = []
        for pos_id, mask in individual_positions.items():
            # 找到mask的轮廓
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) > 0:
                # 取最大的轮廓
                largest_contour = max(contours, key=cv2.contourArea)
                # 计算中心点
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    cx = M["m10"] / M["m00"]
                    cy = M["m01"] / M["m00"]
                    centers.append((cx, cy))
        return centers
    
    def _monte_carlo_optimal_transform(self, current_positions: Dict[int, np.ndarray],
                                     template_positions: Dict[int, np.ndarray],
                                     current_centers: List[Tuple[float, float]],
                                     template_centers: List[Tuple[float, float]]) -> Optional[np.ndarray]:
        """
        使用蒙特卡洛方法找到最优仿射变换参数
        Args:
            current_positions: 当前个体位置
            template_positions: 模板个体位置
            current_centers: 当前中心点
            template_centers: 模板中心点
        Returns:
            最优变换矩阵，如果变换后效果不如原始则返回None
        """
        # 首先计算不进行变换的基准得分
        identity_matrix = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
        baseline_score = self._calculate_transform_score(
            current_positions, template_positions, identity_matrix
        )
        
        baseline_valid = baseline_score >= 0
        
        best_score = baseline_score if baseline_valid else -1
        best_transform = None
        
        # 计算搜索范围
        current_center_mean = np.mean(current_centers, axis=0)
        template_center_mean = np.mean(template_centers, axis=0)
        base_translation = template_center_mean - current_center_mean
        
        # 蒙特卡洛搜索参数范围
        angle_std = 15.0  # 角度标准差（度）
        translation_std = 20.0  # 平移标准差（像素）
        
        # 执行100次蒙特卡洛采样
        for iteration in range(100):
            
            # 随机采样变换参数
            # 角度：正态分布，均值0，标准差15度
            angle = np.random.normal(0, angle_std)
            
            # 平移：基于中心点差异的正态分布
            tx = np.random.normal(base_translation[0], translation_std)
            ty = np.random.normal(base_translation[1], translation_std)
            
            # 构建变换矩阵
            transform_matrix = self._build_transform_matrix(angle, tx, ty, current_center_mean)
            
            # 计算匹配得分
            score = self._calculate_transform_score(
                current_positions, template_positions, transform_matrix
            )
            
            # 检查是否找到更好的解
            if score >= 0 and score > best_score:  # score >= 0 表示满足约束条件
                best_score = score
                best_transform = transform_matrix
            elif score >= 0 and not baseline_valid:
                # 如果基准状态不满足约束，但当前解满足约束，则接受
                if best_score < 0:
                    best_score = score
                    best_transform = transform_matrix
        
        # 判断是否需要进行仿射变换
        if best_transform is not None:
            if baseline_valid:
                if best_score > baseline_score:
                    improvement = best_score - baseline_score
                    return best_transform
                else:
                    return None
            else:
                # 基准状态不满足约束，但找到了满足约束的变换
                return best_transform
        else:
            return None
    
    def _build_transform_matrix(self, angle_deg: float, tx: float, ty: float, 
                               rotation_center: np.ndarray) -> np.ndarray:
        """
        构建仿射变换矩阵
        Args:
            angle_deg: 旋转角度（度）
            tx, ty: 平移量
            rotation_center: 旋转中心
        Returns:
            2x3仿射变换矩阵
        """
        angle_rad = np.radians(angle_deg)
        cos_angle = np.cos(angle_rad)
        sin_angle = np.sin(angle_rad)
        
        # 旋转矩阵
        rotation_matrix = np.array([
            [cos_angle, -sin_angle],
            [sin_angle, cos_angle]
        ])
        
        # 平移向量：先移动到旋转中心，旋转，再移动到目标位置
        translation = np.array([tx, ty]) + rotation_center - rotation_matrix @ rotation_center
        
        # 构建2x3仿射变换矩阵
        transform_matrix = np.zeros((2, 3))
        transform_matrix[:2, :2] = rotation_matrix
        transform_matrix[:, 2] = translation
        
        return transform_matrix
    
    def _calculate_transform_score(self, current_positions: Dict[int, np.ndarray],
                                 template_positions: Dict[int, np.ndarray],
                                 transform_matrix: np.ndarray) -> float:
        """
        计算变换后的匹配得分
        新的优化目标：确保所有个体mask的最高匹配度都大于阈值，然后最大化平均匹配度
        Args:
            current_positions: 当前个体位置
            template_positions: 模板个体位置
            transform_matrix: 变换矩阵
        Returns:
            匹配得分（如果不满足约束条件返回-1，否则返回平均匹配度）
        """
        individual_scores = []
        
        for current_mask in current_positions.values():
            # 对当前mask应用变换
            transformed_mask = self._apply_affine_transform(current_mask, transform_matrix)
            
            # 找到与模板中最匹配的mask
            best_overlap = 0
            for template_mask in template_positions.values():
                overlap = self._calculate_overlap(transformed_mask, template_mask)
                best_overlap = max(best_overlap, overlap)
            
            individual_scores.append(best_overlap)
        
        if len(individual_scores) == 0:
            return 0
        
        # 检查约束条件：所有个体的最高匹配度都必须大于阈值
        min_score = min(individual_scores)
        if min_score < self.overlap_threshold:
            # 不满足约束条件，返回负分以表示无效解
            return -1
        
        # 满足约束条件，返回平均匹配度
        average_score = sum(individual_scores) / len(individual_scores)
        return average_score
    
    def _apply_affine_transform(self, mask: np.ndarray, transform_matrix: np.ndarray) -> np.ndarray:
        """
        应用仿射变换到掩码
        Args:
            mask: 输入掩码
            transform_matrix: 2x3仿射变换矩阵
        Returns:
            变换后的掩码
        """
        height, width = mask.shape
        
        # 应用仿射变换
        transformed_mask = cv2.warpAffine(
            mask, 
            transform_matrix, 
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
        
        return transformed_mask


class DrinkShelfLocator:
    """智能货架饮料定位系统主控制器"""
    
    # 定义饮料类型到边界框(bboxes)的映射
    bbox_map = {
        '可乐': [[328, 241, 359, 315]],
        '雪碧': [[152, 177, 194, 267]],
        '水': [[260, 272, 288, 340]],
        '奶茶': [[332, 236, 371, 367]]
    }
    
    @classmethod
    def load_bbox_map(cls, config_dir: str = "config"):
        """从文件加载bbox_map"""
        bbox_file = os.path.join(config_dir, "bbox_map.json")
        if os.path.exists(bbox_file):
            try:
                with open(bbox_file, 'r', encoding='utf-8') as f:
                    cls.bbox_map = json.load(f)
                print(f"已加载bbox_map: {cls.bbox_map}")
            except Exception as e:
                print(f"加载bbox_map失败: {e}")
    
    @classmethod
    def save_bbox_map(cls, config_dir: str = "config"):
        """保存bbox_map到文件"""
        os.makedirs(config_dir, exist_ok=True)
        bbox_file = os.path.join(config_dir, "bbox_map.json")
        try:
            with open(bbox_file, 'w', encoding='utf-8') as f:
                json.dump(cls.bbox_map, f, ensure_ascii=False, indent=2)
            print(f"已保存bbox_map到: {bbox_file}")
        except Exception as e:
            print(f"保存bbox_map失败: {e}")
    
    def __init__(self, 
                 model_path: str = "weights/yoloe-11l-seg.pt",
                 reference_dir: str = "reference_images",
                 template_dir: str = "position_templates",
                 camera_id: int = 0,
                 config_dir: str = "config"):
        """
        初始化系统
        Args:
            model_path: YOLOE模型路径
            reference_dir: 参考图片目录
            template_dir: 位置模板目录
            camera_id: 相机设备ID
            config_dir: 配置文件目录
        """
        self.model_path = model_path
        self.reference_dir = reference_dir
        self.template_dir = template_dir
        self.config_dir = config_dir
        
        # 创建必要目录
        os.makedirs(reference_dir, exist_ok=True)
        os.makedirs(template_dir, exist_ok=True)
        
        # 加载保存的bbox_map
        self.load_bbox_map(config_dir)
        
        # 初始化各模块
        from .YoloE_Seg import YOLOEProcessor
        self.yoloe_processor = YOLOEProcessor(model_path)
        self.camera = CameraController(camera_id)
        self.calibrator = PositionCalibrator(template_dir)
        self.finder = DrinkFinder(alignment_enabled=True)  # 启用掩码对齐功能
        
    
    def calibrate_positions(self, drink_type: str) -> Dict:
        """
        位置标定接口
        前提：货架上所有可能位置都已放置该种饮料
        Args:
            drink_type: 饮料类型
        Returns:
            标定结果
        """
        try:
            
            # 1. 拍摄当前货架（放满目标饮料）
            print("正在拍摄货架图像...")
            current_image = self.camera.capture_image(
                save_path=f"calibration_{drink_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            )
            # 2. 加载该饮料的参考图片
            visual_prompts = self.get_visual_prompts(drink_type)
            reference_image_path = self.camera.load_reference_image(self.reference_dir, drink_type)
            
            # 3. 使用YOLOE分割出所有饮料位置
            position_masks = self.yoloe_processor.process_images(
                img=current_image,
                visual_prompts=visual_prompts,
                refer_image=reference_image_path
            )
            
            if position_masks is None:
                return {
                    "success": False,
                    "positions_found": 0,
                    "message": "未检测到任何饮料"
                }
            
            # 4. 标定位置
            result = self.calibrator.calibrate_drink_positions(drink_type, position_masks)
            
            print(f"标定完成: {result['message']}")
            return result
        
        except Exception as e:
            error_msg = f"标定过程出错: {str(e)}"
            print(error_msg)
            return {
                "success": False,
                "positions_found": 0,
                "message": error_msg
            }
    
    def find_drinks(self, drink_type: str, quantity: int, shelf_side: Optional[str] = None, grabbing_direction: Optional[str] = None, total_drinks: Optional[int] = None) -> Dict:
        """
        饮料查找接口
        Args:
            drink_type: 饮料类型
            quantity: 需要数量
            grabbing_direction: (可选) 夹取方向 ("left" or "right")
            total_drinks: (可选) 货架总饮料数
        Returns:
            查找结果
        """
        try:
            
            # 1. 拍摄当前货架状态
            print("正在拍摄当前货架状态...")
            current_image = self.camera.capture_image(
                save_path=f"search_{drink_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            )
            
            # 2. 分割当前图像中的目标饮料
            
            visual_prompts = self.get_visual_prompts(drink_type)
            reference_image_path = self.camera.load_reference_image(self.reference_dir, drink_type)
            current_masks = self.yoloe_processor.process_images(
                img=current_image,
                visual_prompts=visual_prompts,
                refer_image=reference_image_path
            )
            
            if current_masks is None:
                return {
                    "success": False,
                    "positions": [],
                    "found_count": 0,
                    "message": f"当前货架上未检测到{drink_type}"
                }
            
            # 如果提供了夹取方向和总数，则使用新策略
            if grabbing_direction and total_drinks is not None:
                print(f"使用新策略查找饮料，夹取方向: {grabbing_direction}, 总数: {total_drinks}")
                
                # 将当前检测到的mask分解为独立的个体位置
                individual_positions, _ = self.finder._decompose_current_mask(current_masks)
                M = len(individual_positions)
                
                if M == 0:
                    return {
                        "success": True,
                        "positions": [],
                        "found_count": 0,
                        "total_available": 0,
                        "message": f"当前货架上未检测到{drink_type}"
                    }

                occupied_positions = []
                if grabbing_direction.lower() == 'left':
                    # 占用位置列表为[N-M+1,N-M+2,...,N-1,N]
                    occupied_positions = list(range(total_drinks - M + 1, total_drinks + 1))
                elif grabbing_direction.lower() == 'right':
                    # 占用位置列表为[1,2,...,M-1,M]
                    occupied_positions = list(range(1, M + 1))
                else:
                    # Invalid direction
                    error_msg = f"无效的夹取方向: {grabbing_direction}。请使用 'left' 或 'right'。"
                    print(error_msg)
                    return {
                        "success": False,
                        "positions": [],
                        "found_count": 0,
                        "message": error_msg
                    }
                if quantity > M:
                    return {
                        "success": False,
                        "positions": [],
                        "found_count": M,
                        "message": f"当前货架上只有{M}个{drink_type}"
                    }
                else:
                    if grabbing_direction.lower() == 'left':
                        selected_positions = occupied_positions[:quantity]
                    else:   
                        selected_positions = occupied_positions[M-quantity:]
                
                result = {
                    "success": True,
                    "positions": selected_positions,
                    "found_count": len(selected_positions),
                    "message": f"找到{len(selected_positions)}个位置: {selected_positions}"
                }
                print(f"查找完成: {result['message']}")
                return result

            # 3. 加载该饮料的位置模板
            position_template = self.calibrator.load_position_template(drink_type)
            
            # 4. 查找饮料位置
            result = self.finder.find_drinks(current_masks, position_template, quantity, grabbing_direction)
            
            print(f"查找完成: {result['message']}")
            return result
        
        except FileNotFoundError as e:
            error_msg = f"找不到{drink_type}的位置模板，请先进行位置标定"
            print(error_msg)
            return {
                "success": False,
                "positions": [],
                "found_count": 0,
                "message": error_msg
            }
        
        except Exception as e:
            error_msg = f"查找过程出错: {str(e)}"
            print(error_msg)
            return {
                "success": False,
                "positions": [],
                "found_count": 0,
                "message": error_msg
            }
    
    def get_available_drink_types(self) -> List[str]:
        """获取已标定的饮料类型列表"""
        drink_types = []
        for file in os.listdir(self.template_dir):
            if file.endswith("_template.pkl"):
                drink_type = file.replace("_template.pkl", "")
                drink_types.append(drink_type)
        return drink_types
    
    def get_calibration_info(self, drink_type: str) -> Dict:
        """获取指定饮料的标定信息"""
        try:
            template = self.calibrator.load_position_template(drink_type)
            return {
                "drink_type": template["drink_type"],
                "calibration_date": template["calibration_date"],
                "total_positions": template["total_positions"],
                "available": True
            }
        except FileNotFoundError:
            return {
                "drink_type": drink_type,
                "available": False,
                "message": "未找到标定信息"
            }

    def get_visual_prompts(self, drink_type: str) -> Optional[dict]:
        """
        根据饮料类型生成视觉提示。
        """
        # 使用 get 方法获取 bboxes，如果 drink_type 不存在则返回 None
        bboxes_data = self.bbox_map.get(drink_type)
        
        # 如果找不到对应的饮料类型，可以决定是返回 None 还是抛出错误
        if bboxes_data is None:
            # 或者 raise ValueError(f"不支持的饮料类型: {drink_type}")
            return None

        return {
            'bboxes': np.array(bboxes_data),
            'cls': np.array([0]),
        }

    def update_bbox_map(self, drink_type: str, bbox: List[int]):
        """
        更新饮料类型的边界框映射
        Args:
            drink_type: 饮料类型
            bbox: 边界框 [x1, y1, x2, y2]
        """
        self.bbox_map[drink_type] = [bbox]
        print(f"已更新 {drink_type} 的边界框: {bbox}")
        # 保存到文件
        self.save_bbox_map(self.config_dir)


if __name__ == "__main__":
    # 使用示例
    
    # 初始化系统
    locator = DrinkShelfLocator(
        model_path="weights/yoloe-11l-seg.pt",
        reference_dir="reference_images",
        template_dir="position_templates",
        camera_id=0
    )
    
    # 示例：标定可乐的位置
    print("=== 位置标定示例 ===")
    print("请确保货架上所有位置都放置了可乐，然后按回车继续...")
    input()
    calibration_result = locator.calibrate_positions("可乐")
    print(f"标定结果: {calibration_result}")
    
    # 示例：查找3瓶可乐
    print("\n=== 饮料查找示例 ===")
    print("请调整货架上的可乐分布，然后按回车继续...")
    input()
    search_result = locator.find_drinks("可乐", 3)
    print(f"查找结果: {search_result}")
    
    # 显示系统信息
    print("\n=== 系统信息 ===")
    available_types = locator.get_available_drink_types()
    print(f"已标定的饮料类型: {available_types}")
    
    for drink_type in available_types:
        info = locator.get_calibration_info(drink_type)
        print(f"{drink_type}: {info}")