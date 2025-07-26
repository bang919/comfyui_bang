import cv2
import numpy as np
import torch
from PIL import Image


class ImageInserter:
    """
    图片插入器 - 将目标图片通过透视变换插入到背景图中
    
    功能:
    - 修复四边形顶点排序问题
    - 解决黑边问题
    - 优化小区域蒙版处理
    """
    
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "background_image": ("IMAGE",),
                "background_mask": ("MASK",),
                "target_image": ("IMAGE",),
                "rotation_angle": ("FLOAT", {
                    "default": 0.0,
                    "min": -180.0,
                    "max": 180.0,
                    "step": 1.0,
                    "display": "number"
                }),
                "feather_inner_expand": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 100.0,
                    "step": 0.5,
                    "display": "number"
                }),
                "feather_outer_expand": ("FLOAT", {
                    "default": 3.0,
                    "min": 0.0,
                    "max": 100.0,
                    "step": 0.5,
                    "display": "number"
                }),
                "mask_expand": ("FLOAT", {
                    "default": 0.0,
                    "min": -10.0,
                    "max": 10.0,
                    "step": 0.5,
                    "display": "number"
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("composited_image", "edge_feather_mask")
    FUNCTION = "execute"
    CATEGORY = "image/transform"

    def tensor_to_cv2(self, tensor_image):
        """将ComfyUI tensor转换为OpenCV格式"""
        if len(tensor_image.shape) == 4:
            tensor_image = tensor_image[0]
        
        image = tensor_image.cpu().numpy()
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
            
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
        return image

    def cv2_to_tensor(self, cv2_image):
        """将OpenCV图像转换为ComfyUI tensor格式"""
        if len(cv2_image.shape) == 3 and cv2_image.shape[2] == 3:
            cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        
        tensor = torch.from_numpy(cv2_image.astype(np.float32) / 255.0)
        
        if len(tensor.shape) == 3:
            tensor = tensor.unsqueeze(0)
        elif len(tensor.shape) == 2:
            tensor = tensor.unsqueeze(-1).unsqueeze(0)
            
        return tensor

    def mask_to_cv2(self, mask_tensor):
        """将mask tensor转换为OpenCV格式"""
        if len(mask_tensor.shape) == 3:
            mask_tensor = mask_tensor[0]
        
        mask = mask_tensor.cpu().numpy()
        if mask.max() <= 1.0:
            mask = (mask * 255).astype(np.uint8)
        else:
            mask = mask.astype(np.uint8)
            
        return mask

    def enhance_small_mask(self, mask, expand_pixels=0):
        """应用蒙版扩展/收缩"""
        if expand_pixels != 0:
            if expand_pixels > 0:
                kernel = np.ones((int(expand_pixels*2+1), int(expand_pixels*2+1)), np.uint8)
                mask = cv2.dilate(mask, kernel, iterations=1)
            else:
                kernel = np.ones((int(abs(expand_pixels)*2+1), int(abs(expand_pixels)*2+1)), np.uint8)
                mask = cv2.erode(mask, kernel, iterations=1)
        
        return mask

    def sort_quadrilateral_points(self, points):
        """根据四边形实际形状智能排序顶点：左上、右上、右下、左下"""
        # 计算质心
        center = np.mean(points, axis=0)
        
        # 分析四边形的形状特征
        # 找到最上面的点和最下面的点
        y_coords = points[:, 1]
        top_indices = np.where(y_coords < center[1])[0]
        bottom_indices = np.where(y_coords >= center[1])[0]
        
        # 确保有正确的上下分组
        if len(top_indices) < 2:
            y_sorted_indices = np.argsort(points[:, 1])
            top_indices = y_sorted_indices[:2]
            bottom_indices = y_sorted_indices[2:]
        elif len(bottom_indices) < 2:
            y_sorted_indices = np.argsort(points[:, 1])
            top_indices = y_sorted_indices[:2]
            bottom_indices = y_sorted_indices[2:]
            
        # 上方的两个点和下方的两个点
        top_points = points[top_indices]
        bottom_points = points[bottom_indices]
        
        # 分析四边形的倾斜方向
        # 通过比较上边和下边的中点位置来判断倾斜
        top_center_x = np.mean(top_points[:, 0])
        bottom_center_x = np.mean(bottom_points[:, 0])
        
        # 检测倾斜方向
        tilt_threshold = 5.0  # 倾斜阈值，小于此值认为是垂直四边形
        is_right_tilted = (top_center_x - bottom_center_x) > tilt_threshold
        is_left_tilted = (bottom_center_x - top_center_x) > tilt_threshold
        
        if is_right_tilted:
            # 向右倾斜：上边偏右，下边偏左
            # 对于向右倾斜，需要交叉匹配
            top_x_sorted = np.argsort(top_points[:, 0])
            bottom_x_sorted = np.argsort(bottom_points[:, 0])
            
            # 交叉匹配：上边较右的点对应右上，下边较左的点对应左下
            top_left = top_points[top_x_sorted[0]]      # 上方x较小的 -> 左上
            top_right = top_points[top_x_sorted[1]]     # 上方x较大的 -> 右上
            bottom_left = bottom_points[bottom_x_sorted[0]]   # 下方x较小的 -> 左下  
            bottom_right = bottom_points[bottom_x_sorted[1]]  # 下方x较大的 -> 右下
            
        elif is_left_tilted:
            # 向左倾斜：上边偏左，下边偏右
            top_x_sorted = np.argsort(top_points[:, 0])
            bottom_x_sorted = np.argsort(bottom_points[:, 0])
            
            top_left = top_points[top_x_sorted[0]]      # 上方x较小的 -> 左上
            top_right = top_points[top_x_sorted[1]]     # 上方x较大的 -> 右上
            bottom_left = bottom_points[bottom_x_sorted[0]]   # 下方x较小的 -> 左下
            bottom_right = bottom_points[bottom_x_sorted[1]]  # 下方x较大的 -> 右下
            
        else:
            # 垂直或接近垂直的四边形
            # 标准处理：按x坐标分左右
            x_sorted_indices = np.argsort(points[:, 0])
            x_sorted_points = points[x_sorted_indices]
            
            left_points = x_sorted_points[:2]
            right_points = x_sorted_points[2:]
            
            left_y_sorted = left_points[np.argsort(left_points[:, 1])]
            top_left = left_y_sorted[0]
            bottom_left = left_y_sorted[1]
            
            right_y_sorted = right_points[np.argsort(right_points[:, 1])]
            top_right = right_y_sorted[0]
            bottom_right = right_y_sorted[1]
        
        # 按标准顺序排列：左上、右上、右下、左下
        quadrilateral = np.array([
            top_left,     # 左上 [0]
            top_right,    # 右上 [1]
            bottom_right, # 右下 [2]
            bottom_left   # 左下 [3]
        ], dtype=np.float32)
        
        return quadrilateral

    def find_quadrilateral_from_mask(self, mask):
        """从mask中提取四边形区域，使用最大面积四边形方法"""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            raise ValueError("未在mask中找到任何轮廓")
        
        largest_contour = max(contours, key=cv2.contourArea)
        
        # 获取轮廓的所有点
        contour_points = largest_contour.reshape(-1, 2)
        
        # 使用最大面积四边形方法
        quadrilateral = self.find_max_area_quadrilateral(contour_points)
        
        return quadrilateral
    
    def find_max_area_quadrilateral(self, points):
        """从点集中找到面积最大的四边形"""
        from itertools import combinations
        
        # 如果点太多，先进行降采样
        if len(points) > 50:
            # 使用Douglas-Peucker算法简化轮廓
            epsilon = 0.01 * cv2.arcLength(points.reshape(-1, 1, 2), True)
            simplified = cv2.approxPolyDP(points.reshape(-1, 1, 2), epsilon, True)
            points = simplified.reshape(-1, 2)
        
        # 如果点数仍然太多，采用均匀采样
        if len(points) > 30:
            indices = np.linspace(0, len(points)-1, 30, dtype=int)
            points = points[indices]
        
        # 如果点数少于4个，使用边界矩形
        if len(points) < 4:
            x_coords = points[:, 0]
            y_coords = points[:, 1]
            min_x, max_x = np.min(x_coords), np.max(x_coords)
            min_y, max_y = np.min(y_coords), np.max(y_coords)
            
            quadrilateral = np.array([
                [min_x, min_y],  # 左上
                [max_x, min_y],  # 右上  
                [max_x, max_y],  # 右下
                [min_x, max_y]   # 左下
            ], dtype=np.float32)
            
            return quadrilateral
        
        max_area = 0
        best_quad = None
        
        # 枚举所有可能的四点组合
        for quad_indices in combinations(range(len(points)), 4):
            quad_points = points[list(quad_indices)]
            
            # 计算四边形面积
            area = self.calculate_quadrilateral_area(quad_points)
            
            if area > max_area:
                max_area = area
                best_quad = quad_points
        
        if best_quad is None:
            best_quad = points[:4]
        
        # 使用智能排序函数
        quadrilateral = self.sort_quadrilateral_points(best_quad)
        
        return quadrilateral
    
    def calculate_quadrilateral_area(self, points):
        """计算四边形面积，使用鞋带公式"""
        if len(points) != 4:
            return 0
        
        # 鞋带公式计算多边形面积
        # Area = 0.5 * |Σ(x_i * y_{i+1} - x_{i+1} * y_i)|
        x = points[:, 0]
        y = points[:, 1]
        
        # 闭合多边形：最后一个点连接到第一个点
        area = 0.5 * abs(sum(x[i] * y[(i + 1) % 4] - x[(i + 1) % 4] * y[i] for i in range(4)))
        
        return area

    def rotate_image(self, image, angle):
        """旋转图像"""
        if angle == 0:
            return image
            
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        cos_val = np.abs(rotation_matrix[0, 0])
        sin_val = np.abs(rotation_matrix[0, 1])
        new_width = int((height * sin_val) + (width * cos_val))
        new_height = int((height * cos_val) + (width * sin_val))
        
        rotation_matrix[0, 2] += (new_width - width) / 2
        rotation_matrix[1, 2] += (new_height - height) / 2
        
        rotated = cv2.warpAffine(image, rotation_matrix, (new_width, new_height))
        
        return rotated

    def create_quadrilateral_mask(self, shape, quadrilateral):
        """根据四边形顶点创建纯四边形蒙版"""
        mask = np.zeros(shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [quadrilateral.astype(np.int32)], 255)
        return mask

    def create_feather_mask(self, shape, quadrilateral, inner_expand, outer_expand):
        """基于四边形创建羽化蒙版，只输出边缘羽化轮廓"""
        quad_mask = self.create_quadrilateral_mask(shape, quadrilateral)
        
        if inner_expand == 0 and outer_expand == 0:
            return quad_mask.astype(np.float32) / 255.0
        
        dist_inside = cv2.distanceTransform(quad_mask, cv2.DIST_L2, 3)
        dist_outside = cv2.distanceTransform(255 - quad_mask, cv2.DIST_L2, 3)
        
        feather_mask = np.zeros_like(quad_mask, dtype=np.float32)
        
        if inner_expand > 0:
            inner_region = dist_inside > inner_expand
            transition_region = (dist_inside <= inner_expand) & (quad_mask > 0)
            feather_mask[inner_region] = 1.0
            feather_mask[transition_region] = dist_inside[transition_region] / inner_expand
        else:
            feather_mask[quad_mask > 0] = 1.0
        
        if outer_expand > 0:
            outer_region = (dist_outside <= outer_expand) & (quad_mask == 0)
            feather_mask[outer_region] = np.maximum(
                feather_mask[outer_region],
                1.0 - (dist_outside[outer_region] / outer_expand)
            )
        
        return feather_mask
    
    def create_edge_feather_mask(self, shape, quadrilateral, inner_expand, outer_expand):
        """创建只包含边缘羽化轮廓的蒙版"""
        quad_mask = self.create_quadrilateral_mask(shape, quadrilateral)
        
        if inner_expand == 0 and outer_expand == 0:
            # 没有羽化，返回空蒙版
            return np.zeros_like(quad_mask, dtype=np.float32)
        
        dist_inside = cv2.distanceTransform(quad_mask, cv2.DIST_L2, 3)
        dist_outside = cv2.distanceTransform(255 - quad_mask, cv2.DIST_L2, 3)
        
        edge_feather_mask = np.zeros_like(quad_mask, dtype=np.float32)
        
        # 内部羽化边缘
        if inner_expand > 0:
            inner_edge_region = (dist_inside <= inner_expand) & (quad_mask > 0)
            edge_feather_mask[inner_edge_region] = dist_inside[inner_edge_region] / inner_expand
        
        # 外部羽化边缘
        if outer_expand > 0:
            outer_edge_region = (dist_outside <= outer_expand) & (quad_mask == 0)
            edge_feather_mask[outer_edge_region] = np.maximum(
                edge_feather_mask[outer_edge_region],
                1.0 - (dist_outside[outer_edge_region] / outer_expand)
            )
        
        return edge_feather_mask

    def perspective_transform_no_black_border(self, target_image, quadrilateral, output_shape):
        """透视变换 - 避免黑边"""
        height, width = target_image.shape[:2]
        output_height, output_width = output_shape[:2]
        
        # 目标图像的四个角点（标准顺序：左上、右上、右下、左下）
        src_points = np.array([
            [0, 0],                    # 左上
            [width - 1, 0],            # 右上
            [width - 1, height - 1],   # 右下
            [0, height - 1]            # 左下
        ], dtype=np.float32)
        
        # 确保四边形顶点在输出图像范围内
        quadrilateral_clipped = np.copy(quadrilateral)
        quadrilateral_clipped[:, 0] = np.clip(quadrilateral_clipped[:, 0], 0, output_width - 1)
        quadrilateral_clipped[:, 1] = np.clip(quadrilateral_clipped[:, 1], 0, output_height - 1)
        
        perspective_matrix = cv2.getPerspectiveTransform(src_points, quadrilateral_clipped)
        
        # 透视变换 - 使用BORDER_TRANSPARENT避免黑边
        warped = cv2.warpPerspective(
            target_image, 
            perspective_matrix, 
            (output_width, output_height),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_TRANSPARENT
        )
        
        return warped

    def execute(self, background_image, background_mask, target_image, 
                rotation_angle, feather_inner_expand, feather_outer_expand, 
                mask_expand):
        try:
            # 转换输入格式
            bg_cv2 = self.tensor_to_cv2(background_image)
            mask_cv2 = self.mask_to_cv2(background_mask)
            target_cv2 = self.tensor_to_cv2(target_image)
            
            # 确保背景图和mask尺寸一致
            bg_height, bg_width = bg_cv2.shape[:2]
            if mask_cv2.shape != (bg_height, bg_width):
                mask_cv2 = cv2.resize(mask_cv2, (bg_width, bg_height), interpolation=cv2.INTER_NEAREST)
            
            # 应用蒙版扩展（如果需要）
            if mask_expand != 0:
                mask_cv2 = self.enhance_small_mask(mask_cv2, mask_expand)
            
            # 1. 从mask中提取四边形
            quadrilateral = self.find_quadrilateral_from_mask(mask_cv2)
            
            # 2. 旋转目标图像
            if rotation_angle != 0:
                target_cv2 = self.rotate_image(target_cv2, rotation_angle)
            
            # 3. 透视变换
            warped_target = self.perspective_transform_no_black_border(
                target_cv2, quadrilateral, bg_cv2.shape
            )
            
            # 4. 创建羽化蒙版
            # 用于图像融合的完整羽化蒙版
            feather_mask_for_blending = self.create_feather_mask(
                bg_cv2.shape, quadrilateral, feather_inner_expand, feather_outer_expand
            )
            
            # 用于输出的边缘羽化轮廓蒙版
            edge_feather_mask = self.create_edge_feather_mask(
                bg_cv2.shape, quadrilateral, feather_inner_expand, feather_outer_expand
            )
            
            # 5. 图像融合
            if len(bg_cv2.shape) == 3:
                feather_mask_3d = np.stack([feather_mask_for_blending] * 3, axis=2)
            else:
                feather_mask_3d = feather_mask_for_blending
            
            # 确保warped_target有正确的通道数
            if len(bg_cv2.shape) == 3 and len(warped_target.shape) == 2:
                warped_target = cv2.cvtColor(warped_target, cv2.COLOR_GRAY2BGR)
            elif len(bg_cv2.shape) == 2 and len(warped_target.shape) == 3:
                warped_target = cv2.cvtColor(warped_target, cv2.COLOR_BGR2GRAY)
            
            # 执行融合
            composited = (bg_cv2.astype(np.float32) * (1 - feather_mask_3d) + 
                         warped_target.astype(np.float32) * feather_mask_3d)
            composited = np.clip(composited, 0, 255).astype(np.uint8)
            
            # 6. 转换回tensor格式
            composited_tensor = self.cv2_to_tensor(composited)
            # 输出边缘羽化轮廓蒙版（只包含边缘，中心为空）
            edge_feather_mask_tensor = torch.from_numpy(edge_feather_mask).unsqueeze(0)
            
            return (composited_tensor, edge_feather_mask_tensor)
            
        except Exception as e:
            print(f"❌ 图片插入器执行错误: {str(e)}")
            import traceback
            print(f"❌ 详细错误信息:\n{traceback.format_exc()}")
            fallback_mask = torch.zeros((1, background_image.shape[1], background_image.shape[2]))
            return (background_image, fallback_mask)


# 节点映射
NODE_CLASS_MAPPINGS = {
    "ImageInserter": ImageInserter
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageInserter": "图片插入器"
} 