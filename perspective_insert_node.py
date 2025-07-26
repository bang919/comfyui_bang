import cv2
import numpy as np
import torch
from PIL import Image


class PerspectiveRegionInsert:
    """
    透视图插入节点 - 将目标图片通过透视变换融合到背景图中
    
    功能:
    - 从mask中提取四边形区域
    - 对目标图进行透视变换
    - 支持羽化边缘处理
    - 输出合成图像和羽化蒙版
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
                    "step": 1.0,
                    "display": "number"
                }),
                "feather_outer_expand": ("FLOAT", {
                    "default": 5.0,
                    "min": 0.0,
                    "max": 100.0,
                    "step": 1.0,
                    "display": "number"
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("composited_image", "feather_mask")
    FUNCTION = "execute"
    CATEGORY = "image/transform"

    def tensor_to_cv2(self, tensor_image):
        """将ComfyUI tensor转换为OpenCV格式"""
        # ComfyUI图像格式: [batch, height, width, channels]
        if len(tensor_image.shape) == 4:
            tensor_image = tensor_image[0]  # 取第一张图
        
        # 转换为numpy并调整范围到0-255
        image = tensor_image.cpu().numpy()
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
            
        # 如果是RGB，转换为BGR (OpenCV格式)
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
        return image

    def cv2_to_tensor(self, cv2_image):
        """将OpenCV图像转换为ComfyUI tensor格式"""
        # 转换BGR到RGB
        if len(cv2_image.shape) == 3 and cv2_image.shape[2] == 3:
            cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        
        # 转换为tensor并标准化到0-1
        tensor = torch.from_numpy(cv2_image.astype(np.float32) / 255.0)
        
        # 添加batch维度: [height, width, channels] -> [1, height, width, channels]
        if len(tensor.shape) == 3:
            tensor = tensor.unsqueeze(0)
        elif len(tensor.shape) == 2:
            # 灰度图，添加channel维度
            tensor = tensor.unsqueeze(-1).unsqueeze(0)
            
        return tensor

    def mask_to_cv2(self, mask_tensor):
        """将mask tensor转换为OpenCV格式"""
        if len(mask_tensor.shape) == 3:
            mask_tensor = mask_tensor[0]  # 取第一张mask
        
        mask = mask_tensor.cpu().numpy()
        if mask.max() <= 1.0:
            mask = (mask * 255).astype(np.uint8)
        else:
            mask = mask.astype(np.uint8)
            
        return mask

    def find_quadrilateral_from_mask(self, mask):
        """从mask中提取四边形区域"""
        # 查找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            raise ValueError("未在mask中找到任何轮廓")
        
        # 选择面积最大的轮廓
        largest_contour = max(contours, key=cv2.contourArea)
        
        # 轮廓逼近为四边形
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        # 如果不是四边形，尝试调整epsilon
        while len(approx) > 4 and epsilon < 0.1 * cv2.arcLength(largest_contour, True):
            epsilon *= 1.5
            approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        if len(approx) < 4:
            # 如果少于4个点，使用边界矩形
            x, y, w, h = cv2.boundingRect(largest_contour)
            approx = np.array([[x, y], [x+w, y], [x+w, y+h], [x, y+h]], dtype=np.float32)
        else:
            approx = approx.reshape(-1, 2).astype(np.float32)
            # 只取前4个点
            approx = approx[:4]
        
        # 按照左上、右上、右下、左下的顺序排序
        center = np.mean(approx, axis=0)
        angles = np.arctan2(approx[:, 1] - center[1], approx[:, 0] - center[0])
        sorted_indices = np.argsort(angles)
        
        # 重新排序为左上、右上、右下、左下
        sorted_points = approx[sorted_indices]
        
        # 找到最上面的两个点和最下面的两个点
        top_points = sorted_points[sorted_points[:, 1] < center[1]]
        bottom_points = sorted_points[sorted_points[:, 1] >= center[1]]
        
        if len(top_points) >= 2 and len(bottom_points) >= 2:
            # 上面两个点按x坐标排序：左上、右上
            top_points = top_points[np.argsort(top_points[:, 0])]
            # 下面两个点按x坐标排序：左下、右下
            bottom_points = bottom_points[np.argsort(bottom_points[:, 0])]
            
            quadrilateral = np.array([
                top_points[0],      # 左上
                top_points[-1],     # 右上
                bottom_points[-1],  # 右下
                bottom_points[0]    # 左下
            ], dtype=np.float32)
        else:
            # 备用方案：使用原始排序
            quadrilateral = sorted_points[:4]
        
        return quadrilateral

    def rotate_image(self, image, angle):
        """旋转图像"""
        if angle == 0:
            return image
            
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        
        # 计算旋转矩阵
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # 计算旋转后的边界框
        cos_val = np.abs(rotation_matrix[0, 0])
        sin_val = np.abs(rotation_matrix[0, 1])
        new_width = int((height * sin_val) + (width * cos_val))
        new_height = int((height * cos_val) + (width * sin_val))
        
        # 调整旋转矩阵以避免裁剪
        rotation_matrix[0, 2] += (new_width - width) / 2
        rotation_matrix[1, 2] += (new_height - height) / 2
        
        # 执行旋转
        rotated = cv2.warpAffine(image, rotation_matrix, (new_width, new_height))
        
        return rotated

    def create_feather_mask(self, mask, inner_expand, outer_expand):
        """创建羽化蒙版"""
        if inner_expand == 0 and outer_expand == 0:
            return mask.astype(np.float32) / 255.0
        
        # 创建距离变换
        dist_inside = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        dist_outside = cv2.distanceTransform(255 - mask, cv2.DIST_L2, 5)
        
        # 计算羽化蒙版
        feather_mask = np.zeros_like(mask, dtype=np.float32)
        
        # 内部区域
        if inner_expand > 0:
            inner_region = dist_inside > inner_expand
            transition_region = (dist_inside <= inner_expand) & (mask > 0)
            feather_mask[inner_region] = 1.0
            feather_mask[transition_region] = dist_inside[transition_region] / inner_expand
        else:
            feather_mask[mask > 0] = 1.0
        
        # 外部羽化
        if outer_expand > 0:
            outer_region = (dist_outside <= outer_expand) & (mask == 0)
            feather_mask[outer_region] = np.maximum(
                feather_mask[outer_region],
                1.0 - (dist_outside[outer_region] / outer_expand)
            )
        
        return feather_mask

    def perspective_transform(self, target_image, quadrilateral):
        """透视变换"""
        height, width = target_image.shape[:2]
        
        # 目标图像的四个角点
        src_points = np.array([
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]
        ], dtype=np.float32)
        
        # 计算透视变换矩阵
        perspective_matrix = cv2.getPerspectiveTransform(src_points, quadrilateral)
        
        # 确定输出图像大小
        canvas_height, canvas_width = target_image.shape[:2]
        if len(quadrilateral) >= 4:
            max_x = int(np.max(quadrilateral[:, 0])) + 50
            max_y = int(np.max(quadrilateral[:, 1])) + 50
            canvas_width = max(canvas_width, max_x)
            canvas_height = max(canvas_height, max_y)
        
        # 执行透视变换
        warped = cv2.warpPerspective(
            target_image, 
            perspective_matrix, 
            (canvas_width, canvas_height)
        )
        
        return warped

    def execute(self, background_image, background_mask, target_image, 
                rotation_angle, feather_inner_expand, feather_outer_expand):
        try:
            # 转换输入格式
            bg_cv2 = self.tensor_to_cv2(background_image)
            mask_cv2 = self.mask_to_cv2(background_mask)
            target_cv2 = self.tensor_to_cv2(target_image)
            
            # 确保背景图和mask尺寸一致
            bg_height, bg_width = bg_cv2.shape[:2]
            if mask_cv2.shape != (bg_height, bg_width):
                mask_cv2 = cv2.resize(mask_cv2, (bg_width, bg_height), interpolation=cv2.INTER_NEAREST)
            
            # 1. 从mask中提取四边形
            quadrilateral = self.find_quadrilateral_from_mask(mask_cv2)
            
            # 2. 旋转目标图像
            if rotation_angle != 0:
                target_cv2 = self.rotate_image(target_cv2, rotation_angle)
            
            # 3. 透视变换
            warped_target = self.perspective_transform(target_cv2, quadrilateral)
            
            # 确保变换后的图像与背景图尺寸一致
            if warped_target.shape[:2] != (bg_height, bg_width):
                warped_target = cv2.resize(warped_target, (bg_width, bg_height))
            
            # 4. 创建羽化蒙版
            feather_mask = self.create_feather_mask(
                mask_cv2, feather_inner_expand, feather_outer_expand
            )
            
            # 5. 图像融合
            if len(bg_cv2.shape) == 3:
                feather_mask_3d = np.stack([feather_mask] * 3, axis=2)
            else:
                feather_mask_3d = feather_mask
            
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
            feather_mask_tensor = torch.from_numpy(feather_mask).unsqueeze(0)
            
            return (composited_tensor, feather_mask_tensor)
            
        except Exception as e:
            print(f"透视插入节点执行错误: {str(e)}")
            # 返回原始背景图和空mask作为fallback
            fallback_mask = torch.zeros((1, background_image.shape[1], background_image.shape[2]))
            return (background_image, fallback_mask)


# 节点映射
NODE_CLASS_MAPPINGS = {
    "PerspectiveRegionInsert": PerspectiveRegionInsert
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PerspectiveRegionInsert": "透视区域插入 (Perspective Region Insert)"
} 