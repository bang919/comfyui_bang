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
                "enhance_small_mask": ("BOOLEAN", {
                    "default": True
                }),
                "debug_mode": ("BOOLEAN", {
                    "default": True
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("composited_image", "feather_mask")
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

    def enhance_small_mask(self, mask, expand_pixels=0, debug=False):
        """增强小蒙版区域"""
        if expand_pixels != 0:
            if expand_pixels > 0:
                kernel = np.ones((int(expand_pixels*2+1), int(expand_pixels*2+1)), np.uint8)
                mask = cv2.dilate(mask, kernel, iterations=1)
                if debug:
                    print(f"[DEBUG] 膨胀蒙版 {expand_pixels} 像素")
            else:
                kernel = np.ones((int(abs(expand_pixels)*2+1), int(abs(expand_pixels)*2+1)), np.uint8)
                mask = cv2.erode(mask, kernel, iterations=1)
                if debug:
                    print(f"[DEBUG] 腐蚀蒙版 {abs(expand_pixels)} 像素")
        
        return mask

    def sort_quadrilateral_points(self, points, debug=False):
        """正确排序四边形顶点：左上、右上、右下、左下"""
        if debug:
            print(f"[DEBUG] 原始顶点: {points}")
        
        # 计算质心
        center = np.mean(points, axis=0)
        
        # 根据相对位置分类点
        top_points = []
        bottom_points = []
        
        for point in points:
            if point[1] < center[1]:  # y坐标小于质心的是上方点
                top_points.append(point)
            else:  # y坐标大于等于质心的是下方点
                bottom_points.append(point)
        
        # 确保我们有正确数量的点
        if len(top_points) < 2:
            # 如果上方点少于2个，按y坐标排序，取最小的2个作为上方点
            sorted_by_y = points[np.argsort(points[:, 1])]
            top_points = sorted_by_y[:2]
            bottom_points = sorted_by_y[2:]
        elif len(bottom_points) < 2:
            # 如果下方点少于2个，按y坐标排序，取最大的2个作为下方点
            sorted_by_y = points[np.argsort(points[:, 1])]
            top_points = sorted_by_y[:2]
            bottom_points = sorted_by_y[2:]
        
        # 转换为numpy数组
        top_points = np.array(top_points)
        bottom_points = np.array(bottom_points)
        
        # 在上方点中，按x坐标排序：左上、右上
        top_sorted = top_points[np.argsort(top_points[:, 0])]
        top_left = top_sorted[0]    # 左上
        top_right = top_sorted[-1]  # 右上
        
        # 在下方点中，按x坐标排序：左下、右下
        bottom_sorted = bottom_points[np.argsort(bottom_points[:, 0])]
        bottom_left = bottom_sorted[0]   # 左下
        bottom_right = bottom_sorted[-1] # 右下
        
        # 按标准顺序排列：左上、右上、右下、左下
        quadrilateral = np.array([
            top_left,     # 左上 [0]
            top_right,    # 右上 [1]
            bottom_right, # 右下 [2]
            bottom_left   # 左下 [3]
        ], dtype=np.float32)
        
        if debug:
            print(f"[DEBUG] 排序后顶点:")
            print(f"[DEBUG]   左上: {quadrilateral[0]}")
            print(f"[DEBUG]   右上: {quadrilateral[1]}")
            print(f"[DEBUG]   右下: {quadrilateral[2]}")
            print(f"[DEBUG]   左下: {quadrilateral[3]}")
            
            # 计算宽高
            width = np.linalg.norm(quadrilateral[1] - quadrilateral[0])
            height = np.linalg.norm(quadrilateral[3] - quadrilateral[0])
            aspect_ratio = width / height if height > 0 else 0
            print(f"[DEBUG] 四边形宽高比: {aspect_ratio:.2f}")
        
        return quadrilateral

    def find_quadrilateral_from_mask(self, mask, debug=False):
        """从mask中提取四边形区域，修复顶点排序"""
        if debug:
            print(f"[DEBUG] Mask shape: {mask.shape}, dtype: {mask.dtype}")
            print(f"[DEBUG] Mask value range: {mask.min()} - {mask.max()}")
            print(f"[DEBUG] White pixels count: {np.sum(mask > 128)}")
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            if debug:
                print("[DEBUG] 错误：未在mask中找到任何轮廓")
            raise ValueError("未在mask中找到任何轮廓")
        
        if debug:
            print(f"[DEBUG] 找到 {len(contours)} 个轮廓")
        
        largest_contour = max(contours, key=cv2.contourArea)
        contour_area = cv2.contourArea(largest_contour)
        
        if debug:
            print(f"[DEBUG] 最大轮廓面积: {contour_area}")
        
        # 对于小区域，使用旋转矩形获得更准确的四边形
        if contour_area < 5000:
            if debug:
                print("[DEBUG] 检测到小蒙版区域，使用旋转矩形优化处理")
            
            rect = cv2.minAreaRect(largest_contour)
            box = cv2.boxPoints(rect)
            box = np.array(box, dtype=np.float32)
            
            # 使用新的排序函数
            quadrilateral = self.sort_quadrilateral_points(box, debug)
            
        else:
            # 大区域使用多边形逼近
            epsilon = 0.02 * cv2.arcLength(largest_contour, True)
            approx = cv2.approxPolyDP(largest_contour, epsilon, True)
            
            while len(approx) > 4 and epsilon < 0.1 * cv2.arcLength(largest_contour, True):
                epsilon *= 1.5
                approx = cv2.approxPolyDP(largest_contour, epsilon, True)
            
            if len(approx) < 4:
                x, y, w, h = cv2.boundingRect(largest_contour)
                points = np.array([[x, y], [x+w, y], [x+w, y+h], [x, y+h]], dtype=np.float32)
            else:
                points = approx.reshape(-1, 2).astype(np.float32)[:4]
            
            # 使用新的排序函数
            quadrilateral = self.sort_quadrilateral_points(points, debug)
        
        return quadrilateral

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
        """基于四边形创建羽化蒙版"""
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

    def perspective_transform_no_black_border(self, target_image, quadrilateral, output_shape, debug=False):
        """透视变换 - 避免黑边"""
        height, width = target_image.shape[:2]
        output_height, output_width = output_shape[:2]
        
        if debug:
            print(f"[DEBUG] 目标图像尺寸: {width}x{height}")
            print(f"[DEBUG] 输出图像尺寸: {output_width}x{output_height}")
        
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
        
        if debug:
            print(f"[DEBUG] 源图像四角点: {src_points}")
            print(f"[DEBUG] 目标四边形顶点: {quadrilateral_clipped}")
        
        perspective_matrix = cv2.getPerspectiveTransform(src_points, quadrilateral_clipped)
        
        if debug:
            print(f"[DEBUG] 透视变换矩阵:\n{perspective_matrix}")
        
        # 透视变换 - 使用BORDER_TRANSPARENT避免黑边
        warped = cv2.warpPerspective(
            target_image, 
            perspective_matrix, 
            (output_width, output_height),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_TRANSPARENT
        )
        
        if debug:
            print(f"[DEBUG] 变换后图像值范围: {warped.min()} - {warped.max()}")
            print(f"[DEBUG] 变换后非零像素数: {np.count_nonzero(warped)}")
            
            # 检查四边形区域
            quad_mask = self.create_quadrilateral_mask(output_shape, quadrilateral_clipped)
            quad_region_pixels = np.sum(warped[quad_mask > 0])
            print(f"[DEBUG] 四边形区域像素总和: {quad_region_pixels}")
            
            if warped.max() == 0:
                print("[DEBUG] ⚠️ 警告：透视变换后图像全为黑色！")
            elif quad_region_pixels == 0:
                print("[DEBUG] ⚠️ 警告：四边形区域内没有内容！")
            else:
                print("[DEBUG] ✅ 透视变换成功，四边形区域有内容")
        
        return warped

    def execute(self, background_image, background_mask, target_image, 
                rotation_angle, feather_inner_expand, feather_outer_expand, 
                mask_expand, enhance_small_mask, debug_mode):
        try:
            print(f"[DEBUG] 开始执行图片插入器，调试模式: {debug_mode}")
            
            # 转换输入格式
            bg_cv2 = self.tensor_to_cv2(background_image)
            mask_cv2 = self.mask_to_cv2(background_mask)
            target_cv2 = self.tensor_to_cv2(target_image)
            
            if debug_mode:
                print(f"[DEBUG] 背景图尺寸: {bg_cv2.shape}")
                print(f"[DEBUG] Mask尺寸: {mask_cv2.shape}")
                print(f"[DEBUG] 目标图尺寸: {target_cv2.shape}")
                print(f"[DEBUG] 目标图值范围: {target_cv2.min()} - {target_cv2.max()}")
            
            # 确保背景图和mask尺寸一致
            bg_height, bg_width = bg_cv2.shape[:2]
            if mask_cv2.shape != (bg_height, bg_width):
                if debug_mode:
                    print(f"[DEBUG] 调整mask尺寸从 {mask_cv2.shape} 到 ({bg_height}, {bg_width})")
                mask_cv2 = cv2.resize(mask_cv2, (bg_width, bg_height), interpolation=cv2.INTER_NEAREST)
            
            # 增强小蒙版
            if enhance_small_mask or mask_expand != 0:
                mask_cv2 = self.enhance_small_mask(mask_cv2, mask_expand, debug_mode)
            
            # 1. 从mask中提取四边形（修复顶点排序）
            quadrilateral = self.find_quadrilateral_from_mask(mask_cv2, debug_mode)
            
            # 2. 旋转目标图像
            if rotation_angle != 0:
                if debug_mode:
                    print(f"[DEBUG] 旋转目标图像 {rotation_angle} 度")
                target_cv2 = self.rotate_image(target_cv2, rotation_angle)
                if debug_mode:
                    print(f"[DEBUG] 旋转后目标图尺寸: {target_cv2.shape}")
            
            # 3. 透视变换（避免黑边）
            warped_target = self.perspective_transform_no_black_border(
                target_cv2, quadrilateral, bg_cv2.shape, debug_mode
            )
            
            # 4. 基于四边形创建羽化蒙版
            if debug_mode:
                print("[DEBUG] 基于检测到的四边形创建羽化蒙版（而非原始mask）")
            
            feather_mask = self.create_feather_mask(
                bg_cv2.shape, quadrilateral, feather_inner_expand, feather_outer_expand
            )
            
            if debug_mode:
                print(f"[DEBUG] 羽化蒙版值范围: {feather_mask.min()} - {feather_mask.max()}")
                quad_area = np.sum(feather_mask > 0)
                print(f"[DEBUG] 四边形羽化区域像素数: {quad_area}")
            
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
            
            if debug_mode:
                print(f"[DEBUG] 融合前 - 背景图值范围: {bg_cv2.min()} - {bg_cv2.max()}")
                print(f"[DEBUG] 融合前 - 变换图值范围: {warped_target.min()} - {warped_target.max()}")
                print(f"[DEBUG] 融合前 - 羽化蒙版值范围: {feather_mask_3d.min()} - {feather_mask_3d.max()}")
            
            # 执行融合
            composited = (bg_cv2.astype(np.float32) * (1 - feather_mask_3d) + 
                         warped_target.astype(np.float32) * feather_mask_3d)
            composited = np.clip(composited, 0, 255).astype(np.uint8)
            
            if debug_mode:
                print(f"[DEBUG] 融合后图像值范围: {composited.min()} - {composited.max()}")
                print("[DEBUG] 图像融合完成")
            
            # 6. 转换回tensor格式
            composited_tensor = self.cv2_to_tensor(composited)
            feather_mask_tensor = torch.from_numpy(feather_mask).unsqueeze(0)
            
            if debug_mode:
                print("[DEBUG] 图片插入器执行成功")
            
            return (composited_tensor, feather_mask_tensor)
            
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