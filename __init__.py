"""
ComfyUI 透视图插入节点插件
Perspective Image Insert Node for ComfyUI

这个插件实现了将目标图片通过透视变换融合到背景图中的功能，
支持从mask自动提取四边形区域、图像旋转、羽化边缘处理等功能。
"""

from .perspective_insert_node_debug import NODE_CLASS_MAPPINGS as DEBUG_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as DEBUG_DISPLAY_MAPPINGS
from .perspective_insert_node_fix import NODE_CLASS_MAPPINGS as FIX_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as FIX_DISPLAY_MAPPINGS

# 合并节点映射
NODE_CLASS_MAPPINGS = {**DEBUG_MAPPINGS, **FIX_MAPPINGS}
NODE_DISPLAY_NAME_MAPPINGS = {**DEBUG_DISPLAY_MAPPINGS, **FIX_DISPLAY_MAPPINGS}

# 插件信息
WEB_DIRECTORY = "./js"
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

# 版本信息
__version__ = "2.0.0"
__author__ = "ComfyUI Perspective Insert Plugin"
__description__ = "透视图插入节点 - 专业版：包含调试版本和问题修复版本，专门优化小区域透视变换" 