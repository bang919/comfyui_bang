"""
ComfyUI 透视图插入节点插件
Perspective Image Insert Node for ComfyUI

这个插件实现了将目标图片通过透视变换融合到背景图中的功能，
支持从mask自动提取四边形区域、图像旋转、羽化边缘处理等功能。
"""

from .perspective_insert_node import NODE_CLASS_MAPPINGS as ORIGINAL_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as ORIGINAL_DISPLAY_MAPPINGS
from .perspective_insert_node_debug import NODE_CLASS_MAPPINGS as DEBUG_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as DEBUG_DISPLAY_MAPPINGS
from .perspective_insert_node_optimized import NODE_CLASS_MAPPINGS as OPTIMIZED_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as OPTIMIZED_DISPLAY_MAPPINGS
from .perspective_insert_node_fix import NODE_CLASS_MAPPINGS as FIX_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as FIX_DISPLAY_MAPPINGS

# 合并所有节点映射
NODE_CLASS_MAPPINGS = {**ORIGINAL_MAPPINGS, **DEBUG_MAPPINGS, **OPTIMIZED_MAPPINGS, **FIX_MAPPINGS}
NODE_DISPLAY_NAME_MAPPINGS = {**ORIGINAL_DISPLAY_MAPPINGS, **DEBUG_DISPLAY_MAPPINGS, **OPTIMIZED_DISPLAY_MAPPINGS, **FIX_DISPLAY_MAPPINGS}

# 插件信息
WEB_DIRECTORY = "./js"
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

# 版本信息
__version__ = "1.2.0"
__author__ = "ComfyUI Perspective Insert Plugin"
__description__ = "透视图插入节点 - 支持四边形区域透视变换和羽化边缘处理（包含调试版本、小区域优化版本和问题修复版本）" 