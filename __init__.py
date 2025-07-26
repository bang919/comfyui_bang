"""
ComfyUI 透视图插入节点插件
Perspective Image Insert Node for ComfyUI

这个插件实现了将目标图片通过透视变换融合到背景图中的功能，
支持从mask自动提取四边形区域、图像旋转、羽化边缘处理等功能。
"""

from .perspective_insert_node import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

# 插件信息
WEB_DIRECTORY = "./js"
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

# 版本信息
__version__ = "2.1.0"
__author__ = "ComfyUI Perspective Insert Plugin"
__description__ = "图片插入器 - 专业的透视图像插入工具，专门优化小区域处理" 