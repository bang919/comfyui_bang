# ComfyUI 透视图插入节点

专业的ComfyUI自定义节点，用于将目标图片通过透视变换融合到背景图中，**专门优化小区域蒙版处理**（如广告牌植入）。

## 🎯 核心功能

- ✅ 从背景mask自动提取四边形区域
- ✅ 目标图像透视变换适配mask区域  
- ✅ 专门优化小区域处理（广告牌、标识等）
- ✅ 支持图像旋转（-180°到180°）
- ✅ 羽化边缘处理（内扩/外扩调节）
- ✅ 输出合成图像和标准四边形羽化蒙版
- ✅ 支持与Flux等工具配合进行inpainting

## 📦 安装方法

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/bang919/comfyui_bang.git
cd comfyui_bang
pip install -r requirements.txt
```

重启ComfyUI后，节点会出现在 `image/transform` 分类下。

## 🚀 推荐使用

### **节点选择**

插件提供两个版本：

1. **"透视区域插入（问题修复版）"** ⭐ **推荐使用**
   - 最新修复版本，解决所有已知问题
   - 专门优化小区域透视变换
   - 集成所有高级功能

2. **"透视区域插入（调试版）"**
   - 详细调试信息输出
   - 用于问题排查和参数调试

### **测试工作流**

使用 `example_workflow.json` - 专门针对小区域广告牌插入的完整工作流。

## 🎛️ 节点参数说明

### 输入参数
- **background_image** (IMAGE) - 背景图像
- **background_mask** (MASK) - 插入区域的蒙版（白色=插入区域）
- **target_image** (IMAGE) - 要插入的目标图像  
- **rotation_angle** (FLOAT) - 目标图旋转角度 (-180~180)
- **feather_inner_expand** (FLOAT) - 羽化边缘内扩像素数 (0~100)
- **feather_outer_expand** (FLOAT) - 羽化边缘外扩像素数 (0~100)
- **mask_expand** (FLOAT) - 蒙版膨胀/腐蚀调节 (-10~10)
- **enhance_small_mask** (BOOL) - 启用小区域增强
- **debug_mode** (BOOL) - 显示详细调试信息

### 输出参数
- **composited_image** (IMAGE) - 融合后的最终图像
- **feather_mask** (MASK) - 标准四边形羽化蒙版（可用于后续inpainting）

## 💡 使用场景

### 🎪 **广告植入**
- 体育场广告牌替换
- 户外广告植入
- 商品包装图案替换

### 🏢 **标识植入**
- 建筑物标识插入
- 店面招牌替换
- 背景标识植入

### 🎨 **创意合成**
- 透视图片合成
- 场景元素插入
- AI修复配合（使用输出的羽化蒙版）

## 🔧 推荐参数配置

### **小区域广告牌场景**
```
rotation_angle: 0.0
feather_inner_expand: 0.0  
feather_outer_expand: 2.0
mask_expand: 1.0
enhance_small_mask: true
debug_mode: true (首次使用)
```

### **大区域图片插入**
```
rotation_angle: 0.0
feather_inner_expand: 2.0  
feather_outer_expand: 5.0
mask_expand: 0.0
enhance_small_mask: false
debug_mode: false
```

## 🎯 蒙版制作要求

1. **格式**: 黑白PNG图像
2. **颜色**: 白色区域 = 插入区域，黑色区域 = 背景
3. **形状**: 白色区域应大致为四边形
4. **尺寸**: 与背景图像同尺寸（或节点会自动调整）

## 🔍 问题排查

### **图像为黑色**
- 使用"问题修复版"节点
- 启用`debug_mode`查看详细信息
- 检查蒙版是否正确（白色区域代表插入位置）

### **透视效果不理想**  
- 调整`rotation_angle`参数
- 优化蒙版的四边形形状
- 使用`mask_expand`微调蒙版大小

### **边缘过渡不自然**
- 调整`feather_inner_expand`和`feather_outer_expand`参数
- 对于小区域，建议使用较小的羽化值

## 📊 技术规格

- **Python**: >= 3.8
- **OpenCV**: >= 4.8.0
- **NumPy**: >= 1.21.0  
- **PyTorch**: >= 1.13.0
- **Pillow**: >= 9.0.0

## 📝 版本历史

- **v2.0.0**: 专业版 - 清理冗余代码，专注小区域优化
- **v1.2.0**: 修复黑色输出问题
- **v1.1.0**: 新增小区域优化版本
- **v1.0.1**: 添加调试版本
- **v1.0.0**: 初始版本

---

## 🌟 快速开始

1. **安装插件** → `git clone` + `pip install`
2. **重启ComfyUI** → 查看新节点
3. **导入工作流** → 使用 `example_workflow.json`
4. **上传素材** → 背景图、蒙版图、目标图
5. **使用修复版节点** → 获得完美效果

**专为小区域透视变换优化，让你的广告植入更加自然！** 🎯✨ 