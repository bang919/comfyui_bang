# ComfyUI 透视图插入节点

这是一个ComfyUI自定义节点，用于将目标图片通过透视变换融合到背景图中，支持从mask自动提取四边形区域和羽化边缘处理。

## 功能特性

- ✅ 从背景mask自动提取四边形区域
- ✅ 目标图像透视变换适配mask区域  
- ✅ 支持图像旋转（-180°到180°）
- ✅ 羽化边缘处理（内扩/外扩调节）
- ✅ 输出合成图像和羽化蒙版
- ✅ 支持与Flux等工具配合进行inpainting

## 安装方法

1. **下载插件文件**
   ```bash
   cd ComfyUI/custom_nodes
   git clone [你的仓库地址] comfyui-perspective-insert
   # 或者直接将所有文件复制到 ComfyUI/custom_nodes/comfyui-perspective-insert/ 目录
   ```

2. **安装依赖**
   ```bash
   cd comfyui-perspective-insert
   pip install -r requirements.txt
   ```

3. **重启ComfyUI**
   重启ComfyUI后，节点会出现在 `image/transform` 分类下

## 测试使用

### 准备测试素材

你需要准备以下三个文件：

1. **background.jpg** - 背景图像
   - 任意尺寸的图片，作为插入目标的背景

2. **mask.png** - 蒙版图像  
   - 黑白图像，白色区域表示要插入目标图的区域
   - 白色区域应该大致是四边形形状
   - 建议使用PNG格式以保持清晰的边缘

3. **target.jpg** - 目标图像
   - 要被插入到背景中的图片
   - 会通过透视变换适配mask区域

### 使用测试工作流

1. **加载工作流**
   - 在ComfyUI中点击 "Load" 按钮
   - 选择 `example.json` 文件

2. **上传测试图片**
   - 在"加载背景图像"节点中上传你的背景图
   - 在"加载蒙版图像"节点中上传你的mask图
   - 在"加载目标图像"节点中上传你的目标图

3. **调整参数**（可选）
   - `rotation_angle`: 目标图旋转角度（默认0°）
   - `feather_inner_expand`: 羽化内扩像素数（默认2.0）
   - `feather_outer_expand`: 羽化外扩像素数（默认8.0）

4. **执行工作流**
   - 点击 "Queue Prompt" 执行
   - 查看预览结果和保存的文件

## 节点参数说明

### 输入参数
- **background_image** (IMAGE) - 背景图像
- **background_mask** (MASK) - 插入区域的蒙版
- **target_image** (IMAGE) - 要插入的目标图像  
- **rotation_angle** (FLOAT) - 目标图旋转角度 (-180~180)
- **feather_inner_expand** (FLOAT) - 羽化边缘内扩像素数 (0~100)
- **feather_outer_expand** (FLOAT) - 羽化边缘外扩像素数 (0~100)

### 输出参数
- **composited_image** (IMAGE) - 融合后的最终图像
- **feather_mask** (MASK) - 羽化边缘蒙版（可用于后续inpainting）

## 使用示例场景

1. **商品广告合成**
   - 将商品图片贴入广告背景画面中

2. **标识植入**
   - 将标志图片插入墙面或其他场景

3. **场景合成**
   - 批量合成带有插画的真实场景图

4. **AI修复配合**
   - 与Flux等工具配合，使用输出的feather_mask进行缺失区域填补

## 常见问题

**Q: mask中没有检测到四边形怎么办？**
A: 节点会自动使用边界矩形作为fallback，或者你可以调整mask使白色区域更接近四边形。

**Q: 透视变换效果不理想？**  
A: 可以尝试调整rotation_angle参数，或者优化mask的四边形形状。

**Q: 边缘过渡不自然？**
A: 调整feather_inner_expand和feather_outer_expand参数来优化羽化效果。

## 技术依赖

- Python >= 3.8
- OpenCV >= 4.8.0
- NumPy >= 1.21.0  
- PyTorch >= 1.13.0
- Pillow >= 9.0.0

## 版本信息

- 版本: 1.0.0
- 兼容: ComfyUI
- 分类: image/transform

## 许可证

[根据你的需要添加许可证信息] 