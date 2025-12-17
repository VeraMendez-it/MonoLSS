# MonoLSS 可视化指南

本指南详细介绍了如何使用 MonoLSS 的可视化工具来展示3D目标检测结果。

## 目录

- [功能特性](#功能特性)
- [安装依赖](#安装依赖)
- [快速开始](#快速开始)
- [详细使用](#详细使用)
- [API 文档](#api-文档)
- [完整示例](#完整示例)
- [常见问题](#常见问题)
- [性能优化建议](#性能优化建议)
- [输出文件说明](#输出文件说明)

## 功能特性

MonoLSS 可视化工具提供以下核心功能：

- **3D边界框可视化**：在图像上绘制3D检测框
- **检测信息标注**：显示类别、置信度、距离等信息
- **批量处理**：支持对整个数据集进行批量可视化
- **GIF动画生成**：将多帧检测结果转换为GIF动画
- **完整推理流程**：集成数据加载、推理和可视化的端到端pipeline
- **自定义配置**：灵活的可视化参数设置

## 安装依赖

在使用可视化工具之前，请确保安装以下依赖包：

```bash
pip install opencv-python
pip install imageio
pip install numpy
pip install torch
pip install tqdm
```

或使用 requirements.txt 一次性安装：

```bash
pip install -r requirements.txt
```

### 依赖包说明

- **opencv-python**: 用于图像处理和绘制
- **imageio**: 用于GIF动画生成
- **numpy**: 数值计算
- **torch**: PyTorch深度学习框架
- **tqdm**: 进度条显示

## 快速开始

### 基础可视化示例

```python
from tools.visualization import DetectionVisualizer, load_kitti_calib
import cv2
import numpy as np

# 初始化可视化器
visualizer = DetectionVisualizer()

# 加载图像
image = cv2.imread('path/to/image.png')

# 加载KITTI标定文件
calib = load_kitti_calib('path/to/calib.txt')
P2 = calib['P2']  # 3x4 投影矩阵

# 准备单个检测结果
detection = {
    'class': 'Car',                    # 类别名称
    'dimensions': [1.5, 1.8, 4.0],     # 3D尺寸 [h, w, l]
    'location': [2.0, 1.5, 10.0],      # 3D位置 [x, y, z]
    'rotation_y': 0.5,                 # 旋转角度
    'score': 0.95                      # 置信度
}

# 可视化单个检测
result_image = visualizer.draw_3d_bbox(image, detection, P2)

# 或者可视化多个检测
detections = [detection]  # 检测列表
result_image = visualizer.visualize_detections(image, detections, P2)

# 保存结果
cv2.imwrite('output.png', result_image)
```

### 生成GIF动画

```python
from tools.gif_generator import FrameToGifConverter
from pathlib import Path

# 初始化转换器
converter = FrameToGifConverter(
    frame_dir='visualization/frames',
    output_path='visualization/demo.gif',
    fps=10
)

# 生成GIF
gif_path = converter.create_gif(duration=100)
print(f"GIF已生成: {gif_path}")

# 也可以生成MP4视频
video_path = converter.create_video(
    output_video='visualization/demo.mp4',
    fps=10
)
print(f"视频已生成: {video_path}")
```

## 详细使用

### DetectionVisualizer 类

`DetectionVisualizer` 是核心可视化类，负责在图像上绘制3D检测框。

#### 初始化参数

```python
from tools.visualization import DetectionVisualizer

visualizer = DetectionVisualizer(
    class_names=['Car', 'Pedestrian', 'Cyclist']  # 类别名称列表
)
```

#### 主要方法

**1. draw_3d_bbox()**

在图像上绘制单个3D边界框。

```python
from tools.visualization import DetectionVisualizer, load_kitti_calib
import cv2

visualizer = DetectionVisualizer()

# 加载标定
calib = load_kitti_calib('path/to/calib.txt')
P2 = calib['P2']

# 单个检测
detection = {
    'class': 'Car',
    'dimensions': [1.5, 1.8, 4.0],  # h, w, l
    'location': [2.0, 1.5, 10.0],   # x, y, z
    'rotation_y': 0.5,
    'score': 0.95
}

result_image = visualizer.draw_3d_bbox(
    image=img,
    detection=detection,
    P2=P2,
    thickness=2,
    show_info=True
)
```

**2. visualize_detections()**

在图像上绘制多个3D边界框。

```python
# 多个检测
detections = [detection1, detection2, detection3]

result_image = visualizer.visualize_detections(
    image=img,
    detections=detections,
    P2=P2,
    thickness=2,
    show_info=True
)
```

### FrameToGifConverter 类

将多个图像帧转换为GIF动画。

#### 初始化参数

```python
from tools.gif_generator import FrameToGifConverter

converter = FrameToGifConverter(
    frame_dir='frames/',           # 输入图像目录
    output_path='output.gif',      # 输出GIF路径
    fps=10                         # 帧率
)
```

#### 方法

**create_gif()**

创建GIF动画。

```python
gif_path = converter.create_gif(duration=100)
```

**create_video()**

创建MP4视频。

```python
video_path = converter.create_video(
    output_video='output.mp4',
    fps=10
)
```

### InferencePipeline 类

集成完整的推理和可视化流程。

#### 示例用法

```python
from tools.inference_pipeline import InferencePipeline

pipeline = InferencePipeline(
    model=None,  # 可选，传入训练好的模型
    output_dir='results/',
    device='cuda',
    class_names=['Car', 'Pedestrian', 'Cyclist']
)

# 运行推理和可视化
output_path = pipeline.inference_and_visualize(
    image_dir='data/kitti/image_2',
    calib_dir='data/kitti/calib',
    save_gif=True,
    show_progress=True
)

print(f"Results saved to: {output_path}")
```

## API 文档

### DetectionVisualizer

#### 属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `class_names` | list | 检测类别名称 |
| `score_threshold` | float | 最小显示置信度 |
| `line_thickness` | int | 边界框线条粗细 |
| `font_scale` | float | 文字大小比例 |

#### 方法

| 方法 | 参数 | 返回值 | 说明 |
|------|------|--------|------|
| `draw_3d_bbox()` | image, detection, P2 | numpy.ndarray | 绘制单个3D边界框 |
| `visualize_detections()` | image, detections, P2 | numpy.ndarray | 绘制多个3D边界框 |
| `compute_box_3d()` | dimensions, location, rotation_y | numpy.ndarray | 计算3D框角点 |
| `project_3d_to_2d()` | corners_3d, P2 | tuple | 3D到2D投影 |
| `save_frame()` | image, detections, P2, output_path | bool | 保存可视化结果 |

### FrameToGifConverter

#### 方法

| 方法 | 参数 | 返回值 | 说明 |
|------|------|--------|------|
| `create_gif()` | duration | str | 创建GIF动画 |
| `create_video()` | output_video, fps | None | 创建MP4视频 |

## 完整示例

### 示例1：KITTI数据集可视化

```python
import os
import cv2
import numpy as np
from pathlib import Path
from tools.visualization import DetectionVisualizer, load_kitti_calib

# 1. 设置路径
data_dir = Path('data/KITTI/object/training')
image_dir = data_dir / 'image_2'
calib_dir = data_dir / 'calib'
output_dir = Path('visualization/kitti_results')
output_dir.mkdir(parents=True, exist_ok=True)

# 2. 初始化可视化器
visualizer = DetectionVisualizer(
    class_names=['Car', 'Pedestrian', 'Cyclist']
)

# 3. 批量可视化
image_files = sorted(image_dir.glob('*.png'))

for img_path in image_files:
    # 获取图像ID
    image_id = img_path.stem
    
    # 加载图像
    image = cv2.imread(str(img_path))
    
    # 加载标定
    calib_path = calib_dir / f"{image_id}.txt"
    calib = load_kitti_calib(str(calib_path))
    P2 = calib['P2']
    
    # 运行检测 (需要您的模型)
    # detections = model.predict(image)
    
    # 示例检测 (演示用)
    detections = [
        {
            'class': 'Car',
            'dimensions': [1.5, 1.8, 4.0],
            'location': [2.0, 1.5, 10.0],
            'rotation_y': 0.5,
            'score': 0.95
        }
    ]
    
    # 可视化
    result = visualizer.visualize_detections(image, detections, P2)
    
    # 保存
    output_path = output_dir / f'{image_id}.png'
    cv2.imwrite(str(output_path), result)
    
    print(f'Processed {image_id}')

print(f"All results saved to {output_dir}")
```

### 示例2：创建检测结果GIF

```python
import os
import cv2
from pathlib import Path
from tools.visualization import DetectionVisualizer, load_kitti_calib
from tools.gif_generator import FrameToGifConverter

# 1. 批量生成可视化图像
visualizer = DetectionVisualizer()
frame_dir = Path('temp_frames/')
frame_dir.mkdir(parents=True, exist_ok=True)

# 假设有一系列检测结果
image_dir = Path('data/kitti/image_2')
calib_dir = Path('data/kitti/calib')

image_files = sorted(image_dir.glob('*.png'))[:20]  # 处理前20张

for i, img_path in enumerate(image_files):
    # 加载图像和标定
    image = cv2.imread(str(img_path))
    calib_path = calib_dir / f"{img_path.stem}.txt"
    calib = load_kitti_calib(str(calib_path))
    P2 = calib['P2']
    
    # 示例检测
    detections = [
        {
            'class': 'Car',
            'dimensions': [1.5, 1.8, 4.0],
            'location': [2.0, 1.5, 10.0 + i * 2],  # 车辆逐渐远离
            'rotation_y': 0.5,
            'score': 0.95
        }
    ]
    
    # 可视化
    vis_img = visualizer.visualize_detections(image, detections, P2)
    
    # 保存帧
    cv2.imwrite(str(frame_dir / f'frame_{i:04d}.png'), vis_img)

# 2. 转换为GIF
converter = FrameToGifConverter(
    frame_dir=frame_dir,
    output_path='detection_demo.gif',
    fps=5
)
converter.create_gif(duration=200)

# 3. 也可以创建视频
converter.create_video(output_video='detection_demo.mp4', fps=5)

print("GIF and video created!")

# 4. 清理临时文件（可选）
import shutil
# shutil.rmtree(frame_dir)
```

### 示例3：完整推理流程

```python
import torch
from pathlib import Path
from tools.inference_pipeline import InferencePipeline
from tools.visualization import DetectionVisualizer

# 1. 设置路径
data_dir = Path('data/kitti/testing')
image_dir = data_dir / 'image_2'
calib_dir = data_dir / 'calib'
output_dir = Path('results')

# 2. 创建推理流程（不使用模型，仅演示）
pipeline = InferencePipeline(
    model=None,  # 如果有模型: model = torch.load('model.pth')
    output_dir=output_dir,
    device='cuda',
    class_names=['Car', 'Pedestrian', 'Cyclist']
)

# 3. 运行批量推理和可视化
result_path = pipeline.inference_and_visualize(
    image_dir=image_dir,
    calib_dir=calib_dir,
    save_gif=True,
    show_progress=True
)

print(f"Results saved to: {result_path}")
print(f"  - Frames: {result_path / 'frames'}")
print(f"  - GIF: {result_path / 'inference_results.gif'}")
print(f"  - Video: {result_path / 'inference_results.mp4'}")
```

## 常见问题

### Q1: 如何获取相机标定矩阵？

**A:** 对于KITTI数据集，标定文件位于 `data/KITTI/object/training/calib/` 目录。读取方法：

```python
from tools.visualization import load_kitti_calib

# 加载KITTI标定文件
calib = load_kitti_calib('data/KITTI/object/training/calib/000000.txt')

# 获取P2投影矩阵 (3x4)
P2 = calib['P2']

print(f"P2 shape: {P2.shape}")  # (3, 4)
print(f"P2:\n{P2}")
```

### Q2: 检测结果格式是什么？

**A:** 标准检测结果字典格式（单个检测）：

```python
detection = {
    'class': 'Car',                    # 类别名称 (字符串)
    'dimensions': [1.5, 1.8, 4.0],     # 3D尺寸 [h, w, l] (米)
    'location': [2.0, 1.5, 10.0],      # 3D位置 [x, y, z] (相机坐标系)
    'rotation_y': 0.5,                 # 旋转角 (弧度, -π到π)
    'score': 0.95                      # 置信度 (可选)
}

# 多个检测使用列表
detections = [detection1, detection2, detection3]
```

### Q3: 为什么3D框显示不正确？

**A:** 常见原因：

1. **标定矩阵错误**：确保使用正确的P2矩阵
2. **坐标系不匹配**：检查位置坐标是否在相机坐标系下
3. **图像尺寸**：确认图像未被resize，或相应调整标定矩阵
4. **角度范围**：rotation_y应在 [-π, π] 范围内

### Q4: 如何调整可视化样式？

**A:** 自定义颜色和样式：

```python
from tools.visualization import DetectionVisualizer

# 使用默认颜色
visualizer = DetectionVisualizer()

# DetectionVisualizer.CLASS_COLORS 定义了默认颜色
# 可以在绘制时调整线条粗细
result = visualizer.draw_3d_bbox(
    image, detection, P2,
    thickness=3,      # 线条粗细
    show_info=True    # 显示标签信息
)
```

### Q5: GIF文件太大怎么办？

**A:** 优化建议：

```python
from tools.gif_generator import FrameToGifConverter

converter = FrameToGifConverter(
    frame_dir='frames/',
    output_path='output.gif',
    fps=5              # 降低帧率减小文件大小
)

# 创建GIF时使用较大的duration值
converter.create_gif(duration=200)  # 每帧持续200ms

# 或者只处理部分帧（在生成帧时跳过）
# 例如，每隔2帧保存一次
```

## 性能优化建议

### 1. 批量处理优化

```python
# 使用多进程加速
from multiprocessing import Pool

def process_image(args):
    idx, image_path = args
    image = cv2.imread(image_path)
    # ... 处理逻辑
    return result

with Pool(processes=8) as pool:
    results = pool.map(process_image, enumerate(image_paths))
```

### 2. GPU加速

```python
# 将图像数据移到GPU
import torch

image_tensor = torch.from_numpy(image).cuda()
# 在GPU上进行处理
```

### 3. 内存优化

```python
# 使用生成器避免一次性加载所有图像
def image_generator(image_dir):
    for filename in sorted(os.listdir(image_dir)):
        yield cv2.imread(os.path.join(image_dir, filename))

for img in image_generator('frames/'):
    # 处理单张图像
    process(img)
```

### 4. 可视化加速

```python
# 预计算不变的部分
visualizer = DetectionVisualizer()
visualizer.precompute_colors()  # 预计算颜色映射

# 使用OpenCV的GPU模块
import cv2.cuda as cuda
gpu_frame = cuda.GpuMat()
gpu_frame.upload(frame)
# GPU上的图像处理
```

## 输出文件说明

### 可视化图像

- **位置**: `{output_dir}/{index}.png`
- **格式**: PNG (无损压缩)
- **内容**: 
  - 原始图像
  - 3D边界框 (绿色线条)
  - 2D边界框 (蓝色虚线)
  - 标签文本 (类别、置信度、距离)

### GIF动画

- **位置**: `{output_dir}/animation.gif`
- **格式**: GIF
- **参数**:
  - 帧率: 默认10 FPS
  - 循环: 无限循环
  - 优化: 启用

### 检测结果文件

```
output_dir/
├── images/           # 可视化图像
│   ├── 000000.png
│   ├── 000001.png
│   └── ...
├── predictions/      # 检测结果
│   ├── 000000.txt
│   ├── 000001.txt
│   └── ...
└── demo.gif         # GIF动画
```

### 结果文本格式 (KITTI格式)

```
Car 0.00 0 -1.57 614.24 181.78 727.31 284.77 1.57 1.73 4.15 1.84 1.47 8.41 -1.56 0.95
```

字段说明:
- 类别 (Car)
- 截断程度 (0.00)
- 遮挡程度 (0)
- 观察角度 (-1.57)
- 2D框 (614.24 181.78 727.31 284.77)
- 3D尺寸 (1.57 1.73 4.15)
- 3D位置 (1.84 1.47 8.41)
- 旋转角度 (-1.56)
- 置信度 (0.95)

---

## 贡献与反馈

如果您在使用过程中遇到问题或有改进建议，欢迎提交 Issue 或 Pull Request。

## 许可证

本项目遵循 MIT 许可证。

## 相关资源

- [KITTI数据集](http://www.cvlibs.net/datasets/kitti/)
- [MonoLSS论文](https://arxiv.org/abs/2104.14445)
- [OpenCV文档](https://docs.opencv.org/)

---

**最后更新时间**: 2025-12-17
