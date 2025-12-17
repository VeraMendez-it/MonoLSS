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
from lib.helpers.visualization_utils import DetectionVisualizer
import cv2

# 初始化可视化器
visualizer = DetectionVisualizer()

# 加载图像
image = cv2.imread('path/to/image.png')

# 准备检测结果 (示例格式)
detections = {
    'bbox': [[100, 100, 200, 200]],  # 2D边界框 [x1, y1, x2, y2]
    'dimensions': [[1.5, 1.8, 4.0]],  # 3D尺寸 [h, w, l]
    'location': [[0, 1.5, 10]],       # 3D位置 [x, y, z]
    'rotation_y': [0.5],               # 旋转角度
    'score': [0.95],                   # 置信度
    'class': ['Car']                   # 类别
}

# 相机标定矩阵
calibration = {
    'P2': [[...]]  # 3x4 投影矩阵
}

# 可视化
result_image = visualizer.draw_3d_box(image, detections, calibration)

# 保存结果
cv2.imwrite('output.png', result_image)
```

### 生成GIF动画

```python
from lib.helpers.visualization_utils import FrameToGifConverter

# 初始化转换器
converter = FrameToGifConverter(
    input_dir='visualization/frames',
    output_path='visualization/demo.gif',
    fps=10
)

# 生成GIF
converter.convert()
print(f"GIF已生成: {converter.output_path}")
```

## 详细使用

### DetectionVisualizer 类

`DetectionVisualizer` 是核心可视化类，负责在图像上绘制3D检测框。

#### 初始化参数

```python
visualizer = DetectionVisualizer(
    class_names=['Car', 'Pedestrian', 'Cyclist'],  # 类别名称列表
    score_threshold=0.3,                            # 置信度阈值
    line_thickness=2,                               # 线条粗细
    font_scale=0.5                                  # 字体大小
)
```

#### 主要方法

**1. draw_3d_box()**

在图像上绘制3D边界框。

```python
result_image = visualizer.draw_3d_box(
    image=img,              # 输入图像 (numpy array)
    detections=det_results, # 检测结果字典
    calibration=calib,      # 相机标定参数
    show_score=True,        # 是否显示置信度
    show_distance=True      # 是否显示距离
)
```

**2. draw_2d_box()**

仅绘制2D边界框（用于对比）。

```python
result_image = visualizer.draw_2d_box(
    image=img,
    detections=det_results,
    show_labels=True
)
```

**3. project_3d_to_2d()**

将3D坐标投影到2D图像平面。

```python
corners_2d = visualizer.project_3d_to_2d(
    corners_3d=corners,  # 3D角点坐标
    P2=calib_matrix      # 投影矩阵
)
```

### FrameToGifConverter 类

将多个图像帧转换为GIF动画。

#### 初始化参数

```python
converter = FrameToGifConverter(
    input_dir='frames/',        # 输入图像目录
    output_path='output.gif',   # 输出GIF路径
    fps=10,                     # 帧率
    loop=0,                     # 循环次数 (0表示无限循环)
    optimize=True               # 是否优化GIF大小
)
```

#### 方法

**convert()**

执行转换操作。

```python
converter.convert()
```

**get_frame_count()**

获取帧数。

```python
num_frames = converter.get_frame_count()
```

### InferencePipeline 类

集成完整的推理和可视化流程。

#### 示例用法

```python
from lib.helpers.visualization_utils import InferencePipeline

pipeline = InferencePipeline(
    model=trained_model,
    data_loader=test_loader,
    output_dir='results/',
    device='cuda'
)

# 运行推理和可视化
pipeline.run(
    visualize=True,
    save_predictions=True,
    generate_gif=True
)
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
| `draw_3d_box()` | image, detections, calibration | numpy.ndarray | 绘制3D边界框 |
| `draw_2d_box()` | image, detections | numpy.ndarray | 绘制2D边界框 |
| `compute_box_3d()` | dimensions, location, rotation_y | numpy.ndarray | 计算3D框角点 |
| `project_3d_to_2d()` | corners_3d, P2 | numpy.ndarray | 3D到2D投影 |

### FrameToGifConverter

#### 方法

| 方法 | 参数 | 返回值 | 说明 |
|------|------|--------|------|
| `convert()` | - | None | 执行帧到GIF转换 |
| `get_frame_count()` | - | int | 获取帧数量 |
| `set_fps()` | fps | None | 设置帧率 |

## 完整示例

### 示例1：KITTI数据集可视化

```python
import os
import cv2
from lib.helpers.visualization_utils import DetectionVisualizer
from lib.datasets.kitti import KITTI

# 1. 设置路径
data_dir = 'data/KITTI/object'
output_dir = 'visualization/kitti_results'
os.makedirs(output_dir, exist_ok=True)

# 2. 初始化数据集
dataset = KITTI(
    root_dir=data_dir,
    split='val'
)

# 3. 初始化可视化器
visualizer = DetectionVisualizer(
    class_names=['Car', 'Pedestrian', 'Cyclist'],
    score_threshold=0.5,
    line_thickness=3
)

# 4. 批量可视化
for idx in range(len(dataset)):
    # 获取数据
    image, calibration, _ = dataset[idx]
    
    # 运行检测 (这里需要您的模型)
    # detections = model.predict(image)
    
    # 可视化
    result = visualizer.draw_3d_box(
        image=image,
        detections=detections,
        calibration=calibration
    )
    
    # 保存
    output_path = os.path.join(output_dir, f'{idx:06d}.png')
    cv2.imwrite(output_path, result)
    
    print(f'Processed {idx+1}/{len(dataset)}')
```

### 示例2：创建检测结果GIF

```python
from lib.helpers.visualization_utils import (
    DetectionVisualizer,
    FrameToGifConverter
)

# 1. 批量生成可视化图像
visualizer = DetectionVisualizer()
frame_dir = 'temp_frames/'
os.makedirs(frame_dir, exist_ok=True)

for i, (img, det) in enumerate(detection_results):
    vis_img = visualizer.draw_3d_box(img, det, calib)
    cv2.imwrite(f'{frame_dir}/{i:04d}.png', vis_img)

# 2. 转换为GIF
converter = FrameToGifConverter(
    input_dir=frame_dir,
    output_path='detection_demo.gif',
    fps=15,
    optimize=True
)
converter.convert()

# 3. 清理临时文件
import shutil
shutil.rmtree(frame_dir)
```

### 示例3：实时可视化

```python
import torch
from lib.helpers.visualization_utils import DetectionVisualizer

# 加载模型
model = torch.load('checkpoints/best_model.pth')
model.eval()

# 初始化可视化器
visualizer = DetectionVisualizer(score_threshold=0.3)

# 视频处理
cap = cv2.VideoCapture('input_video.mp4')
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_video.mp4', fourcc, 30.0, (1920, 1080))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # 推理
    with torch.no_grad():
        detections = model(frame)
    
    # 可视化
    vis_frame = visualizer.draw_3d_box(frame, detections, calibration)
    
    # 写入输出
    out.write(vis_frame)
    
    # 显示
    cv2.imshow('Detection', vis_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
```

## 常见问题

### Q1: 如何获取相机标定矩阵？

**A:** 对于KITTI数据集，标定文件位于 `data/KITTI/object/training/calib/` 目录。读取方法：

```python
def read_calib_file(filepath):
    """读取KITTI标定文件"""
    data = {}
    with open(filepath, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            data[key] = np.array([float(x) for x in value.split()])
    
    # P2是左侧彩色相机的投影矩阵
    P2 = data['P2'].reshape(3, 4)
    return {'P2': P2}

# 使用
calib = read_calib_file('data/KITTI/object/training/calib/000000.txt')
```

### Q2: 检测结果格式是什么？

**A:** 标准检测结果字典格式：

```python
detections = {
    'bbox': np.array([[x1, y1, x2, y2], ...]),      # 2D框 (N, 4)
    'dimensions': np.array([[h, w, l], ...]),        # 3D尺寸 (N, 3)
    'location': np.array([[x, y, z], ...]),          # 3D位置 (N, 3)
    'rotation_y': np.array([ry, ...]),               # 旋转角 (N,)
    'score': np.array([conf, ...]),                  # 置信度 (N,)
    'class': ['Car', 'Pedestrian', ...]              # 类别 (N,)
}
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
class CustomVisualizer(DetectionVisualizer):
    def __init__(self):
        super().__init__()
        self.colors = {
            'Car': (0, 255, 0),        # 绿色
            'Pedestrian': (255, 0, 0),  # 蓝色
            'Cyclist': (0, 0, 255)      # 红色
        }
    
    def get_color(self, class_name):
        return self.colors.get(class_name, (255, 255, 255))
```

### Q5: GIF文件太大怎么办？

**A:** 优化建议：

```python
converter = FrameToGifConverter(
    input_dir='frames/',
    output_path='output.gif',
    fps=10,              # 降低帧率
    optimize=True,       # 启用优化
    quality=80           # 降低质量 (1-100)
)

# 或者减少帧数
converter.convert(step=2)  # 每隔一帧取一帧
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
