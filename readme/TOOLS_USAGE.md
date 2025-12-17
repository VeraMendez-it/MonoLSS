# MonoLSS Visualization and Evaluation Tools

This document describes the visualization and evaluation tools added to the MonoLSS project.

## 📚 Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Modules](#modules)
- [Command-Line Tools](#command-line-tools)
- [Examples](#examples)
- [API Reference](#api-reference)

## Overview

This toolkit provides comprehensive visualization and evaluation capabilities for MonoLSS 3D object detection:

- **3D Bounding Box Visualization**: Draw 3D detection boxes on images with KITTI calibration
- **GIF/Video Generation**: Create animations from detection sequences
- **Inference Pipeline**: Batch processing with progress tracking
- **Metrics Calculation**: Complete KITTI evaluation with AP40/AP11
- **End-to-End Evaluation**: Full pipeline from inference to reporting

## Installation

### Required Dependencies

```bash
pip install numpy opencv-python imageio tqdm
```

### Optional Dependencies

For full functionality (inference and CUDA-accelerated evaluation):

```bash
pip install torch torchvision numba
```

**Note:** The tools work without PyTorch/CUDA but with limited functionality.

## Quick Start

### 1. Basic Visualization

```python
from tools.visualization import DetectionVisualizer, load_kitti_calib
import cv2
import numpy as np

# Load image and calibration
image = cv2.imread('image.png')
calib = load_kitti_calib('calib.txt')
P2 = calib['P2']

# Create detection
detection = {
    'class': 'Car',
    'dimensions': [1.5, 1.8, 4.0],  # h, w, l
    'location': [2.0, 1.5, 10.0],   # x, y, z
    'rotation_y': 0.5,
    'score': 0.95
}

# Visualize
visualizer = DetectionVisualizer()
result = visualizer.draw_3d_bbox(image, detection, P2)
cv2.imwrite('output.png', result)
```

### 2. Calculate Metrics

```bash
python tools/metrics_calculator.py \
    --gt_dir data/kitti/labels \
    --pred_dir results/predictions \
    --classes Car \
    --ap_mode 40 \
    --output metrics.json
```

### 3. Run Complete Evaluation

```bash
python tools/run_evaluation.py \
    --data data/kitti/training \
    --output results/eval \
    --visualize \
    --classes Car Pedestrian Cyclist
```

### 4. Run Examples

```bash
python tools/example_usage.py
```

## Modules

### `tools/visualization.py`

Core visualization module for 3D bounding boxes.

**Key Classes:**
- `DetectionVisualizer`: Main class for 3D bbox visualization

**Key Functions:**
- `load_kitti_calib(path)`: Load KITTI calibration files

**Example:**
```python
from tools.visualization import DetectionVisualizer

visualizer = DetectionVisualizer(class_names=['Car', 'Pedestrian'])
vis_image = visualizer.draw_3d_bbox(image, detection, P2)
```

### `tools/gif_generator.py`

Generate GIF animations and videos from image sequences.

**Example:**
```python
from tools.gif_generator import FrameToGifConverter

converter = FrameToGifConverter(
    frame_dir='frames/',
    output_path='animation.gif',
    fps=10
)
converter.create_gif(duration=100)
converter.create_video('animation.mp4', fps=10)
```

### `tools/inference_pipeline.py`

Batch inference and visualization pipeline.

**Example:**
```python
from tools.inference_pipeline import InferencePipeline

pipeline = InferencePipeline(
    model=None,  # Your model here
    output_dir='results/',
    device='cuda'
)
pipeline.inference_and_visualize(
    image_dir='data/images',
    calib_dir='data/calib',
    save_gif=True
)
```

### `tools/metrics_calculator.py`

Complete KITTI evaluation metrics calculation.

**Example:**
```python
from tools.metrics_calculator import MetricsCalculator

calculator = MetricsCalculator(
    gt_dir='data/labels',
    pred_dir='results/predictions',
    class_names=['Car'],
    ap_mode=40
)
metrics = calculator.calculate_metrics()
calculator.print_results(metrics)
calculator.save_results(metrics, 'metrics.json')
```

### `tools/run_evaluation.py`

End-to-end evaluation pipeline.

**Example:**
```python
from tools.run_evaluation import run_full_evaluation

results = run_full_evaluation(
    model_path='model.pth',
    data_dir='data/kitti/training',
    output_dir='results/',
    visualize=True,
    class_names=['Car']
)
```

### `lib/helpers/visualization_utils.py`

**Backward compatibility bridge** - imports from this module are deprecated.

```python
# ⚠️ Deprecated - shows warning
from lib.helpers.visualization_utils import DetectionVisualizer

# ✅ Use this instead
from tools.visualization import DetectionVisualizer
```

## Command-Line Tools

### metrics_calculator.py

Calculate KITTI evaluation metrics.

```bash
python tools/metrics_calculator.py \
    --gt_dir <ground_truth_dir> \
    --pred_dir <predictions_dir> \
    --classes Car Pedestrian Cyclist \
    --ap_mode 40 \
    --output metrics.json \
    --difficulty 0 1 2
```

**Options:**
- `--gt_dir`: Ground truth labels directory (required)
- `--pred_dir`: Predictions directory (required)
- `--classes`: Classes to evaluate (default: Car)
- `--ap_mode`: 40 for R40, 11 for R11 (default: 40)
- `--output`: Output JSON file (default: metrics_results.json)
- `--difficulty`: Difficulty levels 0=Easy, 1=Moderate, 2=Hard

### run_evaluation.py

Complete evaluation pipeline.

```bash
python tools/run_evaluation.py \
    --model <model_checkpoint> \
    --data <kitti_data_dir> \
    --output <output_dir> \
    --visualize \
    --classes Car \
    --device cuda \
    --ap_mode 40
```

**Options:**
- `--model`: Model checkpoint path (optional)
- `--data`: KITTI dataset directory (required)
- `--output`: Output directory (default: ./evaluation_results)
- `--visualize`: Generate visualizations
- `--classes`: Classes to evaluate (default: Car)
- `--device`: cuda or cpu (default: cuda)
- `--ap_mode`: 40 or 11 (default: 40)

## Examples

### Example 1: Visualize KITTI Dataset

```python
from pathlib import Path
from tools.visualization import DetectionVisualizer, load_kitti_calib
import cv2

visualizer = DetectionVisualizer()

image_dir = Path('data/kitti/image_2')
calib_dir = Path('data/kitti/calib')

for img_path in sorted(image_dir.glob('*.png')):
    image = cv2.imread(str(img_path))
    calib = load_kitti_calib(str(calib_dir / f"{img_path.stem}.txt"))
    
    # Your detections here
    detections = model.predict(image)
    
    vis_img = visualizer.visualize_detections(image, detections, calib['P2'])
    cv2.imwrite(f'output/{img_path.name}', vis_img)
```

### Example 2: Generate Animation

```python
from pathlib import Path
from tools.visualization import DetectionVisualizer, load_kitti_calib
from tools.gif_generator import FrameToGifConverter
import cv2

# Process frames
frame_dir = Path('temp_frames')
frame_dir.mkdir(exist_ok=True)

visualizer = DetectionVisualizer()
for i, (image, detections, calib) in enumerate(data_loader):
    vis_img = visualizer.visualize_detections(image, detections, calib['P2'])
    cv2.imwrite(str(frame_dir / f'frame_{i:04d}.png'), vis_img)

# Create GIF
converter = FrameToGifConverter(frame_dir, 'animation.gif', fps=10)
converter.create_gif(duration=100)
converter.create_video('animation.mp4', fps=10)
```

### Example 3: Batch Evaluation

```bash
# Step 1: Run inference (if needed)
python your_inference_script.py \
    --model checkpoints/model.pth \
    --data data/kitti/testing \
    --output results/predictions

# Step 2: Calculate metrics
python tools/metrics_calculator.py \
    --gt_dir data/kitti/labels \
    --pred_dir results/predictions \
    --classes Car Pedestrian Cyclist \
    --ap_mode 40 \
    --output results/metrics.json

# Or use the complete pipeline
python tools/run_evaluation.py \
    --model checkpoints/model.pth \
    --data data/kitti/testing \
    --output results/complete_eval \
    --visualize
```

## API Reference

### DetectionVisualizer

```python
class DetectionVisualizer:
    def __init__(self, class_names: Optional[List[str]] = None)
    
    def draw_3d_bbox(self,
                     image: np.ndarray,
                     detection: Dict,
                     P2: np.ndarray,
                     thickness: int = 2,
                     show_info: bool = True) -> np.ndarray
    
    def visualize_detections(self,
                            image: np.ndarray,
                            detections: List[Dict],
                            P2: np.ndarray,
                            thickness: int = 2,
                            show_info: bool = True) -> np.ndarray
    
    def compute_box_3d(self,
                       dimensions: np.ndarray,
                       location: np.ndarray,
                       rotation_y: float) -> np.ndarray
    
    def project_3d_to_2d(self,
                         corners_3d: np.ndarray,
                         P2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]
    
    def save_frame(self,
                   image: np.ndarray,
                   detections: List[Dict],
                   P2: np.ndarray,
                   output_path: str,
                   thickness: int = 2,
                   show_info: bool = True) -> bool
```

### Detection Format

```python
detection = {
    'class': str,              # 'Car', 'Pedestrian', 'Cyclist'
    'dimensions': [h, w, l],   # 3D dimensions in meters
    'location': [x, y, z],     # 3D location in camera coords
    'rotation_y': float,       # Rotation around Y-axis (-π to π)
    'score': float             # Confidence score (optional)
}
```

### MetricsCalculator

```python
class MetricsCalculator:
    def __init__(self,
                 gt_dir: Union[str, Path],
                 pred_dir: Union[str, Path],
                 class_names: Optional[List[str]] = None,
                 ap_mode: int = 40)
    
    def load_annotations(self, ann_dir: Path) -> List[Dict]
    
    def calculate_metrics(self,
                         difficultys: List[int] = [0, 1, 2],
                         z_axis: int = 1,
                         z_center: float = 1.0) -> Dict
    
    def calculate_iou_3d(self, box1: np.ndarray, box2: np.ndarray) -> float
    
    def calculate_iou_bev(self, box1: np.ndarray, box2: np.ndarray) -> float
    
    def print_results(self, metrics: Dict)
    
    def save_results(self, metrics: Dict, output_path: Union[str, Path])
```

## Migration Guide

If you're using the old import paths:

```python
# Old (deprecated)
from lib.helpers.visualization_utils import DetectionVisualizer
from lib.helpers.visualization_utils import FrameToGifConverter
from lib.helpers.visualization_utils import InferencePipeline

# New (recommended)
from tools.visualization import DetectionVisualizer
from tools.gif_generator import FrameToGifConverter
from tools.inference_pipeline import InferencePipeline
```

The old imports will still work but will show a deprecation warning.

## Troubleshooting

### CUDA Not Available

If you see CUDA errors, the tools will automatically fall back to CPU mode or limited functionality. To suppress CUDA warnings:

```bash
export NUMBA_DISABLE_CUDA=1
```

### Missing Dependencies

If you get import errors:
- For visualization: `pip install opencv-python numpy`
- For GIF generation: `pip install imageio`
- For progress bars: `pip install tqdm`
- For CUDA evaluation: `pip install numba`
- For inference: `pip install torch`

## Contributing

When adding new features:
1. Use absolute imports (`from tools.xxx import`)
2. Add type hints and docstrings
3. Include examples in docstrings
4. Update this README
5. Add tests if applicable

## License

This project follows the same license as MonoLSS.

## Acknowledgments

- KITTI dataset evaluation code adapted from existing `tools/eval.py`
- Visualization follows KITTI 3D object detection format
- Compatible with official KITTI evaluation protocols
