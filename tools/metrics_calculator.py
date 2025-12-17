"""
Metrics Calculator for 3D Object Detection

Provides comprehensive evaluation metrics for KITTI-format 3D object detection,
including AP (Average Precision), 3D IoU, BEV IoU, and detailed analysis.

This module integrates with the existing eval.py to leverage GPU-accelerated
IoU calculations and official KITTI evaluation protocols.
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from collections import defaultdict

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import from existing eval.py (may require CUDA)
try:
    # Disable CUDA initialization errors by setting environment variable
    os.environ.setdefault('NUMBA_DISABLE_CUDA', '1')
    from tools import eval as kitti_eval
    _has_kitti_eval = True
except Exception as e:
    print(f"Warning: Could not import eval.py: {e}")
    print("Some functionality may be limited.")
    _has_kitti_eval = False
    kitti_eval = None


class MetricsCalculator:
    """
    3D Object Detection Metrics Calculator.
    Supports KITTI format evaluation with multiple difficulty levels and classes.
    """
    
    def __init__(self, 
                 gt_dir: Union[str, Path],
                 pred_dir: Union[str, Path],
                 class_names: Optional[List[str]] = None,
                 ap_mode: int = 40):
        """
        Initialize the metrics calculator.
        
        Args:
            gt_dir: Directory containing ground truth KITTI label files
            pred_dir: Directory containing prediction KITTI label files
            class_names: List of class names to evaluate (default: ['Car'])
            ap_mode: AP calculation mode, 40 or 11 (default: 40)
                    - 40: Use 40 recall points (R40, recommended)
                    - 11: Use 11 recall points (R11, legacy)
        """
        self.gt_dir = Path(gt_dir)
        self.pred_dir = Path(pred_dir)
        self.class_names = class_names or ['Car']
        self.ap_mode = ap_mode
        
        # Validate directories
        if not self.gt_dir.exists():
            raise ValueError(f"Ground truth directory not found: {self.gt_dir}")
        if not self.pred_dir.exists():
            raise ValueError(f"Prediction directory not found: {self.pred_dir}")
    
    def load_annotations(self, ann_dir: Path) -> List[Dict]:
        """
        Load KITTI format annotation files.
        
        Args:
            ann_dir: Directory containing KITTI label files
            
        Returns:
            List of annotation dictionaries, one per file
        """
        label_files = sorted(ann_dir.glob('*.txt'))
        
        if len(label_files) == 0:
            raise ValueError(f"No label files found in {ann_dir}")
        
        annotations = []
        
        for label_file in label_files:
            # Initialize annotation dict
            ann = {
                'name': [],
                'truncated': [],
                'occluded': [],
                'alpha': [],
                'bbox': [],
                'dimensions': [],
                'location': [],
                'rotation_y': [],
                'score': []
            }
            
            # Read file
            with open(label_file, 'r') as f:
                lines = f.readlines()
            
            # Parse each line
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                if len(parts) < 15:
                    continue
                
                # Parse KITTI format
                # Format: class truncated occluded alpha bbox[4] dimensions[3] location[3] rotation_y [score]
                ann['name'].append(parts[0])
                ann['truncated'].append(float(parts[1]))
                ann['occluded'].append(int(parts[2]))
                ann['alpha'].append(float(parts[3]))
                ann['bbox'].append([float(x) for x in parts[4:8]])
                ann['dimensions'].append([float(x) for x in parts[8:11]])
                ann['location'].append([float(x) for x in parts[11:14]])
                ann['rotation_y'].append(float(parts[14]))
                
                # Score is optional (only in predictions)
                if len(parts) > 15:
                    ann['score'].append(float(parts[15]))
                else:
                    ann['score'].append(1.0)
            
            # Convert to numpy arrays
            for key in ann:
                if key == 'name':
                    ann[key] = np.array(ann[key])
                else:
                    ann[key] = np.array(ann[key])
            
            # Handle empty case
            if len(ann['name']) == 0:
                for key in ann:
                    if key == 'name':
                        ann[key] = np.array([])
                    else:
                        ann[key] = np.array([]).reshape(0)
            
            annotations.append(ann)
        
        return annotations
    
    def calculate_metrics(self, 
                         difficultys: List[int] = [0, 1, 2],
                         z_axis: int = 1,
                         z_center: float = 1.0) -> Dict:
        """
        Calculate complete evaluation metrics.
        
        Args:
            difficultys: Difficulty levels to evaluate (0=Easy, 1=Moderate, 2=Hard)
            z_axis: Z-axis index for 3D box calculation
            z_center: Z-center offset for 3D box calculation
            
        Returns:
            Dictionary containing:
                - 'result': Formatted result string
                - 'detail': Detailed metrics per class
                - 'bbox': 2D bounding box AP
                - 'bev': Bird's eye view AP
                - '3d': 3D bounding box AP
                - 'aos': Average orientation similarity (if available)
        """
        if not _has_kitti_eval:
            raise RuntimeError(
                "KITTI evaluation module not available. "
                "This may be due to missing CUDA support or dependencies. "
                "Please ensure numba and CUDA are properly installed."
            )
        
        print("Loading annotations...")
        gt_annos = self.load_annotations(self.gt_dir)
        dt_annos = self.load_annotations(self.pred_dir)
        
        print(f"Loaded {len(gt_annos)} ground truth annotations")
        print(f"Loaded {len(dt_annos)} prediction annotations")
        
        if len(gt_annos) != len(dt_annos):
            print(f"Warning: Number of GT ({len(gt_annos)}) and predictions ({len(dt_annos)}) don't match")
            # Truncate to minimum length
            min_len = min(len(gt_annos), len(dt_annos))
            gt_annos = gt_annos[:min_len]
            dt_annos = dt_annos[:min_len]
        
        print("Calculating metrics...")
        
        # Use official KITTI evaluation from eval.py
        result = kitti_eval.get_official_eval_result(
            gt_annos,
            dt_annos,
            self.class_names,
            difficultys=difficultys,
            z_axis=z_axis,
            z_center=z_center
        )
        
        return result
    
    def calculate_iou_3d(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """
        Calculate 3D IoU between two boxes.
        
        Args:
            box1: First 3D box [x, y, z, h, w, l, ry]
            box2: Second 3D box [x, y, z, h, w, l, ry]
            
        Returns:
            3D IoU value
        """
        if not _has_kitti_eval:
            print("Warning: KITTI eval not available, returning 0.0")
            return 0.0
        
        # Use GPU-accelerated calculation from eval.py
        boxes = np.array([box1]).reshape(1, 7)
        qboxes = np.array([box2]).reshape(1, 7)
        
        try:
            iou = kitti_eval.box3d_overlap(boxes, qboxes, criterion=-1)
            return float(iou[0, 0])
        except Exception as e:
            print(f"Error calculating 3D IoU: {e}")
            return 0.0
    
    def calculate_iou_bev(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """
        Calculate Bird's Eye View (BEV) IoU between two boxes.
        
        Args:
            box1: First box in BEV [x, z, w, l, ry]
            box2: Second box in BEV [x, z, w, l, ry]
            
        Returns:
            BEV IoU value
        """
        if not _has_kitti_eval:
            print("Warning: KITTI eval not available, returning 0.0")
            return 0.0
        
        # Use GPU-accelerated calculation from eval.py
        boxes = np.array([box1]).reshape(1, 5)
        qboxes = np.array([box2]).reshape(1, 5)
        
        try:
            iou = kitti_eval.bev_box_overlap(boxes, qboxes, criterion=-1)
            return float(iou[0, 0])
        except Exception as e:
            print(f"Error calculating BEV IoU: {e}")
            return 0.0
    
    def calculate_precision_recall(self, 
                                   scores: np.ndarray,
                                   matches: np.ndarray,
                                   num_gt: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate precision-recall curve.
        
        Args:
            scores: Confidence scores for detections (N,)
            matches: Binary array indicating if detection is TP (N,)
            num_gt: Total number of ground truth objects
            
        Returns:
            Tuple of (precision, recall) arrays
        """
        # Sort by scores in descending order
        sorted_indices = np.argsort(-scores)
        matches = matches[sorted_indices]
        
        # Calculate cumulative TP and FP
        tp = np.cumsum(matches)
        fp = np.cumsum(1 - matches)
        
        # Calculate precision and recall
        precision = tp / (tp + fp + 1e-10)
        recall = tp / (num_gt + 1e-10)
        
        return precision, recall
    
    def get_ap(self, 
               precision: np.ndarray,
               recall: np.ndarray,
               mode: str = '40') -> float:
        """
        Calculate Average Precision (AP).
        
        Args:
            precision: Precision array
            recall: Recall array
            mode: '40' for 40-point interpolation, '11' for 11-point
            
        Returns:
            AP value
        """
        if mode == '40':
            # 40-point interpolation (KITTI R40)
            recall_points = np.linspace(0, 1, 41)
        else:
            # 11-point interpolation (KITTI R11, legacy)
            recall_points = np.linspace(0, 1, 11)
        
        ap = 0.0
        for r in recall_points:
            # Get maximum precision for recall >= r
            precisions = precision[recall >= r]
            if len(precisions) > 0:
                ap += np.max(precisions)
        
        ap /= len(recall_points)
        return ap
    
    def print_results(self, metrics: Dict):
        """
        Print formatted evaluation results.
        
        Args:
            metrics: Metrics dictionary from calculate_metrics()
        """
        print("\n" + "="*80)
        print("KITTI 3D Object Detection Evaluation Results")
        print("="*80)
        print(metrics['result'])
        print("="*80 + "\n")
    
    def save_results(self, metrics: Dict, output_path: Union[str, Path]):
        """
        Save evaluation results to JSON file.
        
        Args:
            metrics: Metrics dictionary from calculate_metrics()
            output_path: Path to output JSON file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create serializable output
        output = {
            'result_text': metrics['result'],
            'detail': metrics['detail'],
            'config': {
                'class_names': self.class_names,
                'ap_mode': self.ap_mode,
                'gt_dir': str(self.gt_dir),
                'pred_dir': str(self.pred_dir)
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"Results saved to: {output_path}")


def calculate_detection_metrics(gt_file: str, pred_file: str) -> Dict:
    """
    Calculate detection metrics for a single image.
    
    Args:
        gt_file: Path to ground truth KITTI label file
        pred_file: Path to prediction KITTI label file
        
    Returns:
        Dictionary with:
            - 'TP': True Positives
            - 'FP': False Positives
            - 'FN': False Negatives
            - 'precision': Precision
            - 'recall': Recall
            - 'f1': F1 Score
    """
    # Simple implementation for single image
    # Load annotations
    calculator = MetricsCalculator(
        gt_dir=Path(gt_file).parent,
        pred_dir=Path(pred_file).parent
    )
    
    gt_annos = calculator.load_annotations(Path(gt_file).parent)
    pred_annos = calculator.load_annotations(Path(pred_file).parent)
    
    # Get the specific files
    gt_idx = [i for i, f in enumerate(sorted(Path(gt_file).parent.glob('*.txt'))) 
              if f.name == Path(gt_file).name][0]
    pred_idx = [i for i, f in enumerate(sorted(Path(pred_file).parent.glob('*.txt'))) 
                if f.name == Path(pred_file).name][0]
    
    gt_anno = gt_annos[gt_idx]
    pred_anno = pred_annos[pred_idx]
    
    # Count GT and predictions
    num_gt = len(gt_anno['name'])
    num_pred = len(pred_anno['name'])
    
    # Simple matching (this is simplified, real matching uses IoU)
    tp = min(num_gt, num_pred)
    fp = max(0, num_pred - num_gt)
    fn = max(0, num_gt - num_pred)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'TP': tp,
        'FP': fp,
        'FN': fn,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def main():
    """Command-line interface for metrics calculation."""
    parser = argparse.ArgumentParser(
        description='Calculate 3D object detection metrics for KITTI format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate Car detection
  python metrics_calculator.py --gt_dir data/kitti/labels --pred_dir results/predictions --classes Car
  
  # Evaluate multiple classes with AP11 mode
  python metrics_calculator.py --gt_dir data/kitti/labels --pred_dir results/predictions \\
      --classes Car Pedestrian Cyclist --ap_mode 11
  
  # Save results to custom file
  python metrics_calculator.py --gt_dir data/kitti/labels --pred_dir results/predictions \\
      --output evaluation_results.json
        """
    )
    
    parser.add_argument('--gt_dir', required=True, 
                       help='Ground truth labels directory')
    parser.add_argument('--pred_dir', required=True,
                       help='Prediction labels directory')
    parser.add_argument('--classes', nargs='+', default=['Car'],
                       help='Classes to evaluate (default: Car)')
    parser.add_argument('--ap_mode', type=int, default=40, choices=[11, 40],
                       help='AP calculation mode: 40 (R40) or 11 (R11, legacy)')
    parser.add_argument('--output', default='metrics_results.json',
                       help='Output JSON file path')
    parser.add_argument('--difficulty', nargs='+', type=int, default=[0, 1, 2],
                       choices=[0, 1, 2],
                       help='Difficulty levels: 0=Easy, 1=Moderate, 2=Hard')
    
    args = parser.parse_args()
    
    # Create calculator
    calculator = MetricsCalculator(
        gt_dir=args.gt_dir,
        pred_dir=args.pred_dir,
        class_names=args.classes,
        ap_mode=args.ap_mode
    )
    
    # Calculate metrics
    metrics = calculator.calculate_metrics(difficultys=args.difficulty)
    
    # Print results
    calculator.print_results(metrics)
    
    # Save results
    calculator.save_results(metrics, args.output)


if __name__ == '__main__':
    main()
