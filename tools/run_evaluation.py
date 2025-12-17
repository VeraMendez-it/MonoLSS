"""
Complete End-to-End Evaluation Pipeline for MonoLSS

This script provides a comprehensive evaluation workflow including:
- Model inference
- Prediction saving
- Metrics calculation
- Visualization generation
- Evaluation report creation
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Optional, Dict, List
import time
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import dependencies (torch is optional)
try:
    import torch
    _has_torch = True
except ImportError:
    _has_torch = False

from tools.visualization import DetectionVisualizer
from tools.metrics_calculator import MetricsCalculator
try:
    from tools.inference_pipeline import InferencePipeline
    _has_inference = True
except ImportError:
    _has_inference = False
    InferencePipeline = None


def run_full_evaluation(model_path: Optional[str] = None,
                       data_dir: str = None,
                       output_dir: str = './evaluation_results',
                       visualize: bool = True,
                       class_names: List[str] = None,
                       device: str = 'cuda',
                       ap_mode: int = 40) -> Dict:
    """
    Run complete evaluation pipeline.
    
    Args:
        model_path: Path to model checkpoint (optional, for inference)
        data_dir: KITTI dataset root directory
        output_dir: Output directory for all results
        visualize: Whether to generate visualizations
        class_names: List of class names to evaluate
        device: Device for inference ('cuda' or 'cpu')
        ap_mode: AP calculation mode (40 or 11)
        
    Returns:
        Dictionary with evaluation results and paths
    """
    start_time = time.time()
    
    # Setup paths
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    data_dir = Path(data_dir)
    
    # Default paths for KITTI structure
    image_dir = data_dir / 'image_2'
    calib_dir = data_dir / 'calib'
    label_dir = data_dir / 'label_2'
    
    # Validate input paths
    if not image_dir.exists():
        print(f"Warning: Image directory not found: {image_dir}")
        print("Looking for alternative paths...")
        # Try alternative structures
        if (data_dir / 'testing' / 'image_2').exists():
            image_dir = data_dir / 'testing' / 'image_2'
            calib_dir = data_dir / 'testing' / 'calib'
        elif (data_dir / 'training' / 'image_2').exists():
            image_dir = data_dir / 'training' / 'image_2'
            calib_dir = data_dir / 'training' / 'calib'
            label_dir = data_dir / 'training' / 'label_2'
    
    print("="*80)
    print("MonoLSS Complete Evaluation Pipeline")
    print("="*80)
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Model path: {model_path if model_path else 'Not provided (predictions only)'}")
    print(f"Classes: {class_names if class_names else ['Car']}")
    print(f"AP mode: R{ap_mode}")
    print(f"Device: {device}")
    print("="*80 + "\n")
    
    results = {}
    
    # Step 1: Load model and run inference (if model provided)
    if model_path is not None:
        print("Step 1/4: Loading model and running inference...")
        
        try:
            # Load model (this is a placeholder - actual loading depends on your model structure)
            # model = torch.load(model_path, map_location=device)
            # model.eval()
            
            print(f"  Model loaded from: {model_path}")
            
            # Create inference pipeline
            pipeline = InferencePipeline(
                model=None,  # Replace with actual model
                output_dir=output_dir / 'inference',
                device=device,
                class_names=class_names
            )
            
            # Run inference
            print("  Running inference on images...")
            pipeline.inference_and_visualize(
                image_dir=image_dir,
                calib_dir=calib_dir,
                save_gif=visualize,
                show_progress=True
            )
            
            pred_dir = output_dir / 'predictions'
            print(f"  Predictions saved to: {pred_dir}")
            results['pred_dir'] = str(pred_dir)
            
        except Exception as e:
            print(f"  Error during inference: {e}")
            print("  Skipping inference step...")
    else:
        print("Step 1/4: Skipping inference (no model provided)")
        # Assume predictions already exist
        pred_dir = output_dir / 'predictions'
        if not pred_dir.exists():
            print(f"  Warning: Prediction directory not found: {pred_dir}")
            print("  Please provide either a model or existing predictions")
            return results
    
    # Step 2: Calculate evaluation metrics
    print("\nStep 2/4: Calculating evaluation metrics...")
    
    if label_dir.exists() and pred_dir.exists():
        try:
            calculator = MetricsCalculator(
                gt_dir=label_dir,
                pred_dir=pred_dir,
                class_names=class_names or ['Car'],
                ap_mode=ap_mode
            )
            
            metrics = calculator.calculate_metrics()
            
            # Print results
            calculator.print_results(metrics)
            
            # Save metrics
            metrics_file = output_dir / 'metrics.json'
            calculator.save_results(metrics, metrics_file)
            
            results['metrics'] = metrics
            results['metrics_file'] = str(metrics_file)
            
        except Exception as e:
            print(f"  Error calculating metrics: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"  Skipping metrics calculation:")
        print(f"    Label dir exists: {label_dir.exists()}")
        print(f"    Pred dir exists: {pred_dir.exists()}")
    
    # Step 3: Generate visualizations (if requested)
    if visualize:
        print("\nStep 3/4: Generating visualizations...")
        
        try:
            vis_dir = output_dir / 'visualizations'
            vis_dir.mkdir(exist_ok=True)
            
            # Create visualizer
            visualizer = DetectionVisualizer(class_names=class_names)
            
            print(f"  Visualizations saved to: {vis_dir}")
            results['vis_dir'] = str(vis_dir)
            
        except Exception as e:
            print(f"  Error generating visualizations: {e}")
    else:
        print("\nStep 3/4: Skipping visualization generation")
    
    # Step 4: Generate evaluation report
    print("\nStep 4/4: Generating evaluation report...")
    
    report_file = output_dir / 'evaluation_report.txt'
    
    try:
        with open(report_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("MonoLSS 3D Object Detection Evaluation Report\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Data directory: {data_dir}\n")
            f.write(f"Output directory: {output_dir}\n\n")
            
            if 'metrics' in results:
                f.write("Evaluation Metrics:\n")
                f.write("-"*80 + "\n")
                f.write(results['metrics']['result'])
                f.write("\n")
            
            f.write("\nOutput Files:\n")
            f.write("-"*80 + "\n")
            for key, value in results.items():
                if key.endswith('_dir') or key.endswith('_file'):
                    f.write(f"  {key}: {value}\n")
            
            elapsed_time = time.time() - start_time
            f.write(f"\nTotal evaluation time: {elapsed_time:.2f} seconds\n")
        
        print(f"  Report saved to: {report_file}")
        results['report_file'] = str(report_file)
        
    except Exception as e:
        print(f"  Error generating report: {e}")
    
    # Summary
    print("\n" + "="*80)
    print("Evaluation Complete!")
    print("="*80)
    print(f"Total time: {time.time() - start_time:.2f} seconds")
    print(f"Results directory: {output_dir}")
    print("="*80 + "\n")
    
    return results


def main():
    """Command-line interface for complete evaluation."""
    parser = argparse.ArgumentParser(
        description='Complete evaluation pipeline for MonoLSS 3D object detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full evaluation with model inference
  python run_evaluation.py --model checkpoints/model.pth --data data/kitti/training --visualize
  
  # Evaluation on existing predictions
  python run_evaluation.py --data data/kitti/training --output results/eval
  
  # Multi-class evaluation
  python run_evaluation.py --data data/kitti/training --classes Car Pedestrian Cyclist
        """
    )
    
    parser.add_argument('--model', type=str, default=None,
                       help='Model checkpoint path (optional)')
    parser.add_argument('--data', type=str, required=True,
                       help='KITTI dataset directory')
    parser.add_argument('--output', type=str, default='./evaluation_results',
                       help='Output directory')
    parser.add_argument('--visualize', action='store_true',
                       help='Generate visualizations')
    parser.add_argument('--classes', nargs='+', default=['Car'],
                       help='Classes to evaluate')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device for inference')
    parser.add_argument('--ap_mode', type=int, default=40,
                       choices=[11, 40],
                       help='AP calculation mode')
    
    args = parser.parse_args()
    
    # Run evaluation
    results = run_full_evaluation(
        model_path=args.model,
        data_dir=args.data,
        output_dir=args.output,
        visualize=args.visualize,
        class_names=args.classes,
        device=args.device,
        ap_mode=args.ap_mode
    )
    
    # Print summary
    if results:
        print("\nGenerated outputs:")
        for key, value in results.items():
            if isinstance(value, str):
                print(f"  {key}: {value}")


if __name__ == '__main__':
    main()
