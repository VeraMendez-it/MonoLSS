"""
Inference Pipeline for MonoLSS 3D Object Detection
Provides end-to-end inference and visualization functionality
"""

import torch
from pathlib import Path
from typing import Optional, List, Dict, Union
import cv2
import os
from tqdm import tqdm
import numpy as np

from tools.visualization import DetectionVisualizer, load_kitti_calib
from tools.gif_generator import FrameToGifConverter


class InferencePipeline:
    """
    Complete inference pipeline with batch processing and visualization.
    Supports KITTI dataset format and custom image directories.
    """
    
    def __init__(self, 
                 model: Optional[torch.nn.Module] = None,
                 output_dir: str = './vis_results',
                 device: str = 'cuda',
                 class_names: Optional[List[str]] = None):
        """
        Initialize the inference pipeline.
        
        Args:
            model: PyTorch model for inference (optional)
            output_dir: Directory to save visualization results
            device: Device to run inference on ('cuda' or 'cpu')
            class_names: List of class names for visualization
        """
        self.model = model
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize visualizer
        self.visualizer = DetectionVisualizer(class_names=class_names)
        
        # Move model to device if provided
        if self.model is not None:
            self.model.to(device)
            self.model.eval()
    
    def load_kitti_batch(self, 
                        image_dir: Union[str, Path],
                        calib_dir: Union[str, Path]) -> List[Dict]:
        """
        Load KITTI format images and calibration files.
        
        Args:
            image_dir: Directory containing images
            calib_dir: Directory containing calibration files
            
        Returns:
            List of dicts with 'image_path', 'calib', 'image_id'
        """
        image_dir = Path(image_dir)
        calib_dir = Path(calib_dir)
        
        # Find all images
        image_paths = sorted(list(image_dir.glob('*.png')) + 
                           list(image_dir.glob('*.jpg')))
        
        batch_data = []
        for img_path in image_paths:
            # Get image id (filename without extension)
            image_id = img_path.stem
            
            # Load calibration
            calib_path = calib_dir / f"{image_id}.txt"
            
            if not calib_path.exists():
                print(f"Warning: Calibration file not found for {image_id}, skipping")
                continue
            
            try:
                calib = load_kitti_calib(str(calib_path))
                batch_data.append({
                    'image_path': img_path,
                    'calib': calib,
                    'image_id': image_id
                })
            except Exception as e:
                print(f"Error loading calibration for {image_id}: {e}")
                continue
        
        return batch_data
    
    def inference_and_visualize(self, 
                               image_dir: Union[str, Path],
                               calib_dir: Optional[Union[str, Path]] = None,
                               calibration_matrix: Optional[np.ndarray] = None,
                               save_gif: bool = True,
                               show_progress: bool = True) -> Path:
        """
        Batch inference and visualization.
        
        Args:
            image_dir: Directory containing images
            calib_dir: Directory containing KITTI calibration files (optional)
            calibration_matrix: Single calibration matrix to use for all images (optional)
            save_gif: Whether to generate GIF animation
            show_progress: Whether to show progress bar
            
        Returns:
            Path to output directory
            
        Note:
            Either calib_dir or calibration_matrix must be provided.
        """
        if calib_dir is None and calibration_matrix is None:
            raise ValueError("Either calib_dir or calibration_matrix must be provided")
        
        image_dir = Path(image_dir)
        
        # Load batch data
        if calib_dir is not None:
            batch_data = self.load_kitti_batch(image_dir, calib_dir)
        else:
            # Use single calibration matrix for all images
            image_paths = sorted(list(image_dir.glob('*.png')) + 
                               list(image_dir.glob('*.jpg')))
            batch_data = [
                {
                    'image_path': img_path,
                    'calib': {'P2': calibration_matrix},
                    'image_id': img_path.stem
                }
                for img_path in image_paths
            ]
        
        if len(batch_data) == 0:
            print("No valid images found!")
            return self.output_dir
        
        print(f"Processing {len(batch_data)} images...")
        
        # Create frames directory
        frames_dir = self.output_dir / 'frames'
        frames_dir.mkdir(exist_ok=True)
        
        # Process each image
        iterator = tqdm(batch_data) if show_progress else batch_data
        
        for idx, data in enumerate(iterator):
            try:
                # Read image
                image = cv2.imread(str(data['image_path']))
                
                if image is None:
                    print(f"Failed to read image: {data['image_path']}")
                    continue
                
                # Get calibration matrix
                P2 = data['calib']['P2']
                
                # Run inference if model is available
                if self.model is not None:
                    with torch.no_grad():
                        detections = self.model.inference(image)
                else:
                    # Use empty detections if no model provided
                    detections = []
                
                # Visualize
                if isinstance(detections, list) and len(detections) > 0:
                    vis_image = self.visualizer.visualize_detections(
                        image, detections, P2
                    )
                else:
                    vis_image = image.copy()
                
                # Save frame
                frame_path = frames_dir / f"frame_{idx:06d}.png"
                cv2.imwrite(str(frame_path), vis_image)
                
            except Exception as e:
                print(f"Error processing {data['image_id']}: {e}")
                continue
        
        # Generate GIF animation
        if save_gif:
            try:
                print("Generating GIF animation...")
                converter = FrameToGifConverter(
                    frames_dir,
                    output_path=str(self.output_dir / 'inference_results.gif'),
                    fps=10
                )
                converter.create_gif(duration=100)
                
                # Also create video
                converter.create_video(
                    output_video=str(self.output_dir / 'inference_results.mp4'),
                    fps=10
                )
            except Exception as e:
                print(f"Error generating GIF/video: {e}")
        
        print(f"Results saved to: {self.output_dir}")
        return self.output_dir
