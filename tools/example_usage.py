#!/usr/bin/env python3
"""
Example script demonstrating MonoLSS visualization tools usage.

This script shows how to:
1. Load KITTI calibration files
2. Create 3D bounding box visualizations
3. Generate GIF animations from detection sequences
4. Use the inference pipeline
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import cv2
from tools.visualization import DetectionVisualizer, load_kitti_calib
from tools.gif_generator import FrameToGifConverter

def example_basic_visualization():
    """Example 1: Basic 3D bbox visualization"""
    print("="*60)
    print("Example 1: Basic 3D Bounding Box Visualization")
    print("="*60)
    
    # Create visualizer
    visualizer = DetectionVisualizer(class_names=['Car', 'Pedestrian', 'Cyclist'])
    
    # Create example image
    img = np.ones((375, 1242, 3), dtype=np.uint8) * 200
    
    # Example KITTI P2 camera matrix
    P2 = np.array([
        [721.5377, 0.0, 609.5593, 44.85728],
        [0.0, 721.5377, 172.854, 0.2163791],
        [0.0, 0.0, 1.0, 0.002745884]
    ])
    
    # Example detection
    detection = {
        'class': 'Car',
        'dimensions': [1.5, 1.8, 4.0],  # h, w, l in meters
        'location': [2.0, 1.5, 10.0],   # x, y, z in camera coords
        'rotation_y': 0.5,               # rotation around Y-axis
        'score': 0.95                    # confidence score
    }
    
    # Visualize
    vis_img = visualizer.draw_3d_bbox(img, detection, P2, show_info=True)
    
    print("✓ Created visualization with 3D bounding box")
    print(f"  Image shape: {vis_img.shape}")
    print(f"  Detection class: {detection['class']}")
    print(f"  Detection score: {detection['score']}")
    
    # Save result
    output_path = Path('/tmp/example_bbox.png')
    cv2.imwrite(str(output_path), vis_img)
    print(f"✓ Saved to: {output_path}")
    print()


def example_multiple_detections():
    """Example 2: Multiple detections on one image"""
    print("="*60)
    print("Example 2: Multiple Detections Visualization")
    print("="*60)
    
    visualizer = DetectionVisualizer()
    
    # Create example image
    img = np.ones((375, 1242, 3), dtype=np.uint8) * 200
    
    # KITTI P2 matrix
    P2 = np.array([
        [721.5377, 0.0, 609.5593, 44.85728],
        [0.0, 721.5377, 172.854, 0.2163791],
        [0.0, 0.0, 1.0, 0.002745884]
    ])
    
    # Multiple detections
    detections = [
        {
            'class': 'Car',
            'dimensions': [1.5, 1.8, 4.0],
            'location': [2.0, 1.5, 10.0],
            'rotation_y': 0.5,
            'score': 0.95
        },
        {
            'class': 'Pedestrian',
            'dimensions': [1.7, 0.6, 0.8],
            'location': [-1.0, 1.5, 8.0],
            'rotation_y': 0.0,
            'score': 0.88
        },
        {
            'class': 'Cyclist',
            'dimensions': [1.8, 0.8, 1.8],
            'location': [1.5, 1.5, 15.0],
            'rotation_y': -0.3,
            'score': 0.75
        }
    ]
    
    # Visualize all detections
    vis_img = visualizer.visualize_detections(img, detections, P2)
    
    print(f"✓ Visualized {len(detections)} detections")
    for det in detections:
        print(f"  - {det['class']}: score={det['score']:.2f}")
    
    # Save result
    output_path = Path('/tmp/example_multiple.png')
    cv2.imwrite(str(output_path), vis_img)
    print(f"✓ Saved to: {output_path}")
    print()


def example_gif_generation():
    """Example 3: Generate GIF animation from frames"""
    print("="*60)
    print("Example 3: GIF Animation Generation")
    print("="*60)
    
    # Create temporary directory for frames
    frame_dir = Path('/tmp/demo_frames')
    frame_dir.mkdir(exist_ok=True)
    
    visualizer = DetectionVisualizer()
    
    # KITTI P2 matrix
    P2 = np.array([
        [721.5377, 0.0, 609.5593, 44.85728],
        [0.0, 721.5377, 172.854, 0.2163791],
        [0.0, 0.0, 1.0, 0.002745884]
    ])
    
    # Generate sequence of frames with moving car
    num_frames = 10
    for i in range(num_frames):
        # Create image
        img = np.ones((375, 1242, 3), dtype=np.uint8) * 200
        
        # Moving car (getting closer)
        detection = {
            'class': 'Car',
            'dimensions': [1.5, 1.8, 4.0],
            'location': [2.0, 1.5, 20.0 - i * 1.0],  # Moving closer
            'rotation_y': 0.5,
            'score': 0.95
        }
        
        # Visualize
        vis_img = visualizer.draw_3d_bbox(img, detection, P2)
        
        # Save frame
        frame_path = frame_dir / f'frame_{i:04d}.png'
        cv2.imwrite(str(frame_path), vis_img)
    
    print(f"✓ Generated {num_frames} frames")
    
    # Create GIF
    converter = FrameToGifConverter(
        frame_dir=frame_dir,
        output_path='/tmp/demo_animation.gif',
        fps=5
    )
    
    gif_path = converter.create_gif(duration=200)
    print(f"✓ Created GIF: {gif_path}")
    
    # Also create video
    video_path = converter.create_video(
        output_video='/tmp/demo_animation.mp4',
        fps=5
    )
    print(f"✓ Created video: /tmp/demo_animation.mp4")
    print()


def example_kitti_calibration():
    """Example 4: Load KITTI calibration file"""
    print("="*60)
    print("Example 4: KITTI Calibration File Loading")
    print("="*60)
    
    # Create example calibration file
    calib_file = Path('/tmp/example_calib.txt')
    with open(calib_file, 'w') as f:
        f.write("P0: 707.0493 0.0 604.0814 0.0 0.0 707.0493 180.5066 0.0 0.0 0.0 1.0 0.0\n")
        f.write("P1: 707.0493 0.0 604.0814 -379.7842 0.0 707.0493 180.5066 0.0 0.0 0.0 1.0 0.0\n")
        f.write("P2: 721.5377 0.0 609.5593 44.85728 0.0 721.5377 172.854 0.2163791 0.0 0.0 1.0 0.002745884\n")
        f.write("P3: 721.5377 0.0 609.5593 -339.5 0.0 721.5377 172.854 2.199 0.0 0.0 1.0 0.002\n")
        f.write("R0_rect: 0.9999 0.0124 -0.0038 -0.0124 0.9999 -0.0009 0.0038 0.0010 0.9999\n")
        f.write("Tr_velo_to_cam: 0.007534 -0.9999 -0.0068 -0.0027 -0.0144 0.0068 -0.9998 -0.0758 0.9998 0.007553 -0.0144 -0.2721\n")
    
    # Load calibration
    calib = load_kitti_calib(str(calib_file))
    
    print("✓ Loaded KITTI calibration file")
    print(f"  P2 shape: {calib['P2'].shape}")
    print(f"  P2 matrix:\n{calib['P2']}")
    print(f"  R0_rect shape: {calib['R0_rect'].shape}")
    print()


def main():
    """Run all examples"""
    print("\n" + "="*60)
    print("MonoLSS Visualization Tools - Example Usage")
    print("="*60 + "\n")
    
    try:
        example_basic_visualization()
        example_multiple_detections()
        example_gif_generation()
        example_kitti_calibration()
        
        print("="*60)
        print("✅ All examples completed successfully!")
        print("="*60)
        print("\nGenerated files:")
        print("  - /tmp/example_bbox.png")
        print("  - /tmp/example_multiple.png")
        print("  - /tmp/demo_animation.gif")
        print("  - /tmp/demo_animation.mp4")
        print("  - /tmp/demo_frames/ (directory with frames)")
        print()
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
