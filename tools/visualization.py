"""
3D Detection Visualization Tools for KITTI Format
Supports visualization of 3D bounding boxes on images with proper projections
"""

import numpy as np
import cv2
import os
from typing import Dict, List, Tuple, Optional, Union


class DetectionVisualizer:
    """
    Visualizer for 3D object detection results in KITTI format.
    Handles projection of 3D bounding boxes onto 2D images.
    """
    
    # Color mapping for different object classes (BGR format for OpenCV)
    CLASS_COLORS = {
        'Car': (0, 255, 0),           # Green
        'Pedestrian': (255, 0, 0),     # Blue
        'Cyclist': (0, 255, 255),      # Yellow
        'Van': (0, 128, 255),          # Orange
        'Truck': (128, 0, 128),        # Purple
        'Person_sitting': (255, 128, 0), # Light Blue
        'Tram': (128, 128, 0),         # Teal
        'Misc': (128, 128, 128),       # Gray
        'DontCare': (64, 64, 64)       # Dark Gray
    }
    
    def __init__(self, class_names: Optional[List[str]] = None):
        """
        Initialize the visualizer.
        
        Args:
            class_names: List of class names to visualize. If None, uses all KITTI classes.
        """
        if class_names is None:
            self.class_names = ['Car', 'Pedestrian', 'Cyclist']
        else:
            self.class_names = class_names
            
    def compute_box_3d(self, 
                       dimensions: np.ndarray, 
                       location: np.ndarray, 
                       rotation_y: float) -> np.ndarray:
        """
        Compute 3D bounding box corners from object dimensions, location and rotation.
        
        Args:
            dimensions: 3D object dimensions (h, w, l) in meters
            location: 3D object location (x, y, z) in camera coordinates
            rotation_y: Rotation around Y-axis in camera coordinates [-pi, pi]
            
        Returns:
            corners_3d: (8, 3) array of 3D corner coordinates in camera frame
            
        Corner order:
            4 -------- 5
           /|         /|
          7 -------- 6 .
          | |        | |
          . 0 -------- 1
          |/         |/
          3 -------- 2
        """
        # Extract dimensions
        h, w, l = dimensions
        
        # Create 3D bounding box corners in object coordinate system
        # Center of box is at origin
        x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
        y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
        z_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]
        
        # Combine into corner matrix
        corners_3d = np.vstack([x_corners, y_corners, z_corners])  # (3, 8)
        
        # Create rotation matrix around Y-axis
        c = np.cos(rotation_y)
        s = np.sin(rotation_y)
        rotation_matrix = np.array([
            [c, 0, s],
            [0, 1, 0],
            [-s, 0, c]
        ])
        
        # Rotate corners
        corners_3d = np.dot(rotation_matrix, corners_3d)
        
        # Translate to location
        corners_3d[0, :] += location[0]
        corners_3d[1, :] += location[1]
        corners_3d[2, :] += location[2]
        
        return corners_3d.T  # (8, 3)
    
    def project_3d_to_2d(self, 
                         corners_3d: np.ndarray, 
                         P2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Project 3D bounding box corners to 2D image plane.
        
        Args:
            corners_3d: (8, 3) array of 3D corners in camera coordinates
            P2: (3, 4) camera projection matrix
            
        Returns:
            corners_2d: (8, 2) array of 2D pixel coordinates
            depths: (8,) array of depth values for each corner
        """
        # Add homogeneous coordinate
        corners_3d_homogeneous = np.hstack([corners_3d, np.ones((corners_3d.shape[0], 1))])  # (8, 4)
        
        # Project to image plane
        corners_2d_homogeneous = np.dot(P2, corners_3d_homogeneous.T)  # (3, 8)
        
        # Extract depths
        depths = corners_2d_homogeneous[2, :]
        
        # Handle edge case: avoid division by zero
        depths_safe = np.where(np.abs(depths) < 1e-6, 1e-6, depths)
        
        # Normalize to get pixel coordinates
        corners_2d = corners_2d_homogeneous[:2, :] / depths_safe
        
        return corners_2d.T, depths  # (8, 2), (8,)
    
    def draw_3d_bbox(self,
                     image: np.ndarray,
                     detection: Dict,
                     P2: np.ndarray,
                     thickness: int = 2,
                     show_info: bool = True) -> np.ndarray:
        """
        Draw 3D bounding box on image.
        
        Args:
            image: Input image (H, W, 3) BGR format
            detection: Detection dict with keys:
                - 'bbox': 2D bbox [x1, y1, x2, y2] (optional, for reference)
                - 'dimensions': 3D dimensions [h, w, l]
                - 'location': 3D location [x, y, z]
                - 'rotation_y': rotation around Y-axis
                - 'score': confidence score (optional)
                - 'class': class name or index
            P2: (3, 4) camera projection matrix
            thickness: Line thickness
            show_info: Whether to show text information
            
        Returns:
            image: Image with 3D bbox drawn
        """
        # Make a copy to avoid modifying original
        img = image.copy()
        
        # Extract detection information
        try:
            dimensions = np.array(detection['dimensions'])
            location = np.array(detection['location'])
            rotation_y = detection['rotation_y']
            
            # Get class name
            if isinstance(detection.get('class'), str):
                class_name = detection['class']
            elif isinstance(detection.get('class'), int):
                class_name = self.class_names[detection['class']] if detection['class'] < len(self.class_names) else 'Unknown'
            else:
                class_name = 'Unknown'
            
            # Get color for this class
            color = self.CLASS_COLORS.get(class_name, (255, 255, 255))
            
            # Get confidence score
            score = detection.get('score', None)
            
        except (KeyError, IndexError, TypeError) as e:
            print(f"Error parsing detection: {e}")
            return img
        
        # Compute 3D bounding box corners
        corners_3d = self.compute_box_3d(dimensions, location, rotation_y)
        
        # Project to 2D
        corners_2d, depths = self.project_3d_to_2d(corners_3d, P2)
        
        # Check if box is in front of camera
        if np.all(depths < 0):
            return img  # Box is behind camera
        
        # Convert to integer pixel coordinates
        corners_2d = corners_2d.astype(np.int32)
        
        # Define edges of the 3D bounding box
        # Front face
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom face
            (4, 5), (5, 6), (6, 7), (7, 4),  # Top face
            (0, 4), (1, 5), (2, 6), (3, 7),  # Vertical edges
            (0, 5), (1, 4)  # Cross for front face orientation
        ]
        
        # Draw edges
        for start, end in edges:
            # Skip edges with points behind camera
            if depths[start] < 0 or depths[end] < 0:
                continue
                
            pt1 = tuple(corners_2d[start])
            pt2 = tuple(corners_2d[end])
            
            # Check if points are within reasonable image bounds
            h, w = img.shape[:2]
            if (0 <= pt1[0] < w * 2 and 0 <= pt1[1] < h * 2 and
                0 <= pt2[0] < w * 2 and 0 <= pt2[1] < h * 2):
                cv2.line(img, pt1, pt2, color, thickness, cv2.LINE_AA)
        
        # Highlight front face with thicker lines
        front_edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
        for start, end in front_edges:
            if depths[start] >= 0 and depths[end] >= 0:
                pt1 = tuple(corners_2d[start])
                pt2 = tuple(corners_2d[end])
                h, w = img.shape[:2]
                if (0 <= pt1[0] < w * 2 and 0 <= pt1[1] < h * 2 and
                    0 <= pt2[0] < w * 2 and 0 <= pt2[1] < h * 2):
                    cv2.line(img, pt1, pt2, color, thickness + 1, cv2.LINE_AA)
        
        # Draw text information
        if show_info:
            # Find top-left corner for text placement
            valid_corners = corners_2d[depths >= 0]
            if len(valid_corners) > 0:
                text_pos = tuple(valid_corners[np.argmin(valid_corners[:, 1])])
                text_pos = (max(5, text_pos[0]), max(20, text_pos[1] - 10))
                
                # Prepare text
                if score is not None:
                    text = f"{class_name}: {score:.2f}"
                else:
                    text = class_name
                
                # Add background for better readability
                (text_width, text_height), _ = cv2.getTextSize(
                    text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                )
                cv2.rectangle(
                    img,
                    (text_pos[0] - 2, text_pos[1] - text_height - 2),
                    (text_pos[0] + text_width + 2, text_pos[1] + 2),
                    (0, 0, 0),
                    -1
                )
                
                # Draw text
                cv2.putText(
                    img,
                    text,
                    text_pos,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                    cv2.LINE_AA
                )
        
        return img
    
    def visualize_detections(self,
                            image: np.ndarray,
                            detections: List[Dict],
                            P2: np.ndarray,
                            thickness: int = 2,
                            show_info: bool = True) -> np.ndarray:
        """
        Visualize multiple 3D detections on an image.
        
        Args:
            image: Input image
            detections: List of detection dictionaries
            P2: Camera projection matrix
            thickness: Line thickness
            show_info: Whether to show text info
            
        Returns:
            Visualization image
        """
        img = image.copy()
        
        for detection in detections:
            img = self.draw_3d_bbox(img, detection, P2, thickness, show_info)
        
        return img
    
    def save_frame(self,
                   image: np.ndarray,
                   detections: List[Dict],
                   P2: np.ndarray,
                   output_path: str,
                   thickness: int = 2,
                   show_info: bool = True) -> bool:
        """
        Visualize and save detections to file.
        
        Args:
            image: Input image
            detections: List of detections
            P2: Camera projection matrix
            output_path: Path to save visualization
            thickness: Line thickness
            show_info: Whether to show text info
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create output directory if needed
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Visualize
            vis_img = self.visualize_detections(image, detections, P2, thickness, show_info)
            
            # Save
            cv2.imwrite(output_path, vis_img)
            return True
            
        except Exception as e:
            print(f"Error saving visualization: {e}")
            return False


def load_kitti_calib(calib_file: str) -> Dict[str, np.ndarray]:
    """
    Load KITTI calibration file.
    
    Args:
        calib_file: Path to calibration file
        
    Returns:
        Dictionary of calibration matrices
    """
    calib = {}
    
    with open(calib_file, 'r') as f:
        for line in f.readlines():
            if ':' not in line:
                continue
                
            key, value = line.split(':', 1)
            calib[key] = np.array([float(x) for x in value.split()])
    
    # Reshape projection matrices
    for key in ['P0', 'P1', 'P2', 'P3']:
        if key in calib:
            calib[key] = calib[key].reshape(3, 4)
    
    # Reshape rect matrix
    if 'R0_rect' in calib:
        calib['R0_rect'] = calib['R0_rect'].reshape(3, 3)
    
    # Reshape Tr matrices
    if 'Tr_velo_to_cam' in calib:
        calib['Tr_velo_to_cam'] = calib['Tr_velo_to_cam'].reshape(3, 4)
    if 'Tr_imu_to_velo' in calib:
        calib['Tr_imu_to_velo'] = calib['Tr_imu_to_velo'].reshape(3, 4)
    
    return calib


# Example usage
if __name__ == "__main__":
    # Example detection in KITTI format
    example_detection = {
        'class': 'Car',
        'bbox': [100, 200, 300, 400],  # 2D bbox (optional)
        'dimensions': [1.5, 1.8, 4.0],  # height, width, length
        'location': [2.0, 1.5, 10.0],   # x, y, z in camera coords
        'rotation_y': 0.5,               # rotation around Y-axis
        'score': 0.95                    # confidence score
    }
    
    # Example camera calibration matrix (KITTI P2)
    P2 = np.array([
        [721.5377, 0.0, 609.5593, 44.85728],
        [0.0, 721.5377, 172.854, 0.2163791],
        [0.0, 0.0, 1.0, 0.002745884]
    ])
    
    # Create visualizer
    visualizer = DetectionVisualizer()
    
    # Create example image
    img = np.ones((375, 1242, 3), dtype=np.uint8) * 200
    
    # Visualize
    vis_img = visualizer.draw_3d_bbox(img, example_detection, P2)
    
    print("Visualization example completed!")
    print("DetectionVisualizer class ready for use.")
