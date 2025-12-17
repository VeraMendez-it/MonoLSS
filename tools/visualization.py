import cv2
import numpy as np
from pathlib import Path

class DetectionVisualizer:
    def __init__(self, output_dir='./vis_results'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
    
    def draw_3d_bbox(self, image, detections, calibration_matrix):
        """绘制3D检测框的2D投影"""
        vis_image = image.copy()
        
        for det in detections:
            # 提取检测信息
            bbox_2d = det['bbox_2d']  # [x1, y1, x2, y2]
            bbox_3d = det['bbox_3d']  # 3D框顶点
            confidence = det['confidence']
            
            # 绘制2D检测框
            cv2.rectangle(vis_image, 
                         (int(bbox_2d[0]), int(bbox_2d[1])),
                         (int(bbox_2d[2]), int(bbox_2d[3])),
                         (0, 255, 0), 2)
            
            # 投影3D框到2D
            projected_points = self._project_3d_to_2d(bbox_3d, calibration_matrix)
            self._draw_3d_box(vis_image, projected_points)
            
            # 添加置信度标签
            cv2.putText(vis_image, f'Conf: {confidence:.2f}',
                       (int(bbox_2d[0]), int(bbox_2d[1])-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return vis_image
    
    def _project_3d_to_2d(self, bbox_3d, K):
        """将3D框投影到2D图像平面"""
        points_2d = []
        for point_3d in bbox_3d:  
            point_2d = K @ point_3d[:3]
            point_2d = point_2d[:2] / point_2d[2]
            points_2d.append(point_2d)
        return np.array(points_2d)
    
    def _draw_3d_box(self, image, points_2d, color=(0, 0, 255)):
        """绘制3D框的边界线"""
        # 定义立方体的边
        edges = [(0,1), (1,3), (3,2), (2,0), (4,5), (5,7), 
                 (7,6), (6,4), (0,4), (1,5), (2,6), (3,7)]
        
        for edge in edges:  
            pt1 = tuple(map(int, points_2d[edge[0]]))
            pt2 = tuple(map(int, points_2d[edge[1]]))
            cv2.line(image, pt1, pt2, color, 1)
    
    def save_frame(self, image, frame_id):
        """保存单帧图像"""
        output_path = self.output_dir / f'frame_{frame_id:06d}.png'
        cv2.imwrite(str(output_path), image)
        return output_path