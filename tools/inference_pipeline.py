import torch
from pathlib import Path
from visualization import DetectionVisualizer
from gif_generator import FrameToGifConverter
import cv2

class InferencePipeline:
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.visualizer = DetectionVisualizer('./vis_results')
    
    def inference_and_visualize(self, image_dir, calibration_matrix, save_gif=True):
        """
        批量推理并可视化结果
        """
        image_dir = Path(image_dir)
        images = sorted(image_dir.glob('*.jpg')) + sorted(image_dir.glob('*.png'))
        
        for idx, img_path in enumerate(images):
            # 读取图像
            image = cv2.imread(str(img_path))
            
            # 推理
            with torch.no_grad():
                detections = self.model.inference(image)
            
            # 可视化
            vis_image = self.visualizer.draw_3d_bbox(
                image, detections, calibration_matrix
            )
            
            # 保存帧
            self.visualizer.save_frame(vis_image, idx)
        
        # 生成GIF
        if save_gif:
            converter = FrameToGifConverter(
                self.visualizer.output_dir,
                output_path='inference_results.gif',
                fps=10
            )
            converter.create_gif(duration=100)
            converter.create_video(fps=10)
        
        return self.visualizer.output_dir
