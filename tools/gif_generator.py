import cv2
import imageio
from pathlib import Path
from tqdm import tqdm

class FrameToGifConverter:
    def __init__(self, frame_dir, output_path='output.gif', fps=10):
        self.frame_dir = Path(frame_dir)
        self.output_path = output_path
        self.fps = fps
    
    def create_gif(self, duration=100):
        """
        从连续帧创建GIF动画
        Args:
            duration: 每帧持续时间（毫秒）
        """
        # 获取所有帧文件，按顺序排序
        frames = sorted(self.frame_dir.glob('frame_*.png'))
        
        if not frames:
            print(f"未找到帧文件在: {self.frame_dir}")
            return
        
        print(f"发现 {len(frames)} 帧，正在转换...")
        
        # 读取所有帧
        images = []
        for frame_path in tqdm(frames, desc="加载帧"):
            img = imageio.imread(str(frame_path))
            images.append(img)
        
        # 创建GIF
        imageio.mimsave(self.output_path, images, duration=duration, loop=0)
        print(f"GIF已保存到: {self.output_path}")
        
        return self.output_path
    
    def create_video(self, output_video='output.mp4', fps=10):
        """也可以生成MP4视频"""
        frames = sorted(self.frame_dir.glob('frame_*.png'))
        
        if not frames:
            return
        
        # 读取第一帧获取尺寸
        first_frame = cv2.imread(str(frames[0]))
        height, width = first_frame.shape[:2]
        
        # 初始化视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
        
        print(f"正在生成视频: {output_video}")
        for frame_path in tqdm(frames, desc="处理帧"):
            frame = cv2.imread(str(frame_path))
            out.write(frame)
        
        out.release()
        print(f"视频已保存到: {output_video}")