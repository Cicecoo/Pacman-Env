"""
视频录制器工具
支持逐episode录制Pacman游戏视频
"""

import os
import numpy as np
from datetime import datetime
from PIL import Image

class VideoRecorder:
    """
    视频录制器，用于录制每个episode的游戏画面
    支持两种后端：cv2 (OpenCV) 或 imageio
    """
    
    def __init__(self, output_dir='videos', fps=10, backend='imageio'):
        """
        初始化视频录制器
        
        Args:
            output_dir: 视频输出目录
            fps: 帧率 (frames per second)
            backend: 'cv2' 或 'imageio'
        """
        self.output_dir = output_dir
        self.fps = fps
        self.backend = backend
        self.frames = []
        self.is_recording = False
        self.current_episode = 0
        self.video_path = None
        
        # 创建输出目录
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created video directory: {output_dir}")
        
        # 检查并导入后端库
        self._init_backend()
    
    def _init_backend(self):
        """初始化视频编码后端"""
        if self.backend == 'cv2':
            try:
                import cv2
                self.cv2 = cv2
                print("Using OpenCV (cv2) backend for video recording")
            except ImportError:
                print("Warning: OpenCV not found, falling back to imageio")
                self.backend = 'imageio'
                self._init_imageio()
        else:
            self._init_imageio()
    
    def _init_imageio(self):
        """初始化 imageio 后端"""
        try:
            import imageio
            self.imageio = imageio
            print("Using imageio backend for video recording")
        except ImportError:
            print("Error: Neither cv2 nor imageio is available!")
            print("Please install one of them:")
            print("  pip install opencv-python")
            print("  OR")
            print("  pip install imageio imageio-ffmpeg")
            self.backend = None
    
    def start_recording(self, episode_num=None, prefix='episode'):
        """
        开始录制新的episode
        
        Args:
            episode_num: episode编号（如果为None则自动递增）
            prefix: 文件名前缀
        """
        if self.backend is None:
            return
        
        if episode_num is not None:
            self.current_episode = episode_num
        else:
            self.current_episode += 1
        
        # 生成文件名（包含时间戳）
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}_{self.current_episode:04d}_{timestamp}.mp4"
        self.video_path = os.path.join(self.output_dir, filename)
        
        self.frames = []
        self.is_recording = True
        print(f"Started recording episode {self.current_episode}")
    
    def capture_frame(self, frame):
        """
        捕获一帧画面
        
        Args:
            frame: numpy array (H, W, 3) 或 PIL Image
        """
        if not self.is_recording or self.backend is None:
            return
        
        # 转换为numpy array
        if isinstance(frame, Image.Image):
            frame = np.array(frame)
        
        # 确保是RGB格式
        if len(frame.shape) == 2:  # 灰度图
            frame = np.stack([frame] * 3, axis=-1)
        elif frame.shape[-1] == 4:  # RGBA
            frame = frame[:, :, :3]
        
        self.frames.append(frame.copy())
    
    def stop_recording(self, save=True):
        """
        停止录制并保存视频
        
        Args:
            save: 是否保存视频（如果为False则丢弃该episode的录制）
        
        Returns:
            video_path: 保存的视频文件路径（如果save=False则返回None）
        """
        if not self.is_recording or self.backend is None:
            return None
        
        self.is_recording = False
        
        if not save or len(self.frames) == 0:
            print(f"Episode {self.current_episode} recording discarded (no frames or save=False)")
            self.frames = []
            return None
        
        # 保存视频
        try:
            if self.backend == 'cv2':
                self._save_video_cv2()
            else:
                self._save_video_imageio()
            
            print(f"Episode {self.current_episode} saved: {self.video_path} ({len(self.frames)} frames)")
            return self.video_path
        except Exception as e:
            print(f"Error saving video: {e}")
            return None
        finally:
            self.frames = []
    
    def _save_video_cv2(self):
        """使用OpenCV保存视频"""
        if len(self.frames) == 0:
            return
        
        height, width = self.frames[0].shape[:2]
        fourcc = self.cv2.VideoWriter_fourcc(*'mp4v')
        out = self.cv2.VideoWriter(self.video_path, fourcc, self.fps, (width, height))
        
        for frame in self.frames:
            # OpenCV使用BGR格式
            frame_bgr = self.cv2.cvtColor(frame, self.cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
        
        out.release()
    
    def _save_video_imageio(self):
        """使用imageio保存视频"""
        if len(self.frames) == 0:
            return
        
        # imageio使用RGB格式
        self.imageio.mimsave(self.video_path, self.frames, fps=self.fps)
    
    def __del__(self):
        """析构时确保停止录制"""
        if self.is_recording:
            self.stop_recording(save=False)


class RecordableEnvWrapper:
    """
    环境包装器，为Pacman环境添加视频录制功能
    """
    
    def __init__(self, env, recorder=None, output_dir='videos', fps=10, 
                 record_all=True, auto_save=True):
        """
        Args:
            env: Pacman环境实例
            recorder: VideoRecorder实例（如果为None则自动创建）
            output_dir: 视频输出目录
            fps: 视频帧率
            record_all: 是否录制所有episode
            auto_save: 是否自动保存视频（如果为False，需要手动调用save_video）
        """
        self.env = env
        self.record_all = record_all
        self.auto_save = auto_save
        self.current_episode = 0
        
        if recorder is None:
            self.recorder = VideoRecorder(output_dir=output_dir, fps=fps)
        else:
            self.recorder = recorder
    
    def reset(self, **kwargs):
        """重置环境并开始新的episode录制"""
        obs, info = self.env.reset(**kwargs)
        
        self.current_episode += 1
        
        # 如果需要录制，开始新的录制
        if self.record_all or self.should_record_episode():
            self.recorder.start_recording(episode_num=self.current_episode)
            # 捕获初始帧
            self._capture_current_frame()
        
        return obs, info
    
    def step(self, action):
        """执行动作并捕获帧"""
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # 捕获当前帧
        if self.recorder.is_recording:
            self._capture_current_frame()
        
        # 如果episode结束，停止录制
        if (terminated or truncated) and self.recorder.is_recording:
            if self.auto_save:
                video_path = self.recorder.stop_recording(save=True)
                info['video_path'] = video_path
            else:
                # 不自动保存，等待手动调用
                info['recording_active'] = True
        
        return obs, reward, terminated, truncated, info
    
    def _capture_current_frame(self):
        """捕获当前游戏画面"""
        try:
            # 尝试从display获取图像
            if hasattr(self.env, 'display') and hasattr(self.env.display, 'image'):
                frame = self.env.display.image
                self.recorder.capture_frame(frame)
            # 或者使用_get_image方法
            elif hasattr(self.env, '_get_image'):
                frame = self.env._get_image()
                self.recorder.capture_frame(frame)
        except Exception as e:
            print(f"Warning: Could not capture frame: {e}")
    
    def should_record_episode(self):
        """
        判断是否应该录制当前episode
        可以被子类重写来实现自定义的录制策略
        例如：只录制特定的episode或满足特定条件的episode
        """
        return True
    
    def save_video(self):
        """手动保存当前episode的视频"""
        if self.recorder.is_recording:
            return self.recorder.stop_recording(save=True)
        return None
    
    def discard_video(self):
        """丢弃当前episode的录制"""
        if self.recorder.is_recording:
            return self.recorder.stop_recording(save=False)
        return None
    
    def __getattr__(self, name):
        """代理其他属性到原始环境"""
        return getattr(self.env, name)


# 便捷函数
def create_recordable_env(env, output_dir='videos', fps=10, record_all=True):
    """
    创建支持视频录制的环境
    
    Args:
        env: 原始环境
        output_dir: 视频输出目录
        fps: 帧率
        record_all: 是否录制所有episode
    
    Returns:
        包装后的环境
    """
    return RecordableEnvWrapper(env, output_dir=output_dir, fps=fps, record_all=record_all)
