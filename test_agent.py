from agents.q_learning_agent import QLearningAgent
from agents.approximate_q_learning_agent import ApproximateQLearningAgent   

from pacman_env import PacmanEnv

# 可选：导入视频录制功能（如果需要录制视频）
try:
    from video_recorder import RecordableEnvWrapper
    VIDEO_RECORDING_AVAILABLE = True
except ImportError:
    VIDEO_RECORDING_AVAILABLE = False
    print("Warning: video_recorder not available. Videos will not be recorded.")

PACMAN_ACTIONS = {
    'North': 0,
    'South': 1,
    'East': 2,
    'West': 3,
    'Stop': 4
}

layout = 'originalClassic.lay'

checkpoint = 'checkpoints/approx_q_learning_agent-smallClassic-1000ep-a0.2-e0.1-g0.8-EnhenceEx-CanEatCapsule.pkl'

# 配置选项
ENABLE_VIDEO_RECORDING = True  # 设置为True以启用视频录制
VIDEO_OUTPUT_DIR = 'videos/approx_q_learning/SonO'      # 视频输出目录
VIDEO_FPS = 10                   # 视频帧率

# load and test the trained agent
if __name__ == "__main__":
    # 创建基础环境
    base_env = PacmanEnv(use_graphics=True)
    
    # 如果启用视频录制，包装环境
    if ENABLE_VIDEO_RECORDING and VIDEO_RECORDING_AVAILABLE:
        env = RecordableEnvWrapper(
            base_env,
            output_dir=VIDEO_OUTPUT_DIR,
            fps=VIDEO_FPS,
            record_all=True,
            auto_save=True
        )
        print(f"Video recording enabled. Videos will be saved to: {VIDEO_OUTPUT_DIR}")
    else:
        env = base_env
        if ENABLE_VIDEO_RECORDING:
            print("Video recording requested but not available. Please install:")
            print("  pip install imageio imageio-ffmpeg")
    
    # 加载agent
    agent = ApproximateQLearningAgent(alpha=0, epsilon=0, gamma=0.8)
    agent.set_test_mode()
    agent.load(checkpoint)
    test_episodes = 10

    for episode in range(test_episodes):
        obs, info = env.reset(layout=layout)
        cur_state = env.game.state
        agent.register_initial_state(cur_state)
        done = False

        while not done:
            action = agent.choose_action(cur_state)
            next_obs, reward, terminated, truncated, info = env.step(PACMAN_ACTIONS[action])   
            done = terminated or truncated

            new_state = env.game.state
            agent.observe_transition(new_state)
            cur_state = new_state

        agent.reach_terminal_state(cur_state)
        
        # 显示结果（如果有视频路径，也显示出来）
        result_msg = f"Episode {episode + 1}/{test_episodes}, Score: {agent.episode_rewards}"
        if 'video_path' in info:
            result_msg += f", Video: {info['video_path']}"
        print(result_msg)

    print("Testing completed.")
    