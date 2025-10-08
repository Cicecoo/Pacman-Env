"""
训练DQN Agent
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 添加路径
sys.path.append(str(Path(__file__).parent))

from agents.dqn_agent import DQNAgent
from pacman_env import PacmanEnv

# 动作名称映射
PACMAN_ACTIONS = ['North', 'South', 'East', 'West', 'Stop']


def get_legal_action_indices(env):
    """
    获取合法动作的索引列表
    
    Args:
        env: Pacman环境
    
    Returns:
        legal_action_indices: 合法动作的索引列表
    """
    legal_actions = env.game.state.getLegalPacmanActions()
    legal_action_indices = [i for i, action_name in enumerate(PACMAN_ACTIONS) 
                           if action_name in legal_actions]
    return legal_action_indices


def train_dqn(
    num_episodes=1000,
    max_steps_per_episode=200,
    save_freq=50,
    render=False,
    checkpoint_path=None,
    layout=None,
    random_layout=True
):
    """
    训练DQN Agent
    
    Args:
        num_episodes: 训练回合数
        max_steps_per_episode: 每回合最大步数
        save_freq: 保存模型频率
        render: 是否渲染
        checkpoint_path: 检查点路径（用于继续训练）
        layout: 指定地图名称 (例如: 'smallClassic', 'mediumClassic')
                如果为None且random_layout=False，使用默认地图
        random_layout: 是否使用随机地图生成（会覆盖layout参数）
    """
    # 创建环境
    env = PacmanEnv(
        render_mode="human" if render else None,
        use_dict_obs=True,
        max_ghosts=5,
        use_graphics=render,  # 训练时不使用图形可以加速
        episode_length=max_steps_per_episode
    )
    
    # 创建agent
    agent = DQNAgent(
        action_space_size=5,
        max_map_size=20,
        max_ghosts=5,
        max_capsules=10,
        top_k_foods=10,
        use_grid_encoding=False,  # 不使用网格编码（训练更快）
        hidden_dims=[256, 256],
        learning_rate=0.0005,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        buffer_capacity=10000,
        batch_size=64,
        target_update_freq=10
    )
    
    # 加载检查点（如果存在）
    if checkpoint_path and Path(checkpoint_path).exists():
        agent.load(checkpoint_path)
        print(f"Resumed from checkpoint: {checkpoint_path}")
    
    # 训练统计
    episode_rewards = []
    episode_lengths = []
    episode_losses = []
    
    print("\n" + "="*60)
    print("Starting DQN Training")
    print("="*60)
    print(f"Episodes: {num_episodes}")
    print(f"Max steps per episode: {max_steps_per_episode}")
    if random_layout:
        print(f"Layout: Random generation")
    elif layout:
        print(f"Layout: {layout}")
    else:
        print(f"Layout: Default")
    print("="*60)
    print()
    
    try:
        for episode in range(agent.episode_count, agent.episode_count + num_episodes):
            # Reset with specified layout
            if random_layout:
                obs, info = env.reset()  # Use random layout
            else:
                obs, info = env.reset(layout=layout)  # Use specified or default layout
            episode_reward = 0
            episode_loss = []
            step = 0
            
            for step in range(max_steps_per_episode):
                # 获取合法动作
                legal_action_indices = get_legal_action_indices(env)
                
                # 选择动作（只从合法动作中选择）
                action = agent.select_action(obs, legal_actions=legal_action_indices)
                
                # 执行动作
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                # 存储经验
                agent.store_transition(obs, action, reward, next_obs, done)
                
                # 训练
                loss = agent.train_step()
                if loss is not None:
                    episode_loss.append(loss)
                
                episode_reward += reward
                obs = next_obs
                
                if done:
                    break
            
            # 回合结束
            agent.end_episode()
            
            # 记录统计信息
            episode_rewards.append(episode_reward)
            episode_lengths.append(step + 1)
            avg_loss = np.mean(episode_loss) if episode_loss else 0
            episode_losses.append(avg_loss)
            
            # 打印进度
            if (episode + 1) % 10 == 0:
                recent_rewards = episode_rewards[-10:]
                avg_reward = np.mean(recent_rewards)
                max_reward = np.max(recent_rewards)
                min_reward = np.min(recent_rewards)
                
                print(f"\nEpisode {episode + 1}/{agent.episode_count + num_episodes}")
                print(f"  Avg Reward (last 10): {avg_reward:.1f}")
                print(f"  Max Reward (last 10): {max_reward:.1f}")
                print(f"  Min Reward (last 10): {min_reward:.1f}")
                print(f"  Epsilon: {agent.epsilon:.3f}")
                print(f"  Avg Loss: {avg_loss:.4f}")
                print(f"  Buffer Size: {len(agent.replay_buffer)}")
                print(f"  Total Steps: {agent.total_steps}")
            
            # 保存模型
            if (episode + 1) % save_freq == 0:
                save_path = f"checkpoints/dqn_agent_ep{episode+1}.pth"
                Path("checkpoints").mkdir(exist_ok=True)
                agent.save(save_path)
                
                # 打印详细统计信息
                agent.print_stats()
                
                # 保存训练曲线
                plot_training_curves(episode_rewards, episode_losses, 
                                    save_path=f"checkpoints/dqn_training_ep{episode+1}.png")
    
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")
    
    finally:
        # 保存最终模型
        final_path = "checkpoints/dqn_agent_final.pth"
        Path("checkpoints").mkdir(exist_ok=True)
        agent.save(final_path)
        
        # 保存最终训练曲线
        plot_training_curves(episode_rewards, episode_losses, 
                           save_path="checkpoints/dqn_training_final.png")
        
        env.close()
        
        print("\n" + "="*60)
        print("Training Complete!")
        print("="*60)
        print(f"Total Episodes: {len(episode_rewards)}")
        print(f"Total Steps: {agent.total_steps}")
        
        # 安全地打印统计信息（避免空列表错误）
        if len(episode_rewards) > 0:
            print(f"Average Reward: {np.mean(episode_rewards):.1f}")
            print(f"Max Reward: {np.max(episode_rewards):.1f}")
            print(f"Min Reward: {np.min(episode_rewards):.1f}")
        else:
            print("No episodes completed.")
        
        print(f"Final Epsilon: {agent.epsilon:.4f}")
        print("="*60)
        
        # 打印最终统计信息
        if len(episode_rewards) > 0:
            agent.print_stats()
    
    return agent, episode_rewards, episode_losses


def plot_training_curves(rewards, losses, save_path=None):
    """绘制训练曲线"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # 奖励曲线
    ax1.plot(rewards, alpha=0.3, label='Episode Reward')
    
    # 计算移动平均
    if len(rewards) > 20:
        window = 20
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        ax1.plot(range(window-1, len(rewards)), moving_avg, 
                label=f'Moving Avg (window={window})', linewidth=2)
    
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title('Training Rewards')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 损失曲线
    if losses:
        ax2.plot(losses, alpha=0.3, label='Episode Loss')
        
        # 计算移动平均
        if len(losses) > 20:
            window = 20
            moving_avg = np.convolve(losses, np.ones(window)/window, mode='valid')
            ax2.plot(range(window-1, len(losses)), moving_avg, 
                    label=f'Moving Avg (window={window})', linewidth=2)
        
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Loss')
        ax2.set_title('Training Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"Training curves saved to {save_path}")
    
    plt.close()


def evaluate_dqn(checkpoint_path, num_episodes=10, render=True, layout=None, random_layout=True):
    """
    评估训练好的DQN Agent
    
    Args:
        checkpoint_path: 模型检查点路径
        num_episodes: 评估回合数
        render: 是否渲染
        layout: 指定地图名称 (例如: 'smallClassic', 'mediumClassic')
        random_layout: 是否使用随机地图生成（会覆盖layout参数）
    """
    # 创建环境
    env = PacmanEnv(
        render_mode="human" if render else None,
        use_dict_obs=True,
        max_ghosts=5,
        use_graphics=render,
        episode_length=200
    )
    
    # 创建agent并加载模型
    agent = DQNAgent(
        action_space_size=5,
        max_map_size=20,
        max_ghosts=5,
        max_capsules=10,
        top_k_foods=10,
        use_grid_encoding=False
    )
    agent.load(checkpoint_path)
    agent.eval()  # Set to evaluation mode (no exploration)
    
    print("\n" + "="*60)
    print("Evaluating DQN Agent")
    print("="*60)
    print(f"Model: {checkpoint_path}")
    print(f"Episodes: {num_episodes}")
    if random_layout:
        print(f"Layout: Random generation")
    elif layout:
        print(f"Layout: {layout}")
    else:
        print(f"Layout: Default")
    print("="*60)
    print()
    
    test_rewards = []
    wins = 0
    losses = 0
    
    for episode in range(num_episodes):
        # Reset with specified layout
        if random_layout:
            obs, info = env.reset()  # Use random layout
        else:
            obs, info = env.reset(layout=layout)  # Use specified or default layout
        episode_reward = 0
        step = 0
        
        while True:
            # 获取合法动作
            legal_action_indices = get_legal_action_indices(env)
            
            # 使用贪婪策略（epsilon=0）
            action = agent.select_action(obs, epsilon=0.0, legal_actions=legal_action_indices)
            
            # 执行动作
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            obs = next_obs
            step += 1
            
            if done:
                break
        
        test_rewards.append(episode_reward)
        
        # Track wins/losses
        if episode_reward > 0:
            wins += 1
            result = "WIN "
        else:
            losses += 1
            result = "LOSS"
        
        print(f"Episode {episode + 1:3d}/{num_episodes} | "
              f"{result} | "
              f"Reward: {episode_reward:7.1f} | "
              f"Steps: {step:3d}")
    
    env.close()
    
    print("\n" + "="*60)
    print("Evaluation Results:")
    print("="*60)
    print(f"Total Episodes:      {num_episodes}")
    print(f"Wins:                {wins} ({wins/num_episodes*100:.1f}%)")
    print(f"Losses:              {losses} ({losses/num_episodes*100:.1f}%)")
    print(f"Avg Reward:          {np.mean(test_rewards):.2f} ± {np.std(test_rewards):.2f}")
    print(f"Best Reward:         {np.max(test_rewards):.2f}")
    print(f"Worst Reward:        {np.min(test_rewards):.2f}")
    print("="*60)
    
    return test_rewards


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train or evaluate DQN agent')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'eval'],
                       help='Mode: train or eval')
    parser.add_argument('--episodes', type=int, default=500,
                       help='Number of episodes')
    parser.add_argument('--max-steps', type=int, default=200,
                       help='Maximum steps per episode')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Checkpoint path for loading/resuming')
    parser.add_argument('--layout', type=str, default=None,
                       help='Specific layout name (e.g., smallClassic, mediumClassic, testClassic)')
    parser.add_argument('--random-layout', action='store_true',
                       help='Use random layout generation (ignores --layout)')
    parser.add_argument('--render', action='store_true',
                       help='Render the environment')
    
    args = parser.parse_args()

    # 默认配置
    if args.layout is None:
        args.layout = 'smallClassic.lay'
    
    if args.mode == 'train':
        train_dqn(
            num_episodes=args.episodes,
            max_steps_per_episode=args.max_steps,
            save_freq=50,
            render=args.render,
            checkpoint_path=args.checkpoint,
            layout=args.layout,
            random_layout=args.random_layout
        )
        # 训练后评估
        evaluate_dqn(
            checkpoint_path="checkpoints/dqn_agent_final.pth",
            num_episodes=10,
            render=True,
            layout=args.layout,
            random_layout=args.random_layout
        )
        
    else:
        if args.checkpoint is None:
            args.checkpoint = "checkpoints/dqn_agent_final.pth"
        evaluate_dqn(
            checkpoint_path=args.checkpoint,
            num_episodes=args.episodes,
            render=args.render,
            layout=args.layout,
            random_layout=args.random_layout
        )
