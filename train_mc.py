"""
Training script for Monte Carlo Agent on Pacman Environment

This script demonstrates how to train and evaluate the MC agent.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gymnasium
import numpy as np
import matplotlib.pyplot as plt
from pacman_env.envs.pacman_env import PacmanEnv
from pacman_env.envs.improved_pacman_env import ImprovedPacmanEnv
from agents.mc_agent import MCAgent
# import gymnasium
# import pacman_env


def train_agent(
    num_episodes: int = 1000,
    max_steps: int = 100,
    save_every: int = 100,
    model_path: str = "model/mc_agent.pkl",
    layout: str = None,
    random_layout: bool = True,
    render: bool = False
):
    """
    Train the MC agent.
    
    Args:
        num_episodes: Number of training episodes
        max_steps: Maximum steps per episode
        save_every: Save model every N episodes
        model_path: Path to save model
        layout: Specific layout name (e.g., 'smallClassic', 'mediumClassic')
                If None and random_layout=False, uses default layout
        random_layout: Whether to use random layout generation (overrides layout)
        no_render: Disable rendering (default True for faster training)
    """
    # Create environment with graphics disabled by default for training
    env = PacmanEnv(use_graphics=render, episode_length=max_steps)
    # env = ImprovedPacmanEnv(use_graphics=render, episode_length=max_steps)
    
    # Create agent
    agent = MCAgent(
        action_space_size=5,
        gamma=0.99,
        epsilon=1.0,  # Start with high exploration
        epsilon_decay=0.995,
        epsilon_min=0.01,
        learning_rate=0.1
    )
    
    # Training statistics
    episode_rewards = []
    episode_lengths = []
    win_rate_window = []
    
    print("="*60)
    print(f"Starting MC Agent Training")
    print("="*60)
    print(f"Episodes: {num_episodes}")
    print(f"Max steps per episode: {max_steps}")
    if random_layout:
        print(f"Layout: Random generation")
    elif layout:
        print(f"Layout: {layout}")
    else:
        print(f"Layout: Default")
    print(f"Save path: {model_path}")
    print("="*60)
    print()
    
    agent.train()  # Set to training mode
    
    for episode in range(num_episodes):
        # Reset with specified layout
        if random_layout:
            obs, info = env.reset()  # Use random layout
        else:
            obs, info = env.reset(layout=layout)  # Use specified or default layout
        episode_reward = 0
        steps = 0
        
        for step in range(max_steps):
            # Get legal actions from environment
            legal_actions = env.get_legal_actions()
            
            # Select action (only from legal actions)
            action = agent.select_action(obs, legal_actions)
            
            # Store current transition
            agent.store_transition(obs, action, 0)  # Reward will be updated
            
            # Take action
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            # Update reward in last transition
            if len(agent.episode_buffer) > 0:
                state, act, _ = agent.episode_buffer[-1]
                agent.episode_buffer[-1] = (state, act, reward)
            
            episode_reward += reward
            steps += 1
            agent.total_steps += 1
            
            obs = next_obs
            
            if terminated or truncated:
                break
        
        # End episode and update Q-values
        agent.end_episode()
        
        # Record statistics
        episode_rewards.append(episode_reward)
        episode_lengths.append(steps)
        
        # Win/loss tracking (simplified: positive reward = win)
        win_rate_window.append(1 if episode_reward > 0 else 0)
        if len(win_rate_window) > 100:
            win_rate_window.pop(0)
        
        # Print progress
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            avg_length = np.mean(episode_lengths[-100:])
            win_rate = np.mean(win_rate_window) * 100
            
            print(f"Episode {episode + 1:4d}/{num_episodes} | "
                  f"Reward: {episode_reward:7.1f} | "
                  f"Avg(100): {avg_reward:7.1f} | "
                  f"Steps: {steps:3d} | "
                  f"Win%: {win_rate:5.1f} | "
                  f"ε: {agent.epsilon:.3f}")
        
        # Save model periodically
        if (episode + 1) % save_every == 0:
            agent.save(model_path)
            print(f"  → Model saved at episode {episode + 1}")
    
    # Final save
    agent.save(model_path)
    
    # Print final statistics
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    agent.print_stats()
    
    # Plot training curves
    plot_training_curves(episode_rewards, episode_lengths, win_rate_window)
    
    env.close()
    
    return agent, episode_rewards, episode_lengths


def evaluate_agent(
    model_path: str = "model/mc_agent.pkl",
    num_episodes: int = 100,
    max_steps: int = 100,
    render: bool = True,
    layout: str = None,
    random_layout: bool = True
):
    """
    Evaluate a trained agent.
    
    Args:
        model_path: Path to load model
        num_episodes: Number of evaluation episodes
        max_steps: Maximum steps per episode
        render: Whether to render the environment
        layout: Specific layout name (e.g., 'smallClassic', 'mediumClassic')
        random_layout: Whether to use random layout generation (overrides layout)
    """
    # Create environment - enable graphics for evaluation if render is True
    env = PacmanEnv(use_graphics=render)
    # env = ImprovedPacmanEnv(use_graphics=render, episode_length=max_steps)
    
    # Create and load agent
    agent = MCAgent()
    agent.load(model_path)
    agent.eval()  # Set to evaluation mode (no exploration)
    
    print("\n" + "="*60)
    print(f"Evaluating MC Agent")
    print("="*60)
    print(f"Model: {model_path}")
    print(f"Episodes: {num_episodes}")
    if random_layout:
        print(f"Layout: Random generation")
    elif layout:
        print(f"Layout: {layout}")
    else:
        print(f"Layout: Default")
    print("="*60)
    print()
    
    # Evaluation statistics
    episode_rewards = []
    episode_lengths = []
    wins = 0
    losses = 0
    
    for episode in range(num_episodes):
        # Reset with specified layout
        if random_layout:
            obs, info = env.reset()  # Use random layout
        else:
            obs, info = env.reset(layout=layout)  # Use specified or default layout
        episode_reward = 0
        steps = 0
        
        for step in range(max_steps):
            # Get legal actions
            legal_actions = env.get_legal_actions()
            
            # Select action (greedy, only from legal actions)
            action = agent.select_action(obs, legal_actions)
            
            # Take action
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Step {steps+1}: Action {action}, Reward {reward}")
            
            episode_reward += reward
            steps += 1
            
            if terminated or truncated:
                break
        
        # Record statistics
        episode_rewards.append(episode_reward)
        episode_lengths.append(steps)
        
        if episode_reward > 0:
            wins += 1
        else:
            losses += 1
        
        # Print episode result
        result = "WIN " if episode_reward > 0 else "LOSS"
        print(f"Episode {episode + 1:3d}/{num_episodes} | "
              f"{result} | "
              f"Reward: {episode_reward:7.1f} | "
              f"Steps: {steps:3d}")
    
    # Print evaluation summary
    print("\n" + "="*60)
    print("Evaluation Summary")
    print("="*60)
    print(f"Total Episodes:      {num_episodes}")
    print(f"Wins:                {wins} ({wins/num_episodes*100:.1f}%)")
    print(f"Losses:              {losses} ({losses/num_episodes*100:.1f}%)")
    print(f"Avg Reward:          {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Avg Episode Length:  {np.mean(episode_lengths):.2f} ± {np.std(episode_lengths):.2f}")
    print(f"Best Reward:         {np.max(episode_rewards):.2f}")
    print(f"Worst Reward:        {np.min(episode_rewards):.2f}")
    print("="*60 + "\n")
    
    env.close()
    
    return episode_rewards, episode_lengths


def plot_training_curves(episode_rewards, episode_lengths, win_rate_window):
    """Plot training curves."""
    try:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Reward curve
        axes[0, 0].plot(episode_rewards, alpha=0.3, color='blue')
        if len(episode_rewards) > 100:
            smoothed = np.convolve(episode_rewards, np.ones(100)/100, mode='valid')
            axes[0, 0].plot(range(99, len(episode_rewards)), smoothed, color='red', linewidth=2)
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Total Reward')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Episode length
        axes[0, 1].plot(episode_lengths, alpha=0.3, color='green')
        if len(episode_lengths) > 100:
            smoothed = np.convolve(episode_lengths, np.ones(100)/100, mode='valid')
            axes[0, 1].plot(range(99, len(episode_lengths)), smoothed, color='red', linewidth=2)
        axes[0, 1].set_title('Episode Length')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Steps')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Win rate (last 100 episodes)
        win_rates = []
        for i in range(len(episode_rewards)):
            start_idx = max(0, i - 99)
            window = [1 if r > 0 else 0 for r in episode_rewards[start_idx:i+1]]
            win_rates.append(np.mean(window) * 100)
        axes[1, 0].plot(win_rates, color='purple', linewidth=2)
        axes[1, 0].set_title('Win Rate (100-episode window)')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Win Rate (%)')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_ylim([0, 105])
        
        # Reward distribution
        axes[1, 1].hist(episode_rewards, bins=50, color='orange', alpha=0.7, edgecolor='black')
        axes[1, 1].set_title('Reward Distribution')
        axes[1, 1].set_xlabel('Total Reward')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('training_curves.png', dpi=150)
        print("\nTraining curves saved to 'training_curves.png'")
        
    except Exception as e:
        print(f"Could not plot training curves: {e}")


def main():
    """Main function to run training or evaluation."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train or evaluate MC agent for Pacman')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'eval'],
                       help='Mode: train or eval')
    parser.add_argument('--episodes', type=int, default=1000,
                       help='Number of episodes')
    parser.add_argument('--max-steps', type=int, default=100,
                       help='Maximum steps per episode')
    parser.add_argument('--model-path', type=str, default='checkpoints/mc_agent.pkl',
                       help='Path to save/load model')
    parser.add_argument('--save-every', type=int, default=100,
                       help='Save model every N episodes (train mode)')
    parser.add_argument('--layout', type=str, default=None,
                       help='Specific layout name (e.g., smallClassic, mediumClassic, testClassic)')
    parser.add_argument('--random-layout', action='store_true',
                       help='Use random layout generation (ignores --layout)')
    
    args = parser.parse_args()

    # args.layout = 'mediumClassic_noGhosts.lay'
    args.layout = 'smallClassic.lay'
    args.max_steps = 1000

    
    if args.mode == 'train':
        train_agent(
            num_episodes=args.episodes,
            max_steps=args.max_steps,
            save_every=args.save_every,
            model_path=args.model_path,
            layout=args.layout,
            random_layout=args.random_layout,
            render=False  # Disable rendering during training for speed
        )
        evaluate_agent(
            model_path=args.model_path,
            num_episodes=args.episodes,
            max_steps=args.max_steps,
            layout=args.layout,
            random_layout=args.random_layout
        )

    elif args.mode == 'eval':
        evaluate_agent(
            model_path=args.model_path,
            num_episodes=args.episodes,
            max_steps=args.max_steps,
            layout=args.layout,
            random_layout=args.random_layout
        )


if __name__ == '__main__':
    main()
