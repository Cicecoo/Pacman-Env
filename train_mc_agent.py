"""
Training script for MCLearningAgent and ApproximateMCLearningAgent

Usage:
    # Train tabular MC agent
    python train_mc_learning.py --agent tabular --episodes 1000
    
    # Train approximate MC agent with features
    python train_mc_learning.py --agent approximate --episodes 1000
"""

import argparse
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.mc_learning_agent import MCLearningAgent
from agents.feature_extractors import SimpleExtractor, EnhancedSimpleExtractor
from pacman_env import PacmanEnv

PACMAN_ACTIONS = {
    'North': 0,
    'South': 1,
    'East': 2,
    'West': 3,
    'Stop': 4
}


def train_agent(agent, env, layout, num_episodes=1000, verbose=True):
    """
    Train an MC agent in the Pacman environment.
    
    Args:
        agent: MCLearningAgent or ApproximateMCLearningAgent instance
        env: PacmanEnv instance
        num_episodes: Number of training episodes
        verbose: Whether to print training progress
    """
    episode_rewards = []
    episode_lengths = []
    
    print(f"\n{'='*60}")
    print(f"Training {agent.__class__.__name__}")
    print(f"{'='*60}")
    print(f"Episodes: {num_episodes}")
    print(f"Alpha (learning rate): {agent.alpha}")
    print(f"Epsilon (exploration): {agent.epsilon}")
    print(f"Gamma (discount): {agent.discount}")
    print(f"{'='*60}\n")
    
    for episode in range(num_episodes):
        obs, info = env.reset(layout=layout)
        cur_state = env.game.state
        agent.register_initial_state(cur_state)
        
        done = False
        steps = 0
        
        while not done:
            # Choose action
            action = agent.choose_action(cur_state)
            
            # Take action
            next_obs, reward, terminated, truncated, info = env.step(PACMAN_ACTIONS[action])
            done = terminated or truncated
            
            # Get new state from game
            new_state = env.game.state
            
            # Observe transition (stores in episode buffer)
            agent.observe_transition(new_state)
            
            cur_state = new_state
            steps += 1
        
        # Episode finished - trigger MC update
        agent.reach_terminal_state(cur_state)
        
        # Track statistics
        episode_rewards.append(agent.episode_rewards)
        episode_lengths.append(steps)
        
        # Print progress
        if verbose and (episode + 1) % 100 == 0:
            avg_reward = sum(episode_rewards[-100:]) / min(100, len(episode_rewards))
            avg_length = sum(episode_lengths[-100:]) / min(100, len(episode_lengths))
            print(f"Episode {episode + 1}/{num_episodes} | "
                  f"Avg Reward: {avg_reward:.2f} | "
                  f"Avg Length: {avg_length:.1f} | "
                  f"Q-table size: {len(agent.Q_values)}")
    
    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"{'='*60}")
    print(f"Final average reward (last 100 episodes): {sum(episode_rewards[-100:]) / 100:.2f}")
    print(f"Final average length (last 100 episodes): {sum(episode_lengths[-100:]) / 100:.1f}")
    print(f"Final Q-table size: {len(agent.Q_values)} state-action pairs")
    if agent.feat_extractor is not None:
        print(f"Using feature extractor: {agent.feat_extractor.__class__.__name__}")
    print(f"{'='*60}\n")
    
    return episode_rewards, episode_lengths


def test_agent(agent, env, layout, num_episodes=10, render=False):
    """
    Test a trained MC agent.
    
    Args:
        agent: Trained MCLearningAgent or ApproximateMCLearningAgent instance
        env: PacmanEnv instance
        num_episodes: Number of test episodes
        render: Whether to render the environment
    """
    agent.set_test_mode()
    
    print(f"\n{'='*60}")
    print(f"Testing {agent.__class__.__name__}")
    print(f"{'='*60}\n")
    
    test_rewards = []
    test_lengths = []
    
    for episode in range(num_episodes):
        obs, info = env.reset(layout=layout)
        cur_state = env.game.state
        agent.register_initial_state(cur_state)
        
        done = False
        steps = 0
        
        if render:
            env.render()
        
        while not done:
            action = agent.choose_action(cur_state)
            next_obs, reward, terminated, truncated, info = env.step(PACMAN_ACTIONS[action])
            done = terminated or truncated
            
            new_state = env.game.state
            agent.observe_transition(new_state)
            
            cur_state = new_state
            steps += 1
            
            if render:
                env.render()
        
        agent.reach_terminal_state(cur_state)
        
        test_rewards.append(agent.episode_rewards)
        test_lengths.append(steps)
        
        print(f"Test Episode {episode + 1}: Reward = {agent.episode_rewards:.2f}, Steps = {steps}")
    
    print(f"\n{'='*60}")
    print(f"Test Results")
    print(f"{'='*60}")
    print(f"Average reward: {sum(test_rewards) / len(test_rewards):.2f}")
    print(f"Average length: {sum(test_lengths) / len(test_lengths):.1f}")
    print(f"{'='*60}\n")
    
    return test_rewards, test_lengths


def main():
    parser = argparse.ArgumentParser(description='Train MC Learning Agent for Pacman')
    parser.add_argument('--agent', type=str, default='tabular', 
                       choices=['tabular', 'approximate'],
                       help='Type of MC agent to train')
    parser.add_argument('--layout', type=str, default='smallClassic',
                       help='Pacman layout name')
    parser.add_argument('--episodes', type=int, default=2000,
                       help='Number of training episodes')
    parser.add_argument('--test-episodes', type=int, default=10,
                       help='Number of test episodes')
    parser.add_argument('--alpha', type=float, default=0.2,
                       help='Learning rate')
    parser.add_argument('--epsilon', type=float, default=0.1,
                       help='Exploration rate')
    parser.add_argument('--gamma', type=float, default=0.8,
                       help='Discount factor')
    parser.add_argument('--extractor', type=str, default='simple',
                       choices=['simple', 'enhanced'],
                       help='Feature extractor type (for approximate agent)')
    parser.add_argument('--save', type=str, default=None,
                       help='Path to save trained agent')
    parser.add_argument('--load', type=str, default=None,
                       help='Path to load trained agent')
    parser.add_argument('--test-only', action='store_true',
                       help='Only test the agent (requires --load)')
    parser.add_argument('--render', action='store_true',
                       help='Render the environment during testing')
    
    args = parser.parse_args()
    
    # Set default layout
    layout = 'smallGrid'
    layout = 'smallClassic'
    
    # Create environment
    env = PacmanEnv(use_graphics=False)
    
    # # Create agent
    # if args.agent == 'tabular':
    #     # Tabular MC Learning (no feature extractor)
    #     agent = MCLearningAgent(alpha=args.alpha, epsilon=args.epsilon, gamma=args.gamma)
    # else:  # approximate (feature-based)
    #     # Feature-based MC Learning (with feature extractor)
    #     if args.extractor == 'simple':
    #         extractor = SimpleExtractor()
    #     else:
    #         extractor = EnhancedSimpleExtractor()
    #     agent = MCLearningAgent(
    #         alpha=args.alpha, epsilon=args.epsilon, gamma=args.gamma,
    #         feat_extractor=extractor
    #     )

    agent = MCLearningAgent(
            alpha=args.alpha, epsilon=args.epsilon, gamma=args.gamma,
            feat_extractor=EnhancedSimpleExtractor()
        )
    
    # Load existing agent if specified
    if args.load:
        agent.load(args.load)
    
    # Training
    if not args.test_only:
        train_agent(agent, env, layout=layout, num_episodes=args.episodes, verbose=True)
        
        # Save agent if specified
        if args.save:
            agent.save(args.save)
        else:
            # Default save path
            agent.save('checkpoints/mc_agent.pkl')
    
    # Testing
    if args.test_episodes > 0:
        print("\nSwitching to test mode...")
        agent.set_test_mode()
          
        env = PacmanEnv(use_graphics=True)
        
        test_agent(agent, env, layout=layout, num_episodes=args.test_episodes, render=args.render)


if __name__ == '__main__':
    main()
