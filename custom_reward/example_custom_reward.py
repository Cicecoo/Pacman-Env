"""
快速使用自定义 Reward 环境的示例

展示如何使用 CustomRewardPacmanEnv 和配置文件
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pacman_env.envs.custom_reward_env import CustomRewardPacmanEnv
from agents.mc_agent import MCAgent
import numpy as np


def example_1_minimal_config():
    """示例 1: 使用最小配置（仅 UCB 原生）"""
    print("\n" + "="*70)
    print("示例 1: 最小配置（仅 UCB Pacman 原生 reward）")
    print("="*70)
    
    config = {
        'basic_food': 10.0,
        'basic_capsule': 0.0,
        'basic_ghost_eat': 200.0,
        'basic_death': -500.0,
        'basic_win': 500.0,
        'basic_time_penalty': -1.0,
    }
    
    env = CustomRewardPacmanEnv(
        reward_config=config,
        use_graphics=False,
        episode_length=100
    )
    
    obs, info = env.reset(layout='smallClassic.lay')
    
    total_reward = 0
    for step in range(100):
        legal_actions = env.get_legal_actions()
        action = np.random.choice(legal_actions)
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        print(f"Step {step+1}: Action={action}, Reward={reward:.2f}, Total={total_reward:.2f}")
        
        if terminated or truncated:
            print(f"\nEpisode ended: {'WIN' if info['is_win'] else 'LOSE'}")
            print(f"Final reward: {total_reward:.2f}")
            break
    
    env.close()


def example_2_anti_stop():
    """示例 2: 反 Stop 配置"""
    print("\n" + "="*70)
    print("示例 2: 反 Stop 配置（解决喜欢停在原地的问题）")
    print("="*70)
    
    config = {
        'basic_food': 10.0,
        'basic_win': 500.0,
        'basic_time_penalty': -1.0,
        'stop_penalty': -2.0,  # Stop 动作额外惩罚 -2
    }
    
    env = CustomRewardPacmanEnv(
        reward_config=config,
        use_graphics=False,
        episode_length=100
    )
    
    obs, info = env.reset(layout='smallClassic.lay')
    
    # 统计 Stop 动作次数
    action_counts = {i: 0 for i in range(5)}
    total_reward = 0
    
    for step in range(100):
        legal_actions = env.get_legal_actions()
        action = np.random.choice(legal_actions)
        action_counts[action] += 1
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if 'reward_breakdown' in info:
            breakdown = info['reward_breakdown']
            if action == 4:  # Stop action
                print(f"Step {step+1}: STOP! Base={breakdown['base']:.2f}, "
                      f"Shaped={breakdown['shaped']:.2f}, Total={breakdown['total']:.2f}")
        
        if terminated or truncated:
            break
    
    print(f"\n动作统计:")
    action_names = ['North', 'South', 'East', 'West', 'Stop']
    for i, name in enumerate(action_names):
        print(f"  {name}: {action_counts[i]} 次 ({action_counts[i]/sum(action_counts.values())*100:.1f}%)")
    
    env.close()


def example_3_guided_learning():
    """示例 3: 引导学习（接近 food 奖励）"""
    print("\n" + "="*70)
    print("示例 3: 引导学习（接近 food 有奖励，远离有惩罚）")
    print("="*70)
    
    config = {
        'basic_food': 10.0,
        'basic_win': 500.0,
        'basic_time_penalty': -1.0,
        'approach_food_bonus': 0.3,
        'leave_food_penalty': -0.3,
    }
    
    env = CustomRewardPacmanEnv(
        reward_config=config,
        use_graphics=False,
        episode_length=100
    )
    
    obs, info = env.reset(layout='smallClassic.lay')
    
    total_base = 0
    total_shaped = 0
    
    for step in range(100):
        legal_actions = env.get_legal_actions()
        action = np.random.choice(legal_actions)
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        if 'reward_breakdown' in info:
            breakdown = info['reward_breakdown']
            total_base += breakdown['base']
            total_shaped += breakdown['shaped']
            
            if breakdown['shaped'] != 0:
                print(f"Step {step+1}: Base={breakdown['base']:.2f}, "
                      f"Shaped={breakdown['shaped']:.2f} ({'接近' if breakdown['shaped'] > 0 else '远离'} food)")
        
        if terminated or truncated:
            break
    
    print(f"\n奖励统计:")
    print(f"  基础奖励总和: {total_base:.2f}")
    print(f"  塑造奖励总和: {total_shaped:.2f}")
    print(f"  最终总奖励: {total_base + total_shaped:.2f}")
    
    env.close()


def example_4_potential_shaping():
    """示例 4: 势函数塑造（理论保证最优策略不变）"""
    print("\n" + "="*70)
    print("示例 4: 势函数塑造（基于到最近 food 的距离）")
    print("="*70)
    
    config = {
        'basic_food': 10.0,
        'basic_win': 500.0,
        'basic_time_penalty': -1.0,
        'potential_based_shaping': {
            'enabled': True,
            'gamma': 0.99,
            'potential_type': 'nearest_food'
        }
    }
    
    env = CustomRewardPacmanEnv(
        reward_config=config,
        use_graphics=False,
        episode_length=100
    )
    
    obs, info = env.reset(layout='smallClassic.lay')
    
    total_base = 0
    total_shaped = 0
    
    for step in range(100):
        legal_actions = env.get_legal_actions()
        action = np.random.choice(legal_actions)
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        if 'reward_breakdown' in info:
            breakdown = info['reward_breakdown']
            total_base += breakdown['base']
            total_shaped += breakdown['shaped']
            
            if step < 10:  # 只打印前 10 步
                print(f"Step {step+1}: Base={breakdown['base']:.2f}, "
                      f"Potential={breakdown['shaped']:.3f}, Total={breakdown['total']:.2f}")
        
        if terminated or truncated:
            break
    
    print(f"\n奖励统计:")
    print(f"  基础奖励总和: {total_base:.2f}")
    print(f"  势函数塑造总和: {total_shaped:.2f}")
    print(f"  最终总奖励: {total_base + total_shaped:.2f}")
    print(f"\n注意: 势函数塑造保证不改变最优策略，只加速学习")
    
    env.close()


def example_5_from_config_file():
    """示例 5: 从配置文件加载"""
    print("\n" + "="*70)
    print("示例 5: 从配置文件加载 reward 配置")
    print("="*70)
    
    # 先创建一个示例配置文件
    import json
    
    config_data = {
        "metadata": {
            "description": "示例配置：平衡方案",
            "version": "1.0"
        },
        "rewards": {
            "basic_food": 10.0,
            "basic_win": 500.0,
            "basic_time_penalty": -1.0,
            "stop_penalty": -0.5,
            "oscillation_penalty": -2.0,
            "exploration_bonus": 0.1,
            "approach_food_bonus": 0.2
        }
    }
    
    with open('example_reward_config.json', 'w', encoding='utf-8') as f:
        json.dump(config_data, f, indent=2, ensure_ascii=False)
    
    print("已创建配置文件: example_reward_config.json")
    print("配置内容:")
    for key, value in config_data['rewards'].items():
        print(f"  {key}: {value}")
    
    # 使用配置文件创建环境
    env = CustomRewardPacmanEnv(
        reward_config_file='example_reward_config.json',
        use_graphics=False,
        episode_length=100
    )
    
    print("\n✅ 环境创建成功！")
    print("现在可以用这个环境训练 agent 了")
    
    env.close()


def example_6_train_with_custom_reward():
    """示例 6: 使用自定义 reward 训练 MC agent"""
    print("\n" + "="*70)
    print("示例 6: 使用自定义 reward 训练 MC agent")
    print("="*70)
    
    # 使用反 Stop 配置
    config = {
        'basic_food': 10.0,
        'basic_win': 500.0,
        'basic_time_penalty': -1.0,
        'stop_penalty': -1.5,
        'approach_food_bonus': 0.2,
    }
    
    env = CustomRewardPacmanEnv(
        reward_config=config,
        use_graphics=False,
        episode_length=100
    )
    
    agent = MCAgent(
        action_space_size=5,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01
    )
    
    print("训练 10 个 episodes...")
    
    episode_rewards = []
    
    for episode in range(10):
        obs, info = env.reset(layout='smallClassic.lay')
        episode_reward = 0
        
        for step in range(100):
            legal_actions = env.get_legal_actions()
            action = agent.select_action(obs, legal_actions)
            agent.store_transition(obs, action, 0)
            
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            if len(agent.episode_buffer) > 0:
                state, act, _ = agent.episode_buffer[-1]
                agent.episode_buffer[-1] = (state, act, reward)
            
            episode_reward += reward
            obs = next_obs
            
            if terminated or truncated:
                break
        
        agent.end_episode()
        episode_rewards.append(episode_reward)
        
        print(f"Episode {episode+1}: Reward={episode_reward:.1f}, Steps={step+1}, ε={agent.epsilon:.3f}")
    
    print(f"\n训练完成！平均 reward: {np.mean(episode_rewards):.2f}")
    
    env.close()


def main():
    """主函数"""
    print("\n" + "="*70)
    print("  CustomRewardPacmanEnv 使用示例")
    print("="*70)
    print("\n可用示例:")
    print("1. 最小配置（仅 UCB 原生）")
    print("2. 反 Stop 配置")
    print("3. 引导学习（接近 food 奖励）")
    print("4. 势函数塑造")
    print("5. 从配置文件加载")
    print("6. 训练 MC agent")
    print("0. 运行所有示例")
    
    choice = input("\n请选择示例 (0-6): ").strip()
    
    if choice == '1':
        example_1_minimal_config()
    elif choice == '2':
        example_2_anti_stop()
    elif choice == '3':
        example_3_guided_learning()
    elif choice == '4':
        example_4_potential_shaping()
    elif choice == '5':
        example_5_from_config_file()
    elif choice == '6':
        example_6_train_with_custom_reward()
    elif choice == '0':
        example_1_minimal_config()
        example_2_anti_stop()
        example_3_guided_learning()
        example_4_potential_shaping()
        example_5_from_config_file()
        example_6_train_with_custom_reward()
    else:
        print("无效选择")


if __name__ == '__main__':
    main()
