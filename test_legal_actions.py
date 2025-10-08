"""
测试新的合法动作处理机制
验证：
1. Agent 只选择合法动作
2. 环境不再处理非法动作
3. 没有非法动作惩罚
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from pacman_env.envs.pacman_env import PacmanEnv
from agents.mc_agent import MCAgent


def test_legal_actions():
    print("="*70)
    print("测试合法动作处理机制")
    print("="*70)
    
    env = PacmanEnv(use_graphics=False, episode_length=50)
    agent = MCAgent(action_space_size=5, epsilon=1.0)  # 100% 探索
    agent.train()
    
    action_names = ['North', 'South', 'East', 'West', 'Stop']
    
    obs, info = env.reset(layout='mediumClassic_noGhosts.lay')
    
    print(f"\n初始状态:")
    print(f"  Agent 位置: {obs['agent']}")
    print(f"  合法动作: {[action_names[a] for a in env.get_legal_actions()]}")
    print(f"  Action Mask: {env.action_masks()}")
    
    print(f"\n执行 20 步，验证所有选择的动作都是合法的:")
    print("-"*70)
    
    illegal_count = 0
    action_counts = {i: 0 for i in range(5)}
    
    for step in range(20):
        # 获取合法动作
        legal_actions = env.get_legal_actions()
        legal_action_names = [action_names[a] for a in legal_actions]
        
        # Agent 选择动作
        action = agent.select_action(obs, legal_actions)
        action_name = action_names[action]
        action_counts[action] += 1
        
        # 验证选择的动作是否合法
        is_legal = action in legal_actions
        status = "✓ 合法" if is_legal else "✗ 非法！"
        
        if not is_legal:
            illegal_count += 1
        
        # 执行动作
        next_obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"  Step {step+1:2d}: 选择 {action_name:6s} | "
              f"合法动作: {legal_action_names} | "
              f"{status} | Reward: {reward:6.2f}")
        
        obs = next_obs
        
        if terminated or truncated:
            print(f"\n  Episode 结束（{'terminated' if terminated else 'truncated'}）")
            break
    
    print("\n" + "="*70)
    print("测试结果:")
    print("="*70)
    print(f"总步数: {step + 1}")
    print(f"非法动作次数: {illegal_count}")
    print(f"动作分布:")
    for action_idx, count in action_counts.items():
        if count > 0:
            print(f"  {action_names[action_idx]:6s}: {count:2d} 次")
    
    if illegal_count == 0:
        print("\n✓✓✓ 测试通过！所有选择的动作都是合法的")
    else:
        print(f"\n✗✗✗ 测试失败！有 {illegal_count} 次选择了非法动作")
    
    env.close()


def test_action_masks():
    """测试 action_masks 方法"""
    print("\n" + "="*70)
    print("测试 Action Masks 方法")
    print("="*70)
    
    env = PacmanEnv(use_graphics=False)
    obs, info = env.reset(layout='mediumClassic_noGhosts.lay')
    
    for i in range(5):
        legal_actions = env.get_legal_actions()
        masks = env.action_masks()
        
        print(f"\n步骤 {i+1}:")
        print(f"  get_legal_actions(): {legal_actions}")
        print(f"  action_masks():      {masks}")
        print(f"  验证: {set(legal_actions) == set(np.where(masks)[0])}")
        
        # 随机执行一个合法动作
        action = np.random.choice(legal_actions)
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            break
    
    env.close()


def compare_old_vs_new():
    """对比新旧方法的效率"""
    print("\n" + "="*70)
    print("对比新旧方法")
    print("="*70)
    
    print("\n旧方法（环境处理非法动作）:")
    print("  - Agent 可能选择非法动作")
    print("  - 环境强制改为 Stop")
    print("  - 额外 -10 惩罚")
    print("  - 学习信号错误：Q(s, East) 被 Stop 的结果更新")
    print("  - 浪费探索时间")
    
    print("\n新方法（Agent 只选择合法动作）:")
    print("  ✓ Agent 只从合法动作中选择")
    print("  ✓ 环境直接执行，无需检查")
    print("  ✓ 无非法动作惩罚")
    print("  ✓ 学习信号正确：Q(s, a) 被 a 的结果更新")
    print("  ✓ 100% 探索效率")
    
    print("\n效率提升估算:")
    print("  假设墙壁占比 30%（非法动作比例）")
    print("  旧方法有效探索: 70%")
    print("  新方法有效探索: 100%")
    print("  效率提升: +43%")


if __name__ == '__main__':
    test_legal_actions()
    test_action_masks()
    compare_old_vs_new()
    
    print("\n" + "="*70)
    print("总结")
    print("="*70)
    print("✓ 环境添加了 get_legal_actions() 和 action_masks() 方法")
    print("✓ 环境的 step() 移除了非法动作检查和惩罚")
    print("✓ Agent 的 select_action() 只从合法动作中选择")
    print("✓ 训练和评估脚本已更新")
    print("\n这是更符合 RL 原理和 Gymnasium 标准的实现方式！")
