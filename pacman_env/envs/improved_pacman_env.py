"""
改进的 Pacman 环境 - 解决 Stop 动作偏好问题

主要改进:
1. 调整奖励结构，惩罚 Stop 动作
2. 奖励探索新位置
3. 减少非法动作的惩罚
4. 增加接近食物的奖励（reward shaping）
"""

from pacman_env.envs.pacman_env import PacmanEnv
import numpy as np


class ImprovedPacmanEnv(PacmanEnv):
    """
    改进的 Pacman 环境，解决 agent 偏好 Stop 动作的问题
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # 配置参数
        self.stop_penalty = -2.0          # Stop 动作的额外惩罚
        self.illegal_action_penalty = -2.0  # 非法动作惩罚（从 -10 减少）
        self.exploration_bonus = 0.5      # 访问新位置的奖励
        self.approach_food_bonus = 0.2    # 接近食物的奖励
        self.repeat_position_penalty = -0.5  # 重复访问同一位置的惩罚
        
        # 跟踪访问过的位置
        self.visited_positions = set()
        self.recent_positions = []  # 最近 N 步的位置
        self.position_history_length = 10
        
        # 记录上一步到最近食物的距离
        self.prev_min_food_distance = None
    
    def reset(self, **kwargs):
        """重置环境"""
        obs, info = super().reset(**kwargs)
        
        # 重置跟踪变量
        self.visited_positions = set()
        self.recent_positions = []
        self.prev_min_food_distance = self._get_min_food_distance(obs)
        
        # 记录初始位置
        pos = tuple(obs['agent'])
        self.visited_positions.add(pos)
        self.recent_positions.append(pos)
        
        return obs, info
    
    def step(self, action):
        """执行动作，返回改进的奖励"""
        # 获取原始观察和奖励
        obs, base_reward, terminated, truncated, info = super().step(action)
        
        # 计算改进的奖励
        shaped_reward = base_reward
        reward_components = {'base': base_reward}
        
        # 当前位置
        curr_pos = tuple(obs['agent'])
        
        # 1. Stop 动作惩罚
        if action == 4:  # Stop 动作
            shaped_reward += self.stop_penalty
            reward_components['stop_penalty'] = self.stop_penalty
        
        # 2. 探索奖励 - 访问新位置
        if curr_pos not in self.visited_positions:
            shaped_reward += self.exploration_bonus
            reward_components['exploration'] = self.exploration_bonus
            self.visited_positions.add(curr_pos)
        
        # 3. 重复位置惩罚 - 在最近几步中重复
        if len(self.recent_positions) > 0:
            recent_visit_count = self.recent_positions[-self.position_history_length:].count(curr_pos)
            if recent_visit_count > 1:
                penalty = self.repeat_position_penalty * (recent_visit_count - 1)
                shaped_reward += penalty
                reward_components['repeat_penalty'] = penalty
        
        # 4. 接近食物奖励（reward shaping）
        curr_min_food_distance = self._get_min_food_distance(obs)
        if self.prev_min_food_distance is not None and curr_min_food_distance is not None:
            # 如果距离最近的食物更近了，给予奖励
            if curr_min_food_distance < self.prev_min_food_distance:
                shaped_reward += self.approach_food_bonus
                reward_components['approach_food'] = self.approach_food_bonus
            elif curr_min_food_distance > self.prev_min_food_distance:
                shaped_reward -= self.approach_food_bonus / 2
                reward_components['away_food'] = -self.approach_food_bonus / 2
        
        self.prev_min_food_distance = curr_min_food_distance
        
        # 5. 调整非法动作惩罚（已在父类中处理，这里只是记录）
        # 注意：父类使用 -10，我们可以在这里补偿
        if 'illegal_action' in info and info.get('illegal_action', False):
            # 父类已经 -10，我们补偿 8 使实际惩罚为 -2
            compensation = self.illegal_action_penalty - (-10)
            shaped_reward += compensation
            reward_components['illegal_compensation'] = compensation
        
        # 更新位置历史
        self.recent_positions.append(curr_pos)
        if len(self.recent_positions) > self.position_history_length:
            self.recent_positions.pop(0)
        
        # 添加奖励组成到 info
        info['reward_components'] = reward_components
        info['shaped_reward'] = shaped_reward
        
        return obs, shaped_reward, terminated, truncated, info
    
    def _get_min_food_distance(self, obs):
        """计算到最近食物的曼哈顿距离"""
        food_grid = obs['food']
        agent_pos = obs['agent']
        
        # 找到所有食物位置
        food_positions = np.argwhere(food_grid > 0)
        
        if len(food_positions) == 0:
            return None
        
        # 计算到所有食物的曼哈顿距离
        distances = np.abs(food_positions - agent_pos).sum(axis=1)
        
        return np.min(distances)


def test_improved_env():
    """测试改进的环境"""
    print("="*70)
    print("测试改进的 Pacman 环境")
    print("="*70)
    
    env = ImprovedPacmanEnv(use_graphics=False)
    obs, info = env.reset()
    
    print("\n执行一系列动作并观察奖励变化:")
    print("-"*70)
    
    # 测试序列：几个移动动作 + Stop
    test_actions = [
        (0, "North"),
        (2, "East"),
        (4, "Stop"),  # 应该受到惩罚
        (3, "West"),
        (0, "North"),
        (4, "Stop"),  # 应该受到惩罚
        (4, "Stop"),  # 连续 Stop
    ]
    
    for action_idx, action_name in test_actions:
        obs, reward, terminated, truncated, info = env.step(action_idx)
        
        print(f"\n动作: {action_name:6s} (action {action_idx})")
        print(f"  总奖励: {reward:6.2f}")
        
        if 'reward_components' in info:
            print(f"  奖励组成:")
            for component, value in info['reward_components'].items():
                print(f"    - {component:20s}: {value:6.2f}")
        
        if terminated or truncated:
            print("  Episode 结束")
            break
    
    env.close()
    print("\n" + "="*70)


if __name__ == '__main__':
    test_improved_env()
