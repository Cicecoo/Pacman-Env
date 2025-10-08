"""
Custom Reward Pacman Environment

支持从配置文件加载 reward shaping 参数的自定义环境。
可以使用 reward_config_interactive.py 生成的配置文件。
"""

import json
import numpy as np
from .pacman_env import PacmanEnv, PACMAN_ACTIONS


class CustomRewardPacmanEnv(PacmanEnv):
    """
    自定义 Reward 的 Pacman 环境
    
    支持从 JSON 配置文件加载 reward shaping 参数，
    或直接传入 reward_config 字典。
    """
    
    def __init__(
        self,
        reward_config=None,
        reward_config_file=None,
        render_mode=None,
        use_dict_obs=True,
        max_ghosts=5,
        use_graphics=True,
        episode_length=100
    ):
        """
        初始化自定义 reward 环境
        
        Args:
            reward_config (dict): Reward 配置字典
            reward_config_file (str): Reward 配置文件路径
            其他参数同 PacmanEnv
        """
        super().__init__(
            render_mode=render_mode,
            use_dict_obs=use_dict_obs,
            max_ghosts=max_ghosts,
            use_graphics=use_graphics,
            episode_length=episode_length
        )
        
        # 加载 reward 配置
        if reward_config is not None:
            self.reward_config = reward_config
        elif reward_config_file is not None:
            self.reward_config = self._load_config(reward_config_file)
        else:
            # 使用默认配置（最小配置）
            self.reward_config = {
                'basic_food': 10.0,
                'basic_capsule': 0.0,
                'basic_ghost_eat': 200.0,
                'basic_death': -500.0,
                'basic_win': 500.0,
                'basic_time_penalty': -1.0,
            }
        
        # 初始化追踪变量
        self.visited_positions = set()
        self.position_history = []
        self.last_food_distance = None
        self.last_capsule_distance = None
        self.steps_since_food = 0
        self.food_eaten_count = 0
        
        print(f"CustomRewardPacmanEnv initialized with config:")
        for key, value in self.reward_config.items():
            if not isinstance(value, dict):
                print(f"  {key}: {value}")
    
    def _load_config(self, filename):
        """从 JSON 文件加载配置"""
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data.get('rewards', {})
    
    def reset(self, seed=None, options=None, layout=None):
        """重置环境，清空追踪变量"""
        obs, info = super().reset(seed=seed, options=options, layout=layout)
        
        # 重置追踪变量
        self.visited_positions = set()
        self.visited_positions.add(self.location)
        
        self.position_history = [self.location]
        self.last_food_distance = self._get_nearest_food_distance()
        self.last_capsule_distance = self._get_nearest_capsule_distance()
        self.steps_since_food = 0
        self.food_eaten_count = 0
        
        return obs, info
    
    def step(self, action):
        """执行动作并应用自定义 reward shaping"""
        
        # 保存当前状态信息
        prev_location = self.location
        prev_food_count = self.game.state.getNumFood()
        prev_score = self.game.state.getScore()
        
        # 调用父类 step
        obs, base_reward, terminated, truncated, info = super().step(action)
        
        # 应用 reward shaping
        shaped_reward = self._apply_reward_shaping(
            action, 
            base_reward,
            prev_location,
            prev_food_count,
            prev_score
        )
        
        # 更新追踪变量
        self._update_tracking(action, base_reward, prev_food_count)
        
        # 在 info 中记录 reward 分解
        info['reward_breakdown'] = {
            'base': base_reward,
            'shaped': shaped_reward - base_reward,
            'total': shaped_reward
        }
        
        return obs, shaped_reward, terminated, truncated, info
    
    def _apply_reward_shaping(
        self, 
        action, 
        base_reward,
        prev_location,
        prev_food_count,
        prev_score
    ):
        """
        应用 reward shaping
        
        注意：UCB Pacman 的 base_reward 已经包含了基础事件奖励
        我们只需要添加额外的 shaping rewards
        """
        reward = base_reward
        config = self.reward_config
        
        # ========== 动作相关奖励 ==========
        
        # Stop 动作惩罚
        if 'stop_penalty' in config and action == 4:  # Stop = 4
            reward += config['stop_penalty']
        
        # 反向移动惩罚
        if 'reverse_direction_penalty' in config:
            if self._is_reverse_direction(action):
                reward += config['reverse_direction_penalty']
        
        # 震荡行为惩罚
        if 'oscillation_penalty' in config:
            if self._detect_oscillation():
                reward += config['oscillation_penalty']
        
        # ========== 位置相关奖励 ==========
        
        # 探索奖励（访问新位置）
        if 'exploration_bonus' in config:
            if self.location not in self.visited_positions:
                reward += config['exploration_bonus']
        
        # 重访惩罚
        if 'revisit_penalty' in config:
            if self.location in self.visited_positions and self.location != prev_location:
                reward += config['revisit_penalty']
        
        # 困角惩罚
        if 'corner_penalty' in config:
            if self._is_in_corner():
                reward += config['corner_penalty']
        
        # ========== 距离相关奖励 ==========
        
        # 接近/远离 Food 奖励
        if 'approach_food_bonus' in config or 'leave_food_penalty' in config:
            food_reward = self._compute_food_distance_reward()
            reward += food_reward
        
        # 接近 Capsule 奖励
        if 'approach_capsule_bonus' in config:
            capsule_reward = self._compute_capsule_distance_reward()
            reward += capsule_reward
        
        # Ghost 距离奖励
        if 'ghost_distance_reward' in config:
            ghost_reward = self._compute_ghost_distance_reward()
            reward += ghost_reward
        
        # ========== 状态相关奖励 ==========
        
        # 剩余 Food 惩罚
        if 'food_remaining_penalty' in config:
            food_count = self.game.state.getNumFood()
            reward += config['food_remaining_penalty'] * food_count
        
        # 进度奖励（里程碑）
        if 'progress_reward' in config:
            progress_reward = self._compute_progress_reward(prev_food_count)
            reward += progress_reward
        
        # 效率奖励
        if 'efficiency_bonus' in config:
            efficiency_reward = self._compute_efficiency_reward()
            reward += efficiency_reward
        
        # ========== 高级奖励：势函数塑造 ==========
        
        if 'potential_based_shaping' in config:
            if config['potential_based_shaping'].get('enabled', False):
                potential_reward = self._compute_potential_shaping(prev_location)
                reward += potential_reward
        
        return reward
    
    # ========== 辅助方法：动作检测 ==========
    
    def _is_reverse_direction(self, action):
        """检测是否是反向移动（180度掉头）"""
        if len(self.position_history) < 2:
            return False
        
        # 反向动作对：North-South (0-1), East-West (2-3)
        reverse_pairs = [(0, 1), (1, 0), (2, 3), (3, 2)]
        last_action = self.orientation_history[-1] if self.orientation_history else None
        
        if last_action is None:
            return False
        
        return (action, last_action) in reverse_pairs
    
    def _detect_oscillation(self, window=4):
        """检测震荡模式：A→B→A→B"""
        if len(self.position_history) < window:
            return False
        
        recent = self.position_history[-window:]
        
        # 检测 A-B-A-B 模式
        if window == 4:
            return (recent[0] == recent[2] and 
                    recent[1] == recent[3] and 
                    recent[0] != recent[1])
        
        return False
    
    def _is_in_corner(self):
        """检测是否在角落或死胡同"""
        legal_actions = self.get_legal_actions()
        # 如果只有 ≤2 个合法动作（包括 Stop），可能在死胡同
        return len(legal_actions) <= 2
    
    # ========== 辅助方法：距离计算 ==========
    
    def _get_nearest_food_distance(self):
        """获取到最近 food 的曼哈顿距离"""
        food_grid = self.game.state.data.food
        food_list = food_grid.asList()
        
        if not food_list:
            return None
        
        distances = [
            abs(self.location[0] - food[0]) + abs(self.location[1] - food[1])
            for food in food_list
        ]
        return min(distances)
    
    def _get_nearest_capsule_distance(self):
        """获取到最近 capsule 的曼哈顿距离"""
        capsules = self.game.state.getCapsules()
        
        if not capsules:
            return None
        
        distances = [
            abs(self.location[0] - cap[0]) + abs(self.location[1] - cap[1])
            for cap in capsules
        ]
        return min(distances)
    
    def _compute_food_distance_reward(self):
        """计算接近/远离 food 的奖励"""
        current_distance = self._get_nearest_food_distance()
        
        if current_distance is None or self.last_food_distance is None:
            return 0.0
        
        delta = self.last_food_distance - current_distance
        self.last_food_distance = current_distance
        
        config = self.reward_config
        
        if delta > 0:  # 接近了
            return config.get('approach_food_bonus', 0.0)
        elif delta < 0:  # 远离了
            return config.get('leave_food_penalty', 0.0)
        
        return 0.0
    
    def _compute_capsule_distance_reward(self):
        """计算接近 capsule 的奖励（仅在有 ghost 且未惊吓时）"""
        # 检查是否有未惊吓的 ghost
        has_dangerous_ghost = any(
            agent.scaredTimer == 0
            for agent in self.game.state.data.agentStates[1:]
        )
        
        if not has_dangerous_ghost:
            return 0.0
        
        current_distance = self._get_nearest_capsule_distance()
        
        if current_distance is None or self.last_capsule_distance is None:
            return 0.0
        
        if current_distance < self.last_capsule_distance:
            self.last_capsule_distance = current_distance
            return self.reward_config.get('approach_capsule_bonus', 0.0)
        
        self.last_capsule_distance = current_distance
        return 0.0
    
    def _compute_ghost_distance_reward(self):
        """计算 ghost 距离相关奖励"""
        config = self.reward_config.get('ghost_distance_reward', {})
        
        if not isinstance(config, dict):
            return 0.0
        
        mode = config.get('mode', 'avoid')
        threshold = config.get('threshold', 3.0)
        scale = config.get('reward_scale', 1.0)
        
        min_ghost_dist = float('inf')
        scared_ghost_dist = float('inf')
        
        for agent in self.game.state.data.agentStates[1:]:
            ghost_pos = agent.getPosition()
            dist = abs(self.location[0] - ghost_pos[0]) + abs(self.location[1] - ghost_pos[1])
            
            if agent.scaredTimer > 0:
                scared_ghost_dist = min(scared_ghost_dist, dist)
            else:
                min_ghost_dist = min(min_ghost_dist, dist)
        
        reward = 0.0
        
        if mode == 'avoid' and min_ghost_dist < threshold:
            # 距离危险 ghost 太近，惩罚
            reward = -scale * (threshold - min_ghost_dist) / threshold
        elif mode == 'chase' and scared_ghost_dist < float('inf'):
            # 追逐惊吓的 ghost，奖励
            reward = scale * (threshold - scared_ghost_dist) / threshold if scared_ghost_dist < threshold else 0
        
        return reward
    
    # ========== 辅助方法：状态奖励 ==========
    
    def _compute_progress_reward(self, prev_food_count):
        """计算进度奖励（里程碑）"""
        current_food_count = self.game.state.getNumFood()
        total_food = prev_food_count + self.food_eaten_count
        
        if total_food == 0:
            return 0.0
        
        prev_progress = (total_food - prev_food_count) / total_food
        current_progress = (total_food - current_food_count) / total_food
        
        milestones = [0.25, 0.5, 0.75]
        reward = 0.0
        
        for milestone in milestones:
            if prev_progress < milestone <= current_progress:
                reward += self.reward_config.get('progress_reward', 0.0)
        
        return reward
    
    def _compute_efficiency_reward(self):
        """计算效率奖励（快速吃到 food）"""
        threshold = 20  # 步数阈值
        
        if self.steps_since_food < threshold:
            ratio = 1.0 - (self.steps_since_food / threshold)
            return self.reward_config.get('efficiency_bonus', 0.0) * ratio
        
        return 0.0
    
    def _compute_potential_shaping(self, prev_location):
        """
        计算基于势函数的 reward shaping
        
        F(s, a, s') = γ × Φ(s') - Φ(s)
        """
        config = self.reward_config.get('potential_based_shaping', {})
        gamma = config.get('gamma', 0.99)
        potential_type = config.get('potential_type', 'nearest_food')
        
        # 计算势函数
        if potential_type == 'nearest_food':
            # Φ(s) = -distance_to_nearest_food
            prev_potential = -self._manhattan_distance_to_nearest_food(prev_location)
            current_potential = -self._get_nearest_food_distance()
            
            if current_potential is None:
                return 0.0
            
            return gamma * current_potential - prev_potential
        
        elif potential_type == 'all_food':
            # Φ(s) = -sum_of_all_food_distances
            food_list = self.game.state.data.food.asList()
            
            prev_sum = sum(
                abs(prev_location[0] - f[0]) + abs(prev_location[1] - f[1])
                for f in food_list
            )
            current_sum = sum(
                abs(self.location[0] - f[0]) + abs(self.location[1] - f[1])
                for f in food_list
            )
            
            prev_potential = -prev_sum
            current_potential = -current_sum
            
            return gamma * current_potential - prev_potential
        
        return 0.0
    
    def _manhattan_distance_to_nearest_food(self, position):
        """计算给定位置到最近 food 的曼哈顿距离"""
        food_list = self.game.state.data.food.asList()
        
        if not food_list:
            return 0
        
        distances = [
            abs(position[0] - food[0]) + abs(position[1] - food[1])
            for food in food_list
        ]
        return min(distances)
    
    # ========== 辅助方法：追踪更新 ==========
    
    def _update_tracking(self, action, base_reward, prev_food_count):
        """更新追踪变量"""
        # 更新访问位置集合
        self.visited_positions.add(self.location)
        
        # 更新位置历史
        self.position_history.append(self.location)
        if len(self.position_history) > 10:  # 只保留最近 10 步
            self.position_history.pop(0)
        
        # 更新 steps_since_food
        current_food_count = self.game.state.getNumFood()
        if current_food_count < prev_food_count:
            # 吃到了 food
            self.steps_since_food = 0
            self.food_eaten_count += 1
        else:
            self.steps_since_food += 1


# ========== 辅助函数：加载配置 ==========

def load_reward_config(filename='reward_config.json'):
    """加载 reward 配置文件"""
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
        return data.get('rewards', {})


# ========== 示例用法 ==========

if __name__ == '__main__':
    # 示例 1: 使用配置文件
    env = CustomRewardPacmanEnv(
        reward_config_file='reward_config.json',
        use_graphics=False,
        episode_length=100
    )
    
    # 示例 2: 直接传入配置
    custom_config = {
        'basic_food': 10.0,
        'basic_win': 500.0,
        'basic_time_penalty': -1.0,
        'stop_penalty': -1.0,
        'approach_food_bonus': 0.3,
    }
    env = CustomRewardPacmanEnv(
        reward_config=custom_config,
        use_graphics=False
    )
    
    # 测试环境
    obs, info = env.reset()
    for _ in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        if 'reward_breakdown' in info:
            print(f"Base: {info['reward_breakdown']['base']:.2f}, "
                  f"Shaped: {info['reward_breakdown']['shaped']:.2f}, "
                  f"Total: {info['reward_breakdown']['total']:.2f}")
        
        if terminated or truncated:
            break
    
    env.close()
