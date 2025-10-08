"""
Monte Carlo Learning Agent for Pacman Environment

This implementation includes:
1. First-Visit Monte Carlo for state-value estimation
2. Monte Carlo Control with epsilon-greedy policy
3. State feature extraction for generalization
4. Training and inference modes
5. Model persistence (save/load)
"""

import numpy as np
import pickle
import os
from collections import defaultdict
from typing import Tuple, Dict, List, Any


class MCAgent:
    """
    Monte Carlo Learning Agent with epsilon-greedy policy improvement.
    
    Uses first-visit MC to estimate Q(s, a) and improves policy greedily.
    Features are extracted from observations to enable generalization.
    """
    
    def __init__(
        self,
        action_space_size: int = 5,
        gamma: float = 0.99,
        epsilon: float = 0.5,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
        learning_rate: float = 0.1
    ):
        """
        Initialize Monte Carlo agent.
        
        Args:
            action_space_size: Number of possible actions (default: 5 for Pacman)
            gamma: Discount factor for future rewards
            epsilon: Initial exploration rate for epsilon-greedy
            epsilon_decay: Decay rate for epsilon after each episode
            epsilon_min: Minimum epsilon value
            learning_rate: Step size for Q-value updates (for incremental MC)
        """
        self.action_space_size = action_space_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.learning_rate = learning_rate
        
        # Q-values: Q(state_features, action) -> estimated return
        self.Q = defaultdict(lambda: np.zeros(action_space_size))
        
        # Returns: (state_features, action) -> list of observed returns
        self.returns = defaultdict(list)
        
        # Visit counts for each state-action pair
        self.visit_counts = defaultdict(lambda: np.zeros(action_space_size))
        
        # Episode buffer
        self.episode_buffer = []
        
        # Training mode flag
        self.training = True
        
        # Statistics
        self.episodes_trained = 0
        self.total_steps = 0

    def _get_direction_8(self, dx: float, dy: float) -> int:
        """
        将 (dx, dy) 向量编码为8个方向之一
        
        方向编码：
        7(NW)  0(N)  1(NE)
           \\   |   /
        6(W) - @ - 2(E)
           /   |   \\
        5(SW)  4(S)  3(SE)
        
        Args:
            dx: x方向的差值 (target_x - agent_x)
            dy: y方向的差值 (target_y - agent_y)
            
        Returns:
            方向编码 0-7
        """
        if dx == 0 and dy == 0:
            return 0  # 默认返回北方
        
        # 使用角度来确定方向
        # atan2 返回 [-π, π]，我们将其转换为 [0, 2π]
        import math
        angle = math.atan2(dy, dx)  # 注意：atan2(y, x)
        
        # 将角度转换为 [0, 2π]
        if angle < 0:
            angle += 2 * math.pi
        
        # 将 [0, 2π] 分成8个区域
        # 0=N (π/2), 1=NE (π/4), 2=E (0), 3=SE (-π/4), 
        # 4=S (-π/2), 5=SW (-3π/4), 6=W (π), 7=NW (3π/4)
        
        # 调整角度：从东方(0)开始，逆时针
        # 我们需要从北方开始，顺时针
        # 北方在 atan2 中是 π/2
        angle = (math.pi / 2 - angle) % (2 * math.pi)
        
        # 分成8个区域，每个区域 π/4 (45度)
        direction = int((angle + math.pi / 8) / (math.pi / 4)) % 8
        
        return direction

    def extract_features(self, obs: Dict) -> Tuple:
        agent_pos = obs['agent']
        agent_x, agent_y = int(agent_pos[0]), int(agent_pos[1])
        
        # ========== 1. 最近 Food 的方向和距离 ==========
        food_grid = obs['food']
        food_positions = np.argwhere(food_grid > 0)
        nearest_food = food_positions[np.argmin(np.linalg.norm(food_positions - agent_pos, axis=1))]
        
        food_dx = nearest_food[0] - agent_x
        food_dy = nearest_food[1] - agent_y
        food_distance = abs(food_dx) + abs(food_dy)  # 曼哈顿距离
        
        # 方向：8个方向编码
        # 0=N, 1=NE, 2=E, 3=SE, 4=S, 5=SW, 6=W, 7=NW
        food_dir = self._get_direction_8(food_dx, food_dy)
        
        # 距离等级
        if food_distance <= 2:
            food_dist_level = 0
        elif food_distance <= 5:
            food_dist_level = 1
        elif food_distance <= 10:
            food_dist_level = 2
        else:
            food_dist_level = 3
        
        # ========== 2. 临近 Food 位置（立即可吃）==========
        food_nearby = 0
        # 注意：numpy 数组索引是 [row, col] = [y, x]
        # food_grid.shape = (height, width)
        # agent_y 对应 row (第1维)，agent_x 对应 col (第2维)
        
        # North (y+1)
        if agent_y < food_grid.shape[0] - 1 and food_grid[agent_y + 1, agent_x]:
            food_nearby |= 1
        # South (y-1)
        if agent_y > 0 and food_grid[agent_y - 1, agent_x]:
            food_nearby |= 2
        # East (x+1)
        if agent_x < food_grid.shape[1] - 1 and food_grid[agent_y, agent_x + 1]:
            food_nearby |= 4
        # West (x-1)
        if agent_x > 0 and food_grid[agent_y, agent_x - 1]:
            food_nearby |= 8
        
        # ========== 3. Ghost 方向和危险等级 ==========
        ghosts = obs['ghosts']
        ghost_scared = obs['ghost_scared']
        valid_ghosts = ghosts[(~np.all(ghosts == 0, axis=1)) & (ghost_scared == 0)]
        
        if len(valid_ghosts) > 0:
            # 找到最近的危险 ghost
            ghost_distances = np.linalg.norm(valid_ghosts - agent_pos, axis=1)
            nearest_ghost_idx = np.argmin(ghost_distances)
            min_ghost_dist = ghost_distances[nearest_ghost_idx]
            nearest_ghost = valid_ghosts[nearest_ghost_idx]
            
            # Ghost 方向（8方向）
            ghost_dx = nearest_ghost[0] - agent_x
            ghost_dy = nearest_ghost[1] - agent_y
            ghost_dir = self._get_direction_8(ghost_dx, ghost_dy)
            
            # 危险等级
            if min_ghost_dist <= 3:
                danger_level = 3  # 危险：ghost 很近
            elif min_ghost_dist <= 6:
                danger_level = 2  # 警戒：ghost 接近
            elif min_ghost_dist <= 10:
                danger_level = 1  # 注意：ghost 在中等距离
            else:
                danger_level = 0  # 安全：ghost 很远
        else:
            ghost_dir = 8  # 特殊值：无威胁 ghost
            danger_level = 0  # 安全：无 ghost 或都在惊吓状态
        
        # ========== 4. 墙壁信息（隐含绕路信息）==========
        walls = obs['walls']
        walls_nearby = 0
        # 注意：numpy 数组索引是 [row, col] = [y, x]
        # walls.shape = (height, width)
        
        # West (x-1)
        if agent_x > 0 and walls[agent_y, agent_x - 1]:
            walls_nearby |= 1
        # East (x+1)
        if agent_x < walls.shape[1] - 1 and walls[agent_y, agent_x + 1]:
            walls_nearby |= 2
        # South (y-1)
        if agent_y > 0 and walls[agent_y - 1, agent_x]:
            walls_nearby |= 4
        # North (y+1)
        if agent_y < walls.shape[0] - 1 and walls[agent_y + 1, agent_x]:
            walls_nearby |= 8
        
        # ========== 5. 组合特征：优先级判断 ==========
        # 当危险等级高时，ghost方向更重要；否则food方向更重要
        # 这样可以帮助智能体在不同情况下做出正确决策
        
        if danger_level >= 2:
            # 危险时：优先考虑ghost方向，但仍保留food信息
            priority = 1  # 逃避模式
        else:
            # 安全时：优先考虑food方向
            priority = 0  # 觅食模式
        
        features = (
            food_dir,         # 0-7 (8种) - Food 的8个方向
            ghost_dir,        # 0-8 (9种) - Ghost 的8个方向 + 无威胁
            food_dist_level,  # 0-3 (4种) - 距离远近
            danger_level,     # 0-3 (4种) - 危险程度
            priority,         # 0-1 (2种) - 行为优先级：0=觅食，1=逃避
            # food_nearby,      # 0-15 (16种) - 临近格子 food 信息
            # walls_nearby      # 0-15 (16种) - 临近格子墙壁信息
        )
        # 状态空间 = 8 × 9 × 4 × 4 × 2 = 2,304
        
        return features

    
    def select_action(self, obs: Dict, legal_actions: List[int] = None) -> int:
        """
        Select action using epsilon-greedy policy.
        只从合法动作中选择，避免非法动作。
        
        Args:
            obs: Observation from environment
            legal_actions: List of legal action indices. If None, assumes all actions are legal.
            
        Returns:
            Selected action index (guaranteed to be legal if legal_actions is provided)
        """
        if legal_actions is None:
            legal_actions = list(range(self.action_space_size))
        
        # 确保至少有一个合法动作
        if len(legal_actions) == 0:
            raise ValueError("No legal actions available!")
        
        state_features = self.extract_features(obs)
        
        # Epsilon-greedy action selection (only from legal actions)
        if self.training and np.random.random() < self.epsilon:
            # Explore: random legal action
            action = np.random.choice(legal_actions)
        else:
            # Exploit: greedy action from legal actions
            q_values = self.Q[state_features]
            
            # 只考虑合法动作的 Q 值
            legal_q_values = [q_values[a] for a in legal_actions]
            best_action_idx = np.argmax(legal_q_values)
            action = legal_actions[best_action_idx]
        
        return action
    
    def store_transition(self, obs: Dict, action: int, reward: float):
        """
        Store a transition in the episode buffer.
        
        Args:
            obs: Observation
            action: Action taken
            reward: Reward received
        """
        state_features = self.extract_features(obs)
        self.episode_buffer.append((state_features, action, reward))
    
    def end_episode(self):
        """
        Process the completed episode and update Q-values using MC learning.
        
        Uses first-visit Monte Carlo:
        1. Calculate returns (G) for each step
        2. Update Q-values for first occurrences of (s, a) pairs
        """
        if len(self.episode_buffer) == 0:
            return
        
        # Calculate returns (G) for each time step
        G = 0
        returns_list = []
        
        for t in range(len(self.episode_buffer) - 1, -1, -1):
            state, action, reward = self.episode_buffer[t]
            G = self.gamma * G + reward
            returns_list.append((state, action, G))
        
        # Reverse to get chronological order
        returns_list.reverse()
        
        # First-visit MC: update Q-values
        # visited = set()
        
        # for state, action, G in returns_list:
        #     sa_pair = (state, action)
            
        #     if sa_pair not in visited:
        #         visited.add(sa_pair)
                
        #         # Store return
        #         self.returns[sa_pair].append(G)
                
        #         # Update Q-value (average of all returns)
        #         # Incremental update for efficiency
        #         self.visit_counts[state][action] += 1
        #         n = self.visit_counts[state][action]
                
        #         # Incremental mean: Q_new = Q_old + (G - Q_old) / n
        #         # Or with learning rate: Q_new = Q_old + alpha * (G - Q_old)
        #         if self.learning_rate is not None:
        #             alpha = self.learning_rate
        #         else:
        #             alpha = 1.0 / n
                
        #         self.Q[state][action] += alpha * (G - self.Q[state][action])

        # 改为 every-visit MC
        for state, action, G in returns_list:
            sa_pair = (state, action)
                
            # Store return
            self.returns[sa_pair].append(G)
                
            # Update Q-value (average of all returns)
            # Incremental update for efficiency
            self.visit_counts[state][action] += 1
            n = self.visit_counts[state][action]
                
            # Incremental mean: Q_new = Q_old + (G - Q_old) / n
            # Or with learning rate: Q_new = Q_old + alpha * (G - Q_old)
            if self.learning_rate is not None:
                alpha = self.learning_rate
            else:
                alpha = 1.0 / n
                
            self.Q[state][action] += alpha * (G - self.Q[state][action])
        
        # Clear episode buffer
        self.episode_buffer = []
        
        # Decay epsilon
        if self.training:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            self.episodes_trained += 1
    
    def train(self):
        """Set agent to training mode (exploration enabled)."""
        self.training = True
    
    def eval(self):
        """Set agent to evaluation mode (no exploration)."""
        self.training = False
    
    def save(self, filepath: str):
        """
        Save agent state to file.
        
        Args:
            filepath: Path to save file
        """
        state = {
            'Q': dict(self.Q),
            'returns': dict(self.returns),
            'visit_counts': dict(self.visit_counts),
            'epsilon': self.epsilon,
            'episodes_trained': self.episodes_trained,
            'total_steps': self.total_steps,
            'action_space_size': self.action_space_size,
            'gamma': self.gamma,
            'epsilon_decay': self.epsilon_decay,
            'epsilon_min': self.epsilon_min,
            'learning_rate': self.learning_rate
        }
        
        # Create directory if needed
        dirpath = os.path.dirname(filepath)
        if dirpath and dirpath != '.':
            # Check if a file with the directory name exists and remove it
            if os.path.exists(dirpath) and os.path.isfile(dirpath):
                os.remove(dirpath)
            os.makedirs(dirpath, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        
        print(f"Agent saved to {filepath}")
    
    def load(self, filepath: str):
        """
        Load agent state from file.
        
        Args:
            filepath: Path to load file
        """
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        self.Q = defaultdict(lambda: np.zeros(self.action_space_size), state['Q'])
        self.returns = defaultdict(list, state['returns'])
        self.visit_counts = defaultdict(
            lambda: np.zeros(self.action_space_size), 
            state['visit_counts']
        )
        self.epsilon = state['epsilon']
        self.episodes_trained = state['episodes_trained']
        self.total_steps = state['total_steps']
        self.action_space_size = state['action_space_size']
        self.gamma = state['gamma']
        self.epsilon_decay = state['epsilon_decay']
        self.epsilon_min = state['epsilon_min']
        self.learning_rate = state['learning_rate']
        
        print(f"Agent loaded from {filepath}")
        print(f"  Episodes trained: {self.episodes_trained}")
        print(f"  Current epsilon: {self.epsilon:.4f}")
        print(f"  Q-table size: {len(self.Q)} states")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get training statistics.
        
        Returns:
            Dictionary of statistics
        """
        return {
            'episodes_trained': self.episodes_trained,
            'total_steps': self.total_steps,
            'epsilon': self.epsilon,
            'q_table_size': len(self.Q),
            'avg_q_value': np.mean([np.mean(q) for q in self.Q.values()]) if len(self.Q) > 0 else 0.0,
            'max_q_value': np.max([np.max(q) for q in self.Q.values()]) if len(self.Q) > 0 else 0.0,
            'min_q_value': np.min([np.min(q) for q in self.Q.values()]) if len(self.Q) > 0 else 0.0,
        }
    
    def print_stats(self):
        """Print training statistics."""
        stats = self.get_stats()
        print("\n" + "="*60)
        print("MC Agent Statistics")
        print("="*60)
        print(f"Episodes Trained:    {stats['episodes_trained']}")
        print(f"Total Steps:         {stats['total_steps']}")
        print(f"Current Epsilon:     {stats['epsilon']:.4f}")
        print(f"Q-table Size:        {stats['q_table_size']} unique states")
        print(f"Avg Q-value:         {stats['avg_q_value']:.2f}")
        print(f"Max Q-value:         {stats['max_q_value']:.2f}")
        print(f"Min Q-value:         {stats['min_q_value']:.2f}")
        print("="*60 + "\n")
