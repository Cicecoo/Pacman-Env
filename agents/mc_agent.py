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
        
    # def extract_features(self, obs: Dict) -> Tuple:
    #     """
    #     Extract key features from observation for state representation.
        
    #     This is crucial for generalization - we can't store Q-values for
    #     every possible game state, so we extract important features.
        
    #     Args:
    #         obs: Dictionary observation from environment
            
    #     Returns:
    #         Tuple of features representing the state
    #     """
    #     # Agent position (discretized)
    #     agent_x, agent_y = int(obs['agent'][0]), int(obs['agent'][1])
        
    #     # Agent direction
    #     agent_dir = int(obs['agent_direction'])
        
    #     # Number of food pellets remaining
    #     food_count = int(np.sum(obs['food']))
        
    #     # Closest ghost information
    #     ghosts = obs['ghosts']
    #     agent_pos = obs['agent']
        
    #     if len(ghosts) > 0 and not np.all(ghosts == 0):
    #         # Find non-zero ghost positions
    #         valid_ghosts = ghosts[~np.all(ghosts == 0, axis=1)]
    #         if len(valid_ghosts) > 0:
    #             # Distance to nearest ghost
    #             distances = np.linalg.norm(valid_ghosts - agent_pos, axis=1)
    #             min_dist = int(np.min(distances))
                
    #             # Direction to nearest ghost (simplified to 4 directions)
    #             nearest_ghost_idx = np.argmin(distances)
    #             nearest_ghost = valid_ghosts[nearest_ghost_idx]
    #             dx = nearest_ghost[0] - agent_pos[0]
    #             dy = nearest_ghost[1] - agent_pos[1]
                
    #             if abs(dx) > abs(dy):
    #                 ghost_dir = 2 if dx > 0 else 3  # East or West
    #             else:
    #                 ghost_dir = 0 if dy > 0 else 1  # North or South
    #         else:
    #             min_dist = 99
    #             ghost_dir = -1
    #     else:
    #         min_dist = 99
    #         ghost_dir = -1
        
    #     # Discretize ghost distance into bins
    #     if min_dist <= 2:
    #         dist_bin = 0  # Very close
    #     elif min_dist <= 4:
    #         dist_bin = 1  # Close
    #     elif min_dist <= 7:
    #         dist_bin = 2  # Medium
    #     else:
    #         dist_bin = 3  # Far
        
    #     # Any ghosts scared?
    #     any_scared = int(np.any(obs['ghost_scared']))
        
    #     # Food in adjacent cells (4-directional)
    #     food_grid = obs['food']
    #     food_nearby = 0
    #     # Ensure indices are within bounds
    #     if agent_x > 0 and agent_x < food_grid.shape[0] and agent_y >= 0 and agent_y < food_grid.shape[1]:
    #         if food_grid[agent_x - 1, agent_y]:
    #             food_nearby |= 1  # West
    #     if agent_x >= 0 and agent_x < food_grid.shape[0] - 1 and agent_y >= 0 and agent_y < food_grid.shape[1]:
    #         if food_grid[agent_x + 1, agent_y]:
    #             food_nearby |= 2  # East
    #     if agent_x >= 0 and agent_x < food_grid.shape[0] and agent_y > 0 and agent_y < food_grid.shape[1]:
    #         if food_grid[agent_x, agent_y - 1]:
    #             food_nearby |= 4  # South
    #     if agent_x >= 0 and agent_x < food_grid.shape[0] and agent_y >= 0 and agent_y < food_grid.shape[1] - 1:
    #         if food_grid[agent_x, agent_y + 1]:
    #             food_nearby |= 8  # North
        
    #     # Walls in adjacent cells
    #     walls = obs['walls']
    #     walls_nearby = 0
    #     # Ensure indices are within bounds
    #     if agent_x > 0 and agent_x < walls.shape[0] and agent_y >= 0 and agent_y < walls.shape[1]:
    #         if walls[agent_x - 1, agent_y]:
    #             walls_nearby |= 1
    #     if agent_x >= 0 and agent_x < walls.shape[0] - 1 and agent_y >= 0 and agent_y < walls.shape[1]:
    #         if walls[agent_x + 1, agent_y]:
    #             walls_nearby |= 2
    #     if agent_x >= 0 and agent_x < walls.shape[0] and agent_y > 0 and agent_y < walls.shape[1]:
    #         if walls[agent_x, agent_y - 1]:
    #             walls_nearby |= 4
    #     if agent_x >= 0 and agent_x < walls.shape[0] and agent_y >= 0 and agent_y < walls.shape[1] - 1:
    #         if walls[agent_x, agent_y + 1]:
    #             walls_nearby |= 8
        
    #     # Return feature tuple
    #     features = (
    #         agent_dir,
    #         dist_bin,
    #         ghost_dir,
    #         any_scared,
    #         food_nearby,
    #         walls_nearby,
    #         min(food_count, 10)  # Cap food count for discretization
    #     )
        
    #     return features

    def extract_features(self, obs: Dict) -> Tuple:
        # 与最近的食物的相对位置
        agent_x, agent_y = int(obs['agent'][0]), int(obs['agent'][1])
        food_grid = obs['food']
        food_positions = np.argwhere(food_grid > 0)
        # 游戏没有结束则必有食物
        nearest_food = food_positions[np.argmin(np.linalg.norm(food_positions - obs['agent'], axis=1))]
        food_dx = int(nearest_food[0]) - agent_x
        food_dy = int(nearest_food[1]) - agent_y

        # # 与最近的鬼的相对位置
        # ghosts = obs['ghosts']
        # valid_ghosts = ghosts[~np.all(ghosts == 0, axis=1)]
        # if len(valid_ghosts) > 0:
        #     nearest_ghost = valid_ghosts[np.argmin(np.linalg.norm(valid_ghosts - obs['agent'], axis=1))]
        #     ghost_dx = int(nearest_ghost[0]) - agent_x
        #     ghost_dy = int(nearest_ghost[1]) - agent_y
        #     ghost_dist = int(np.linalg.norm(nearest_ghost - obs['agent']))  
        # else:
        #     ghost_dx, ghost_dy, ghost_dist = 0, 0, 99  # No ghosts
        # # 鬼是否被吓跑
        # any_scared = int(np.any(obs['ghost_scared']))

        # 与最近的未被吓跑的鬼的相对位置
        ghosts = obs['ghosts']
        ghost_scared = obs['ghost_scared']
        valid_ghosts = ghosts[(~np.all(ghosts == 0, axis=1)) & (ghost_scared == 0)]
        if len(valid_ghosts) > 0:
            nearest_ghost = valid_ghosts[np.argmin(np.linalg.norm(valid_ghosts - obs['agent'], axis=1))]
            ghost_dx = int(nearest_ghost[0]) - agent_x
            ghost_dy = int(nearest_ghost[1]) - agent_y
            ghost_dist = int(np.linalg.norm(nearest_ghost - obs['agent']))      
        else:
            ghost_dx, ghost_dy, ghost_dist = 0, 0, 99  # No non-scared ghosts

        # 与最近的被吓跑的鬼的相对位置
        scared_ghosts = ghosts[(~np.all(ghosts == 0, axis=1)) & (ghost_scared > 0)]
        if len(scared_ghosts) > 0:
            nearest_scared_ghost = scared_ghosts[np.argmin(np.linalg.norm(scared_ghosts - obs['agent'], axis=1))]
            scared_ghost_dx = int(nearest_scared_ghost[0]) - agent_x
            scared_ghost_dy = int(nearest_scared_ghost[1]) - agent_y
            scared_ghost_dist = int(np.linalg.norm(nearest_scared_ghost - obs['agent']))
        else:
            scared_ghost_dx, scared_ghost_dy, scared_ghost_dist = 0, 0, 99  # No scared ghosts
        # 存在被吓跑的鬼
        any_scared = int(np.any(obs['ghost_scared']))
        
        # 墙壁信息
        walls = obs['walls']
        walls_nearby = 0
        if agent_x > 0 and agent_x < walls.shape[0] and agent_y >= 0 and agent_y < walls.shape[1]:
            if walls[agent_x - 1, agent_y]:
                walls_nearby |= 1
        if agent_x >= 0 and agent_x < walls.shape[0] - 1 and agent_y >= 0 and agent_y < walls.shape[1]:
            if walls[agent_x + 1, agent_y]:
                walls_nearby |= 2
        if agent_x >= 0 and agent_x < walls.shape[0] and agent_y > 0 and agent_y < walls.shape[1]:
            if walls[agent_x, agent_y - 1]:
                walls_nearby |= 4
        if agent_x >= 0 and agent_x < walls.shape[0] and agent_y >= 0 and agent_y < walls.shape[1] - 1:
            if walls[agent_x, agent_y + 1]:
                walls_nearby |= 8

        features = (
            food_dx,
            food_dy,
            ghost_dx,
            ghost_dy,
            scared_ghost_dx,
            scared_ghost_dy,
            any_scared,
            walls_nearby,
        )

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
