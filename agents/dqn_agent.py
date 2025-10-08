import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import pickle
from pathlib import Path


class DQN(nn.Module):
    """Deep Q-Network"""
    
    def __init__(self, input_dim, output_dim, hidden_dims=[256, 256]):
        super(DQN, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))  # 防止过拟合
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class ReplayBuffer:
    """经验回放缓冲区"""
    
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """DQN Agent for Pacman"""
    
    def __init__(self,
                 action_space_size=5,
                 encoder=None,
                 hidden_dims=[256, 256],
                 learning_rate=0.001,
                 gamma=0.99,
                 epsilon_start=1.0,
                 epsilon_end=0.01,
                 epsilon_decay=0.995,
                 buffer_capacity=10000,
                 batch_size=64,
                 target_update_freq=10,
                 device=None):
        # 状态编码器
        # self.encoder = FixedSizeStateEncoder(
        #     max_map_size=max_map_size,
        #     max_ghosts=max_ghosts,
        #     max_capsules=max_capsules,
        #     top_k_foods=top_k_foods,
        #     use_grid_encoding=use_grid_encoding
        # )

        self.encoder = encoder 

        
        self.action_space_size = action_space_size
        self.state_dim = self.encoder.feature_dim
        
        # 设备
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        print(f"DQN Agent initialized:")
        print(f"  State dimension: {self.state_dim}")
        print(f"  Action space: {action_space_size}")
        print(f"  Device: {self.device}")
        
        # Q网络和目标网络
        self.q_network = DQN(self.state_dim, action_space_size, hidden_dims).to(self.device)
        self.target_network = DQN(self.state_dim, action_space_size, hidden_dims).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()  # 目标网络不需要梯度
        
        # 优化器
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # 超参数
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # 经验回放
        self.replay_buffer = ReplayBuffer(buffer_capacity)
        
        # 统计信息
        self.episode_count = 0
        self.step_count = 0
        self.total_steps = 0  # 添加总步数统计
        self.loss_history = []
        
        # 训练模式标志
        self.training = True
    
    def select_action(self, obs_dict, epsilon=None, legal_actions=None):
        """
        选择动作（epsilon-greedy策略）
        只从合法动作中选择，避免非法动作。
        
        Args:
            obs_dict: 环境观测字典
            epsilon: 探索率（如果为None则使用self.epsilon）
            legal_actions: 合法动作列表。如果为None，假设所有动作都合法
        
        Returns:
            action: 选择的动作索引（保证是合法动作）
        """
        if legal_actions is None:
            legal_actions = list(range(self.action_space_size))
        
        # 确保至少有一个合法动作
        if len(legal_actions) == 0:
            raise ValueError("No legal actions available!")
        
        if epsilon is None:
            epsilon = self.epsilon
        
        # Epsilon-greedy (只从合法动作中选择)
        if self.training and random.random() < epsilon:
            # 探索：随机选择合法动作
            return random.choice(legal_actions)
        else:
            # 利用：从合法动作中选择Q值最大的
            state = self.encoder.encode(obs_dict)
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            # 获取Q值
            with torch.no_grad():
                q_values = self.q_network(state_tensor).cpu().numpy()[0]
            
            # 只考虑合法动作的Q值
            legal_q_values = [(q_values[a], a) for a in legal_actions]
            best_action = max(legal_q_values, key=lambda x: x[0])[1]
            
            return best_action
    
    def store_transition(self, obs, action, reward, next_obs, done):
        """存储经验到回放缓冲区"""
        state = self.encoder.encode(obs)
        next_state = self.encoder.encode(next_obs)
        self.replay_buffer.add(state, action, reward, next_state, done)
    
    def train_step(self):
        """
        执行一次训练步骤（从经验回放中采样并更新网络）
        
        Returns:
            loss: 当前训练损失（如果缓冲区样本不足则返回None）
        """
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # 采样批次
        batch = self.replay_buffer.sample(self.batch_size)
        
        states = torch.FloatTensor([t[0] for t in batch]).to(self.device)
        actions = torch.LongTensor([t[1] for t in batch]).to(self.device)
        rewards = torch.FloatTensor([t[2] for t in batch]).to(self.device)
        next_states = torch.FloatTensor([t[3] for t in batch]).to(self.device)
        dones = torch.FloatTensor([t[4] for t in batch]).to(self.device)
        
        # 计算当前Q值
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # 计算目标Q值（使用目标网络）
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)
        
        # 计算损失
        loss = nn.MSELoss()(current_q_values, target_q_values)
        
        # 反向传播和优化
        self.optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
        self.optimizer.step()
        
        self.step_count += 1
        self.total_steps += 1
        
        # 记录损失到历史
        loss_value = loss.item()
        self.loss_history.append(loss_value)
        
        return loss_value
    
    def train(self):
        """设置为训练模式（启用探索）"""
        self.training = True
    
    def eval(self):
        """设置为评估模式（禁用探索）"""
        self.training = False
    
    def update_target_network(self):
        """更新目标网络"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def decay_epsilon(self):
        """衰减探索率"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def end_episode(self):
        """回合结束时的处理"""
        self.episode_count += 1
        
        # 衰减epsilon
        self.decay_epsilon()
        
        # 定期更新目标网络
        if self.episode_count % self.target_update_freq == 0:
            self.update_target_network()
            print(f"  [Target network updated at episode {self.episode_count}]")
    
    def save(self, filepath):
        """保存模型"""
        checkpoint = {
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'episode_count': self.episode_count,
            'step_count': self.step_count,
            'total_steps': self.total_steps,
            'loss_history': self.loss_history,
            'encoder_config': {

            },
            # 保存超参数
            'hyperparameters': {
                'action_space_size': self.action_space_size,
                'gamma': self.gamma,
                'epsilon_end': self.epsilon_end,
                'epsilon_decay': self.epsilon_decay,
                'batch_size': self.batch_size,
                'target_update_freq': self.target_update_freq,
            }
        }
        
        # 创建目录（如果需要）
        import os
        dirpath = os.path.dirname(filepath)
        if dirpath and dirpath != '.':
            # 检查是否存在同名文件并删除
            if os.path.exists(dirpath) and os.path.isfile(dirpath):
                os.remove(dirpath)
            os.makedirs(dirpath, exist_ok=True)
        
        torch.save(checkpoint, filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.episode_count = checkpoint['episode_count']
        self.step_count = checkpoint['step_count']
        self.total_steps = checkpoint.get('total_steps', self.step_count)
        self.loss_history = checkpoint.get('loss_history', [])
        print(f"Model loaded from {filepath}")
        print(f"  Episodes trained: {self.episode_count}")
        print(f"  Current epsilon: {self.epsilon:.4f}")
        print(f"  Total steps: {self.total_steps}")
    
    def get_q_values(self, obs_dict):
        """获取给定状态下所有动作的Q值（用于调试）"""
        state = self.encoder.encode(obs_dict)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        
        return q_values.cpu().numpy()[0]
    
    def get_stats(self):
        """
        获取训练统计信息
        
        Returns:
            包含统计信息的字典
        """
        # 计算平均Q值
        if len(self.replay_buffer) > 0:
            # 从buffer中采样一些状态来估计平均Q值
            sample_size = min(100, len(self.replay_buffer))
            sample_states = [self.replay_buffer.buffer[i][0] for i in range(sample_size)]
            
            with torch.no_grad():
                states_tensor = torch.FloatTensor(sample_states).to(self.device)
                q_values = self.q_network(states_tensor).cpu().numpy()
                avg_q = np.mean(q_values)
                max_q = np.max(q_values)
                min_q = np.min(q_values)
        else:
            avg_q = max_q = min_q = 0.0
        
        return {
            'episodes_trained': self.episode_count,
            'total_steps': self.total_steps,
            'epsilon': self.epsilon,
            'buffer_size': len(self.replay_buffer),
            'avg_q_value': avg_q,
            'max_q_value': max_q,
            'min_q_value': min_q,
            'recent_loss': np.mean(self.loss_history[-100:]) if self.loss_history else 0.0
        }
    
    def print_stats(self):
        """打印训练统计信息"""
        stats = self.get_stats()
        print("\n" + "="*60)
        print("DQN Agent Statistics")
        print("="*60)
        print(f"Episodes Trained:    {stats['episodes_trained']}")
        print(f"Total Steps:         {stats['total_steps']}")
        print(f"Current Epsilon:     {stats['epsilon']:.4f}")
        print(f"Buffer Size:         {stats['buffer_size']}")
        print(f"Avg Q-value:         {stats['avg_q_value']:.2f}")
        print(f"Max Q-value:         {stats['max_q_value']:.2f}")
        print(f"Min Q-value:         {stats['min_q_value']:.2f}")
        print(f"Recent Loss:         {stats['recent_loss']:.4f}")
        print("="*60 + "\n")


# 测试代码
if __name__ == "__main__":
    # 创建一个简单的测试
    print("Testing DQN Agent...")
    
    # 创建agent
    agent = DQNAgent(
        action_space_size=5,
        max_map_size=20,
        max_ghosts=5,
        max_capsules=4,
        top_k_foods=10,
        use_grid_encoding=False,  # 不使用网格编码（更轻量）
        hidden_dims=[128, 128]
    )
    
    # 创建模拟观测
    dummy_obs = {
        "agent": np.array([5.0, 5.0]),
        "agent_direction": 0,
        "ghosts": np.array([[3.0, 3.0], [7.0, 7.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]),
        "ghost_scared": np.array([0, 1, 0, 0, 0]),
        "food": np.random.randint(0, 2, (10, 10)),
        "capsules": np.array([[2.0, 2.0], [8.0, 8.0], [0.0, 0.0], [0.0, 0.0], 
                             [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], 
                             [0.0, 0.0], [0.0, 0.0]]),
        "walls": np.random.randint(0, 2, (10, 10)),
        "image": np.zeros((84, 84, 3), dtype=np.uint8)
    }
    
    # 测试动作选择
    action = agent.select_action(dummy_obs, epsilon=0.1)
    print(f"\nSelected action: {action}")
    
    # 测试Q值
    q_values = agent.get_q_values(dummy_obs)
    print(f"Q-values: {q_values}")
    
    # 测试存储和训练
    agent.store_transition(dummy_obs, action, 10.0, dummy_obs, False)
    
    # 测试保存和加载
    test_path = "test_dqn_agent.pth"
    agent.save(test_path)
    agent.load(test_path)
    
    print("\n✓ All tests passed!")
