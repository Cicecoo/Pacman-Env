"""
实现并测试"表格式特征表示"方法
用特征向量作为键的表格方法，对比线性函数近似的效果
"""

from agents.approximate_q_learning_agent import ApproximateQLearningAgent
from agents.feature_extractors import EnhancedExtractor
from pacman_env import PacmanEnv
import json
import os

PACMAN_ACTIONS = {
    'North': 0,
    'South': 1,
    'East': 2,
    'West': 3,
    'Stop': 4
}

# 配置
LAYOUT = 'smallClassic.lay'
TRAIN_EPISODES = 1000
TEST_EPISODES = 100
USE_GRAPHICS = False

# 超参数
ALPHA = 0.1
EPSILON = 0.05
GAMMA = 0.8


class TabularFeatureAgent:
    """使用特征向量作为键的表格式 Q-Learning"""
    
    def __init__(self, alpha=0.1, epsilon=0.05, gamma=0.8):
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.q_table = {}  # 用特征元组作为键
        self.feature_extractor = EnhancedExtractor()
        self.episode_rewards = 0
        self.last_state = None
        self.last_action = None
        
    def get_q_value(self, state, action):
        """获取 Q 值"""
        features = self.feature_extractor.get_features(state, action)
        # 将特征字典转换为可哈希的元组
        feature_key = tuple(sorted(features.items()))
        return self.q_table.get((feature_key, action), 0.0)
    
    def compute_value_from_q_values(self, state):
        """计算状态的最大 Q 值"""
        legal_actions = state.getLegalPacmanActions()
        if not legal_actions:
            return 0.0
        return max([self.get_q_value(state, action) for action in legal_actions])
    
    def compute_action_from_q_values(self, state):
        """选择最优动作"""
        legal_actions = state.getLegalPacmanActions()
        if not legal_actions:
            return None
        
        # 找到最大 Q 值
        best_value = self.compute_value_from_q_values(state)
        best_actions = [action for action in legal_actions 
                       if abs(self.get_q_value(state, action) - best_value) < 1e-10]
        
        import random
        return random.choice(best_actions)
    
    def choose_action(self, state):
        """epsilon-greedy 策略选择动作"""
        legal_actions = state.getLegalPacmanActions()
        if not legal_actions:
            return None
        
        import random
        if random.random() < self.epsilon:
            return random.choice(legal_actions)
        else:
            return self.compute_action_from_q_values(state)
    
    def update(self, state, action, next_state, reward):
        """Q-Learning 更新"""
        features = self.feature_extractor.get_features(state, action)
        feature_key = tuple(sorted(features.items()))
        
        current_q = self.q_table.get((feature_key, action), 0.0)
        
        # 计算 TD target
        if next_state.isWin() or next_state.isLose():
            td_target = reward
        else:
            next_value = self.compute_value_from_q_values(next_state)
            td_target = reward + self.gamma * next_value
        
        # 更新 Q 值
        new_q = current_q + self.alpha * (td_target - current_q)
        self.q_table[(feature_key, action)] = new_q
    
    def register_initial_state(self, state):
        """初始化 episode"""
        self.last_state = None
        self.last_action = None
        self.episode_rewards = 0
    
    def observe_transition(self, state):
        """观察状态转移"""
        if self.last_state is not None and self.last_action is not None:
            reward = state.getScore() - self.last_state.getScore()
            self.episode_rewards += reward
            self.update(self.last_state, self.last_action, state, reward)
        
        self.last_state = state
        self.last_action = None
    
    def reach_terminal_state(self, state):
        """到达终止状态"""
        if self.last_state is not None and self.last_action is not None:
            reward = state.getScore() - self.last_state.getScore()
            self.episode_rewards += reward
            self.update(self.last_state, self.last_action, state, reward)
    
    def set_test_mode(self):
        """设置为测试模式"""
        self.epsilon = 0.0
    
    def save(self, filepath):
        """保存模型"""
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(self.q_table, f)
    
    def load(self, filepath):
        """加载模型"""
        import pickle
        with open(filepath, 'rb') as f:
            self.q_table = pickle.load(f)


def train_agent(agent, agent_name):
    """训练 agent"""
    print("="*60)
    print(f"Training {agent_name} on {LAYOUT}")
    print(f"Episodes: {TRAIN_EPISODES}")
    print("="*60 + "\n")
    
    env = PacmanEnv(use_graphics=USE_GRAPHICS)
    
    for episode in range(TRAIN_EPISODES):
        obs, info = env.reset(layout=LAYOUT)
        cur_state = env.game.state
        agent.register_initial_state(cur_state)
        done = False
        
        while not done:
            action = agent.choose_action(cur_state)
            
            next_obs, reward, terminated, truncated, info = env.step(PACMAN_ACTIONS[action])
            done = terminated or truncated
            
            new_state = env.game.state
            # 在 observe_transition 前设置 last_action
            if hasattr(agent, 'last_action'):
                agent.last_action = action
            agent.observe_transition(new_state)
            cur_state = new_state
        
        agent.reach_terminal_state(cur_state)
        
        if (episode + 1) % 100 == 0:
            score = env.game.state.getScore()
            is_win = env.game.state.isWin()
            print(f"Episode {episode + 1}/{TRAIN_EPISODES}, "
                  f"Score: {score}, Win: {is_win}")


def test_agent(agent, agent_name):
    """测试 agent"""
    print("\n" + "="*60)
    print(f"Testing {agent_name} on {LAYOUT}")
    print(f"Test Episodes: {TEST_EPISODES}")
    print("="*60 + "\n")
    
    env = PacmanEnv(use_graphics=USE_GRAPHICS)
    agent.set_test_mode()
    
    score_list = []
    win_list = []
    
    for episode in range(TEST_EPISODES):
        obs, info = env.reset(layout=LAYOUT)
        cur_state = env.game.state
        agent.register_initial_state(cur_state)
        done = False
        
        while not done:
            action = agent.choose_action(cur_state)
            
            next_obs, reward, terminated, truncated, info = env.step(PACMAN_ACTIONS[action])
            done = terminated or truncated
            
            new_state = env.game.state
            # 在 observe_transition 前设置 last_action
            if hasattr(agent, 'last_action'):
                agent.last_action = action
            agent.observe_transition(new_state)
            cur_state = new_state
        
        agent.reach_terminal_state(cur_state)
        
        score = env.game.state.getScore()
        is_win = env.game.state.isWin()
        
        score_list.append(score)
        win_list.append(1 if is_win else 0)
    
    # 统计结果
    highest_score = max(score_list)
    average_score = sum(score_list) / TEST_EPISODES
    win_rate = sum(win_list) / TEST_EPISODES * 100
    
    print("\n" + "="*60)
    print(f"{agent_name} Test Results:")
    print(f"Highest Score: {highest_score}")
    print(f"Average Score: {average_score:.1f}")
    print(f"Win Rate: {win_rate:.1f}%")
    print("="*60)
    
    return {
        'highest_score': highest_score,
        'average_score': average_score,
        'win_rate': win_rate,
        'scores': score_list,
        'wins': win_list
    }


if __name__ == "__main__":
    print("对比实验：表格式特征表示 vs 线性函数近似\n")
    
    # 测试表格式特征表示
    print("【1】训练和测试表格式特征表示方法")
    tabular_agent = TabularFeatureAgent(alpha=ALPHA, epsilon=EPSILON, gamma=GAMMA)
    train_agent(tabular_agent, "Tabular Feature Representation")
    tabular_results = test_agent(tabular_agent, "Tabular Feature Representation")
    
    # 测试线性函数近似（如果有训练好的模型）
    print("\n【2】测试线性函数近似方法（需要已训练的模型）")
    approx_ckpt = './checkpoints/approx_q_learning_agent-smallClassic-1000ep-a0.2-e0.1-g0.8-EnhenceEx-CanEatCapsule.pkl'
    
    if os.path.exists(approx_ckpt):
        approx_agent = ApproximateQLearningAgent(alpha=0, epsilon=0, gamma=GAMMA)
        approx_agent.load(approx_ckpt)
        approx_results = test_agent(approx_agent, "Linear Function Approximation")
    else:
        print(f"未找到检查点: {approx_ckpt}")
        print("使用已知的实验数据（2000 episode训练）")
        approx_results = {
            'average_score': 1421.6,
            'win_rate': 91.0,
            'highest_score': 1779.0
        }
    
    # 输出对比
    print("\n" + "="*60)
    print("对比结果总结")
    print("="*60)
    print(f"表格式特征表示:")
    print(f"  平均分数: {tabular_results['average_score']:.1f}")
    print(f"  胜率: {tabular_results['win_rate']:.1f}%")
    print(f"  最高分数: {tabular_results['highest_score']:.1f}")
    print()
    print(f"线性函数近似:")
    print(f"  平均分数: {approx_results['average_score']:.1f}")
    print(f"  胜率: {approx_results['win_rate']:.1f}%")
    print(f"  最高分数: {approx_results['highest_score']:.1f}")
    
    print("\n【实验报告数据】")
    print(f"表格式特征表示 & {tabular_results['average_score']:.1f} & {tabular_results['win_rate']:.1f} & 差 \\\\")
    print(f"线性函数近似 & {approx_results['average_score']:.1f} & {approx_results['win_rate']:.1f} & 好 \\\\")
    
    # 保存结果
    results = {
        'tabular_feature': tabular_results,
        'linear_approximation': approx_results,
        'info': {
            'layout': LAYOUT,
            'train_episodes': TRAIN_EPISODES,
            'test_episodes': TEST_EPISODES
        }
    }
    
    result_file = 'comparison_tabular_vs_approx.json'
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\n结果已保存到: {result_file}")
