import random

from agents.q_learning_agent import QLearningAgent
from .feature_extractors import SimpleExtractor

class ApproximateQLearningAgent(QLearningAgent):
    """
      ValueEstimationAgent assigns values to (state,action)
      Q-Values for an environment. As well as a value to a
      state and a policy given respectively by,

      V(s) = max_{a in actions} Q(s,a)
      policy(s) = arg_max_{a in actions} Q(s,a)
    """

    def __init__(self, alpha=1.0, epsilon=0.05, gamma=0.8, extractor='SimpleExtractor'):
        """
        初始化近似Q学习智能体
        Args:
            alpha: 学习率
            epsilon: 探索率
            gamma: 折扣因子
            extractor: 特征提取器名称
        """
        super().__init__(alpha, epsilon, gamma)
        self.feat_extractor = SimpleExtractor()
        self.weights = {}  # 特征权重，而非Q值表


    def get_q_value(self, state, action):
        """
        计算Q值: Q(state,action) = w · f(state,action)
        使用特征权重和特征向量的内积
        """
        features = self.feat_extractor.get_features(state, action)
        q_value = sum(self.weights.get(f, 0.0) * value for f, value in features.items())
        return q_value

    def get_state_value(self, state):
        """
        计算状态价值: V(s) = max_{a in actions} Q(s,a)
        返回在最优动作下的状态价值
        """
        legal_actions = state.getLegalPacmanActions()
        if not legal_actions:
            return 0.0  # 终止状态没有合法动作

        state_values = [
            self.get_q_value(state, action)
            for action in legal_actions
        ]
        return max(state_values)

    def get_policy(self, state):
        """
        计算最优策略: policy(s) = arg_max_{a in actions} Q(s,a)
        返回Q值最大的动作（如有多个则随机选择）
        """
        legal_actions = state.getLegalPacmanActions()
        if not legal_actions:
            return None  # 终止状态没有合法动作

        action_values = [
            (self.get_q_value(state, action), action)
            for action in legal_actions
        ]
        max_value = max(action_values, key=lambda x: x[0])[0]
        best_actions = [action for value, action in action_values if value == max_value]
        return random.choice(best_actions)

    def get_action(self, state):
        """
        使用epsilon-greedy策略选择动作
        """
        legal_actions = state.getLegalPacmanActions()
        
        if not legal_actions:
            return None

        if random.random() < self.epsilon:
            return random.choice(legal_actions)
        else:
            return self.get_policy(state)


    """
      Reinforcemnt Agent: A ValueEstimationAgent
            which estimates Q-Values (as well as policies) from experience
            rather than a model

        - The environment will call
          observeTransition(state,action,nextState,deltaReward),
          which will call update(state, action, nextState, deltaReward)
    """

    def update(self, state, action, next_state, reward):
        """
        更新特征权重
        TD-error: difference = [R + γ·max_a' Q(s',a')] - Q(s,a)
        权重更新: w_i = w_i + α·difference·f_i(s,a)
        """
        features = self.feat_extractor.get_features(state, action)
        q_value = self.get_q_value(state, action)
        next_value = self.get_state_value(next_state)
        # 修复：使用 self.discount 而不是 self.gamma
        difference = (reward + self.discount * next_value) - q_value

        for f, value in features.items():
            self.weights[f] = self.weights.get(f, 0.0) + self.alpha * difference * value


    def observe_transition(self, state, action, next_state, delta_reward):
        """
        观察状态转移，调用update更新权重
        """
        self.update(state, action, next_state, delta_reward)

        self.last_state = next_state
        self.last_action = action
        self.episode_rewards += delta_reward

    def start_episode(self):
        """
        开始新的episode
        """
        self.last_state = None
        self.last_action = None
        self.episode_rewards = 0.0

    def stop_episode(self):
        """
        结束当前episode
        """
        self.episodes += 1

    def observation_function(self, state):
        """
        观察函数：记录上一个动作后到达的状态
        """
        if self.last_state is not None:
            reward = state.getScore() - self.last_state.getScore()
            self.observe_transition(self.last_state, self.last_action, state, reward)
        return state

    def register_initial_state(self, state):
        """
        注册初始状态，开始新的episode
        """
        self.start_episode()
        if self.episodes == 0:
            if self.mode == 'train':
                print(f"Beginning training with Approximate Q-Learning...")
            else:
                print(f"Beginning testing with Approximate Q-Learning...")

    def reach_terminal_state(self, state):
        """
        到达终止状态，完成最后的更新
        """
        # 如果 last_state 不为 None，说明至少执行了一步，需要更新最后一个 transition
        if self.last_state is not None:
            delta_reward = state.getScore() - self.last_state.getScore()
            self.observe_transition(self.last_state, self.last_action, state, delta_reward)
        self.stop_episode()


    def print_weights(self):
        """
        打印特征权重（用于调试）
        """
        print('========== Feature Weights ==========')   
        for feature in sorted(self.weights.keys()):
            print(f"  {feature}: {self.weights[feature]:.4f}")
        print('=====================================')
    
    def save(self, filename):
        """
        保存特征权重到文件
        """
        import pickle
        with open(filename, 'wb') as f:
            pickle.dump(self.weights, f)
        print(f"Approximate Q-Learning weights saved to {filename}")

    def load(self, filename):
        """
        从文件加载特征权重
        """
        import pickle
        with open(filename, 'rb') as f:
            self.weights = pickle.load(f)
        print(f"Approximate Q-Learning weights loaded from {filename}")

    def set_test_mode(self):
        """
        设置为测试模式：不探索，不学习
        """
        self.epsilon = 0.0
        self.alpha = 0.0
        self.episodes = 0
        self.mode = 'test'

    def set_train_mode(self, epsilon=0.05, alpha=1.0):
        """
        设置为训练模式：启用探索和学习
        """
        self.epsilon = epsilon
        self.alpha = alpha
        self.episodes = 0
        self.mode = 'train'
    
    def get_weights(self):
        """
        获取当前特征权重
        """
        return self.weights.copy()
    
    def get_weights_count(self):
        """
        获取学习到的特征数量
        """
        return len(self.weights)




