import random
import pickle

class QLearningAgent:
    """
      ValueEstimationAgent assigns values to (state,action)
      Q-Values for an environment. As well as a value to a
      state and a policy given respectively by,

      V(s) = max_{a in actions} Q(s,a)
      policy(s) = arg_max_{a in actions} Q(s,a)
    """

    def __init__(self, alpha=0.2, epsilon=0.05, gamma=0.8):
        """
        hyperparameters:
        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        """
        self.alpha = alpha
        self.epsilon = epsilon
        self.discount = gamma
        self.mode = 'train'  # 'train' or 'test'
        
        """
        memory:
        Q_values: dictionary of Q-values
        """
        self.Q_values = {}

        # self.accumulated_train_rewards = 0.0
        # self.accumulated_test_rewards = 0.0
        self.episodes = 0


    def get_q_value(self, state, action):
        """
        return Q(state,action)
        """
        return self.Q_values.get((state, action), 0.0)

    def get_state_value(self, state):
        """
        What is the value of this state under the best action?
        V(s) = max_{a in actions} Q(s,a)
        """
        legal_actions = state.getLegalPacmanActions()
        if not legal_actions:
            return 0.0  # terminal state 时没有合法动作

        state_values = [
            self.get_q_value(state, action)
            for action in legal_actions
        ]
        return max(state_values)

    def get_policy(self, state):
        """
        e.g., get action values?
        What is the best action to take in the state. 
        policy(s) = arg_max_{a in actions} Q(s,a)
        """
        legal_actions = state.getLegalPacmanActions()

        action_values = [
            (self.get_q_value(state, action), action)
            for action in legal_actions
        ]
        max_value = max(action_values)[0]
        best_actions = [
            action
            for value, action in action_values
            if value == max_value
        ]
        
        return random.choice(best_actions) # 随机选择一个最优动作


    def choose_action(self, state):
        legal_actions = state.getLegalPacmanActions()

        # epsilon-greedy
        if random.random() < self.epsilon:
            action = random.choice(legal_actions)
        else:
            action = self.get_policy(state)
        
        # 关键：保存当前状态和动作，供observation_function使用
        self.last_state = state 
        self.last_action = action
        return action


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
        Q(s,a) = (1-alpha)*Q(s,a) + alpha*(reward + discount * V(s'))
        """
        td_target = reward + self.discount * self.get_state_value(next_state)
        td_delta = td_target - self.get_q_value(state, action)
        new_q_value = self.get_q_value(state, action) + self.alpha * td_delta

        self.Q_values[(state, action)] = new_q_value


    def observe_transition(self, state):
        if self.last_state is not None:
            delta_reward = state.getScore() - self.last_state.getScore()

            self.update(self.last_state, self.last_action, state, delta_reward)

            # 应该在choose_action里更新
            # self.last_state = state
            # self.last_action = 还没有选出新动作
            self.episode_rewards += delta_reward
        return state

    def start_episode(self):
        """
          Called by environment when new episode is starting
        """
        self.last_state = None
        self.last_action = None
        self.episode_rewards = 0.0

    def stop_episode(self):
        self.episodes += 1


    def register_initial_state(self, state):
        self.start_episode()
        if self.episodes == 0:
            if self.mode == 'train':
                print(f"Beginning training...")
            else:
                print(f"Beginning testing...")

    def reach_terminal_state(self, state):
        """
          Called by Pacman game at the terminal state
        """
        # 如果 last_state 不为 None，说明至少执行了一步，需要更新最后一个 transition
        if self.last_state is not None:
            delta_reward = state.getScore() - self.last_state.getScore()
            self.observe_transition(state)
        self.stop_episode()


    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.Q_values, f)

    def load(self, filename):
        with open(filename, 'rb') as f:
            self.Q_values = pickle.load(f)

    # 推理模式
    def set_test_mode(self):
        self.epsilon = 0.0
        self.alpha = 0.0
        self.episodes = 0
        self.mode = 'test'

    # 训练模式
    def set_train_mode(self, epsilon=0.05, alpha=1.0):
        self.epsilon = epsilon
        self.alpha = alpha
        self.episodes = 0
        self.mode = 'train'




