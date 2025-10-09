"""
Unified Monte Carlo Learning Agent for Pacman

This module provides a single MCLearningAgent class that supports both:
1. Tabular MC Learning (feat_extractor=None)
2. Feature-based MC Learning (feat_extractor provided)

Key Design:
- Uses feature tuples as lookup keys when feature extractor is provided
- Falls back to raw states when no feature extractor is given
- Maintains the same Q-table structure for both modes
- Compatible with QLearningAgent and ApproximateQLearningAgent interfaces

Usage:
    # Tabular MC Learning
    agent = MCLearningAgent(alpha=0.1, epsilon=0.1, gamma=0.8)
    
    # Feature-based MC Learning
    from agents.feature_extractors import SimpleExtractor
    agent = MCLearningAgent(alpha=0.1, epsilon=0.1, gamma=0.8,
                           feat_extractor=SimpleExtractor())
"""

import random
import pickle
from collections import defaultdict


class MCLearningAgent:
    """
    Monte Carlo Learning Agent for Pacman.
    
    Supports both tabular and feature-based learning through optional feature extraction.
    - Tabular mode (feat_extractor=None): Uses raw game states as keys in Q-table
    - Feature mode (feat_extractor provided): Uses extracted feature tuples as keys
    
    Uses every-visit MC to estimate Q(s, a) from complete episodes.
    Updates Q-values based on actual returns rather than bootstrapped estimates.
    """
    
    def __init__(self, alpha=0.1, epsilon=0.05, gamma=0.8, feat_extractor=None):
        """
        Initialize MC Learning Agent.
        
        Args:
            alpha: Learning rate (step size for incremental averaging)
            epsilon: Exploration rate for epsilon-greedy policy
            gamma: Discount factor for future rewards
            feat_extractor: Optional feature extractor (e.g., SimpleExtractor, EnhancedSimpleExtractor)
                          If None: tabular learning (uses raw states as keys)
                          If provided: feature-based learning (uses feature tuples as keys)
        """
        self.alpha = alpha
        self.epsilon = epsilon
        self.discount = gamma
        self.mode = 'train'  # 'train' or 'test'
        
        # Feature extractor (optional)
        self.feat_extractor = feat_extractor
        self.use_features = feat_extractor is not None
        
        # Q-values: {(state_key, action): q_value}
        # state_key is either:
        #   - raw state object (when feat_extractor is None)
        #   - feature tuple (when feat_extractor is provided)
        self.Q_values = {}
        
        # Episode buffer: stores (state, action, reward) tuples for current episode
        self.episode_buffer = []
        
        # For compatibility with environment
        self.last_state = None
        self.last_action = None
        self.episode_rewards = 0.0
        self.episodes = 0
    
    def _get_state_key(self, state, action):
        """
        Get the key for Q-table lookup.
        
        Args:
            state: Game state
            action: Action
            
        Returns:
            If feat_extractor is None: returns state directly
            If feat_extractor provided: returns tuple of feature values
        """
        if not self.use_features:
            return state
        
        # Extract features and convert to hashable tuple
        features = self.feat_extractor.get_features(state, action)
        
        # Convert feature dict to sorted tuple for hashing
        # Format: ((feature_name1, value1), (feature_name2, value2), ...)
        feature_tuple = tuple(sorted(features.items()))
        return feature_tuple
    
    def get_q_value(self, state, action):
        """
        Return Q(state, action).
        
        Uses either raw state or extracted features as the key, depending on configuration.
        """
        state_key = self._get_state_key(state, action)
        return self.Q_values.get((state_key, action), 0.0)
    
    def get_state_value(self, state):
        """
        Return the value of the best action in state.
        V(s) = max_{a in actions} Q(s, a)
        """
        legal_actions = state.getLegalPacmanActions()
        if not legal_actions:
            return 0.0  # Terminal state
        
        state_values = [
            self.get_q_value(state, action)
            for action in legal_actions
        ]
        return max(state_values)
    
    def get_policy(self, state):
        """
        Return the best action in the given state according to learned Q-values.
        policy(s) = arg_max_{a in actions} Q(s, a)
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
        
        return random.choice(best_actions)
    
    def choose_action(self, state):
        """
        Choose action using epsilon-greedy policy.
        During training: explore with probability epsilon
        During testing: always exploit (epsilon = 0)
        """
        legal_actions = state.getLegalPacmanActions()
        
        # Epsilon-greedy
        if random.random() < self.epsilon:
            action = random.choice(legal_actions)
        else:
            action = self.get_policy(state)
        
        # Save state and action for observe_transition
        self.last_state = state
        self.last_action = action
        return action
    
    def observe_transition(self, state):
        """
        Called by environment after each action.
        Stores (state, action, reward) in episode buffer for MC update at episode end.
        """
        if self.last_state is not None:
            delta_reward = state.getScore() - self.last_state.getScore()
            
            # Store transition in episode buffer
            self.episode_buffer.append((self.last_state, self.last_action, delta_reward))
            
            self.episode_rewards += delta_reward
        
        return state
    
    def update_from_episode(self):
        """
        Update Q-values using Monte Carlo learning after episode completes.
        Uses every-visit MC with incremental updates.
        
        Works for both tabular and feature-based modes by using _get_state_key().
        """
        if len(self.episode_buffer) == 0:
            return
        
        # Calculate returns (G) for each step using backward iteration
        G = 0.0
        returns_list = []
        
        for t in range(len(self.episode_buffer) - 1, -1, -1):
            state, action, reward = self.episode_buffer[t]
            G = self.discount * G + reward
            returns_list.append((state, action, G))
        
        # Reverse to get chronological order (optional, but clearer)
        returns_list.reverse()
        
        # Every-visit MC: update Q-values for all (state, action) pairs
        for state, action, G in returns_list:
            # Get the appropriate key (raw state or features)
            state_key = self._get_state_key(state, action)
            
            # Incremental update: Q(s,a) = Q(s,a) + alpha * (G - Q(s,a))
            old_q = self.get_q_value(state, action)
            new_q = old_q + self.alpha * (G - old_q)
            self.Q_values[(state_key, action)] = new_q
        
        # Clear episode buffer
        self.episode_buffer = []
    
    def start_episode(self):
        """
        Called by environment when new episode is starting.
        """
        self.last_state = None
        self.last_action = None
        self.episode_rewards = 0.0
        self.episode_buffer = []
    
    def stop_episode(self):
        """
        Called by environment when episode ends.
        Triggers MC update from collected episode data.
        """
        self.update_from_episode()
        self.episodes += 1
    
    def register_initial_state(self, state):
        """
        Called by Pacman game at the start of each episode.
        """
        self.start_episode()
        if self.episodes == 0:
            if self.mode == 'train':
                print(f"Beginning training with MC Learning...")
            else:
                print(f"Beginning testing with MC Learning...")
    
    def reach_terminal_state(self, state):
        """
        Called by Pacman game at the terminal state.
        Processes final transition and triggers episode update.
        """
        # Process final transition if exists
        if self.last_state is not None:
            delta_reward = state.getScore() - self.last_state.getScore()
            self.episode_buffer.append((self.last_state, self.last_action, delta_reward))
            self.episode_rewards += delta_reward
        
        # Update Q-values from complete episode
        self.stop_episode()
    
    def save(self, filename):
        """
        Save Q-values to file.
        """
        with open(filename, 'wb') as f:
            pickle.dump(self.Q_values, f)
        
        mode_str = "feature-based" if self.use_features else "tabular"
        print(f"MC agent ({mode_str}) saved to {filename}")
        print(f"  Q-table size: {len(self.Q_values)} entries")
    
    def load(self, filename):
        """
        Load Q-values from file.
        """
        with open(filename, 'rb') as f:
            self.Q_values = pickle.load(f)
        
        mode_str = "feature-based" if self.use_features else "tabular"
        print(f"MC agent ({mode_str}) loaded from {filename}")
        print(f"  Q-table size: {len(self.Q_values)} entries")
    
    def set_test_mode(self):
        """
        Switch to inference mode: no exploration, no learning.
        """
        self.epsilon = 0.0
        self.alpha = 0.0
        self.episodes = 0
        self.mode = 'test'
    
    def set_train_mode(self, epsilon=0.05, alpha=0.1):
        """
        Switch to training mode: enable exploration and learning.
        """
        self.epsilon = epsilon
        self.alpha = alpha
        self.episodes = 0
        self.mode = 'train'


# Alias for backward compatibility
ApproximateMCLearningAgent = MCLearningAgent