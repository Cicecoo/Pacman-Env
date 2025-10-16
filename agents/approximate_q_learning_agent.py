import random

from agents.q_learning_agent import QLearningAgent
from .feature_extractors import SimpleExtractor, EnhancedExtractor, EnhancedExtractor_noFoodFlags, EnhancedExtractor_2ghosts

class ApproximateQLearningAgent(QLearningAgent):
    def __init__(self, alpha=1.0, epsilon=0.05, gamma=0.8):
        super().__init__(alpha, epsilon, gamma)
        self.feat_extractor = EnhancedExtractor_2ghosts() # EnhancedExtractor_noFoodFlags() # EnhancedSimpleExtractor() #_noFlags() #SimpleExtractor()
        self.weights = {}  # 特征权重，而非Q值表

    def get_q_value(self, state, action):
        """
        return Q(state,action) = w * featureVector
        """
        features = self.feat_extractor.get_features(state, action)
        q_value = sum(self.weights.get(f, 0.0) * value for f, value in features.items())
        return q_value
    
    def update(self, state, action, next_state, reward):
        features = self.feat_extractor.get_features(state, action)
        td_target = reward + self.discount * self.get_state_value(next_state)
        td_delta = td_target - self.get_q_value(state, action)

        for f in features:
            self.weights[f] = self.weights.get(f, 0.0) + self.alpha * td_delta * features[f]

    def save(self, filename):
        import pickle
        with open(filename, 'wb') as f:
            pickle.dump(self.weights, f)
        print(f"Weights saved to {filename}. Total features: {len(self.weights)}")

    def load(self, filename):
        import pickle
        with open(filename, 'rb') as f:
            self.weights = pickle.load(f)
        print(f"Weights loaded from {filename}. Total features: {len(self.weights)}")

