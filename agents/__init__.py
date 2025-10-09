"""
RL Agents for Pacman Environment
"""

from .q_learning_agent import QLearningAgent
from .approximate_q_learning_agent import ApproximateQLearningAgent
from .mc_learning_agent import MCLearningAgent, ApproximateMCLearningAgent

# Keep old MCAgent for backwards compatibility if needed
# from .mc_agent import MCAgent

__all__ = [
    'QLearningAgent', 
    'ApproximateQLearningAgent',
    'MCLearningAgent',
    'ApproximateMCLearningAgent'
]
