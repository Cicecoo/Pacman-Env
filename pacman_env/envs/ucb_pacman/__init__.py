"""
UCB Pacman game logic package.

This package contains the core game logic from UC Berkeley's Pacman AI project,
adapted for use as a Gymnasium environment.
"""

# Core game classes
from .game import (
    Agent,
    Actions,
    Directions,
    # GameState,
    GameStateData,
    Configuration,
    AgentState,
    Grid,
)

# Pacman game rules and main game logic
from .pacman import (
    ClassicGameRules,
    PacmanRules,
    GhostRules,
)

# Layout management
from .layout import (
    Layout,
    getLayout,
    getRandomLayout,
)

# Ghost agents
from .ghostAgents import (
    GhostAgent,
    RandomGhost,
    DirectionalGhost,
)

# Pacman agents
from .pacmanAgents import (
    OpenAIAgent,
    GreedyAgent,
    LeftTurnAgent,
)

# Graphics display
from .graphicsDisplay import (
    PacmanGraphics,
    DEFAULT_GRID_SIZE,
)

# Graphics utilities
from .graphicsUtils import (
    GraphicsUtils,
)

# Utility functions
from .util import (
    manhattanDistance,
    nearestPoint,
    Counter,
    PriorityQueue,
    normalize,
)

__all__ = [
    # Game core
    'Agent',
    'Actions',
    'Directions',
    'GameState',
    'GameStateData',
    'Configuration',
    'AgentState',
    'Grid',
    
    # Game rules
    'ClassicGameRules',
    'PacmanRules',
    'GhostRules',
    
    # Layout
    'Layout',
    'getLayout',
    'getRandomLayout',
    
    # Agents
    'GhostAgent',
    'RandomGhost',
    'DirectionalGhost',
    'OpenAIAgent',
    'GreedyAgent',
    'LeftTurnAgent',
    
    # Graphics
    'PacmanGraphics',
    'DEFAULT_GRID_SIZE',
    'GraphicsUtil',
    
    # Utils
    'manhattanDistance',
    'nearestPoint',
    'Counter',
    'PriorityQueue',
    'normalize',
]
