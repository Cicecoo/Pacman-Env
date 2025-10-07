import gymnasium
from gymnasium import spaces
from gymnasium.utils import seeding
import numpy as np

from .ucb_pacman.graphicsDisplay import PacmanGraphics, DEFAULT_GRID_SIZE
from .ucb_pacman.game import Actions
from .ucb_pacman.pacman import ClassicGameRules
from .ucb_pacman.layout import getLayout, getRandomLayout
from .ucb_pacman.ghostAgents import DirectionalGhost
from .ucb_pacman.pacmanAgents import OpenAIAgent

from gymnasium.utils import seeding

import json
import os


MAX_GHOSTS = 5

PACMAN_ACTIONS = ['North', 'South', 'East', 'West', 'Stop']

PACMAN_DIRECTIONS = ['North', 'South', 'East', 'West']

MAX_EP_LENGTH = 100

# Load layout params (best-effort, fall back to empty dict if missing)
try:
    base_dir = os.path.dirname(__file__)
    params_path = os.path.abspath(os.path.join(base_dir, '..', '..', 'layout_params.json'))
    layout_params = json.load(open(params_path))
except Exception:
    layout_params = {}

class PacmanEnv(gymnasium.Env):
    """Gymnasium-style Pacman environment.
    
    This environment provides a Pacman game interface following the Gymnasium API.
    Actions include 5 directions: North, South, East, West, and Stop.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, use_dict_obs=True, max_ghosts=MAX_GHOSTS):
        """Initialize the Pacman environment.
        
        Args:
            render_mode: One of ["human", "rgb_array"] or None
            use_dict_obs: If True, use Dict observation space; if False, use image only
            max_ghosts: Maximum number of ghosts in the game
        """
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.use_dict_obs = use_dict_obs
        self.max_ghosts = max_ghosts
        
        # Action space: 5 actions (North, South, East, West, Stop)
        self.action_space = spaces.Discrete(len(PACMAN_ACTIONS))
        
        
        self.observation_space = spaces.Dict({
            "agent": spaces.Box(low=0, high=100, shape=(2,), dtype=np.float32),  # Pacman position (x, y)
            "agent_direction": spaces.Discrete(4),  # 0=North, 1=South, 2=East, 3=West
            "ghosts": spaces.Box(low=0, high=100, shape=(self.max_ghosts, 2), dtype=np.float32),  # Ghost positions
            "ghost_scared": spaces.MultiBinary(self.max_ghosts),  # Whether each ghost is scared
            "food": spaces.Box(low=0, high=1, shape=(100, 100), dtype=np.uint8),  # Food grid (will be resized)
            "capsules": spaces.Box(low=0, high=100, shape=(10, 2), dtype=np.float32),  # Power capsule positions
            "walls": spaces.Box(low=0, high=1, shape=(100, 100), dtype=np.uint8),  # Wall grid (will be resized)
            "image": spaces.Box(low=0, high=255, shape=(84, 84, 3), dtype=np.uint8),  # Visual observation
        })
        
        
        self.display = PacmanGraphics(1.0)
        self.location = None
        self.viewer = None
        
        self.layout = None
        self.np_random = None
        self.terminated = False
        self.truncated = False

    def _update_observation_space(self):
        """Update observation space based on actual layout dimensions."""
        
        self.observation_space = spaces.Dict({
            "agent": spaces.Box(low=0, high=max(self.layout.width, self.layout.height), 
                               shape=(2,), dtype=np.float32),
            "agent_direction": spaces.Discrete(4),
            "ghosts": spaces.Box(low=0, high=max(self.layout.width, self.layout.height), 
                                shape=(self.max_ghosts, 2), dtype=np.float32),
            "ghost_scared": spaces.MultiBinary(self.max_ghosts),
            "food": spaces.Box(low=0, high=1, shape=(self.layout.width, self.layout.height), 
                              dtype=np.uint8),
            "capsules": spaces.Box(low=0, high=max(self.layout.width, self.layout.height), 
                                  shape=(10, 2), dtype=np.float32),
            "walls": spaces.Box(low=0, high=1, shape=(self.layout.width, self.layout.height), 
                               dtype=np.uint8),
            "image": spaces.Box(low=0, high=255, shape=(84, 84, 3), dtype=np.uint8),
        })

    def _choose_layout(self, random_layout=True, chosen_layout=None):
        """Choose a layout for the game."""
        if random_layout:
            self.layout = getRandomLayout(layout_params, self.np_random)
        else:
            if chosen_layout is None:
                chosen_layout = 'smallClassic'
            self.layout = getLayout(chosen_layout)

    def _get_obs(self):
        """Get current observation (following grid_world pattern)."""
        if not self.use_dict_obs:
            # Simple image-only observation
            return self._get_image()
        
        # Structured observation
        # Agent info
        agent_pos = np.array(self.game.state.data.agentStates[0].getPosition(), dtype=np.float32)
        agent_direction = PACMAN_DIRECTIONS.index(self.game.state.data.agentStates[0].getDirection())
        
        # Ghost info
        ghost_positions = np.zeros((self.max_ghosts, 2), dtype=np.float32)
        ghost_scared = np.zeros(self.max_ghosts, dtype=np.int8)
        for i, agent_state in enumerate(self.game.state.data.agentStates[1:]):
            if i < self.max_ghosts:
                ghost_positions[i] = agent_state.getPosition()
                ghost_scared[i] = 1 if agent_state.scaredTimer > 0 else 0
        
        # Food grid
        food_grid = np.array(self.game.state.data.food.data, dtype=np.uint8).T
        
        # Capsules
        capsules_list = self.game.state.data.capsules
        capsules = np.zeros((10, 2), dtype=np.float32)
        for i, cap in enumerate(capsules_list[:10]):
            capsules[i] = cap
        
        # Walls
        walls_grid = np.array(self.game.state.data.layout.walls.data, dtype=np.uint8).T
        
        # Image observation
        image = self._get_image()
        
        return {
            "agent": agent_pos,
            "agent_direction": agent_direction,
            "ghosts": ghost_positions,
            "ghost_scared": ghost_scared,
            "food": food_grid,
            "capsules": capsules,
            "walls": walls_grid,
            "image": image,
        }

    def _get_info(self):
        """Get auxiliary information (following grid_world pattern)."""
        # Calculate useful metrics
        food_left = self.game.state.getNumFood()
        capsules_left = len(self.game.state.getCapsules())
        
        # Distance to nearest ghost
        ghost_distances = [
            np.linalg.norm(np.array(self.location) - np.array(g))
            for g in self.ghostLocations
        ]
        min_ghost_distance = min(ghost_distances) if ghost_distances else float('inf')
        
        # Check if any ghost is scared
        any_ghost_scared = any(
            agent.scaredTimer > 0 
            for agent in self.game.state.data.agentStates[1:]
        )
        
        return {
            "step": self.step_counter,
            "cumulative_reward": self.cumulative_reward,
            "food_remaining": food_left,
            "capsules_remaining": capsules_left,
            "illegal_moves": self.illegal_move_counter,
            "min_ghost_distance": min_ghost_distance,
            "any_ghost_scared": any_ghost_scared,
            "is_win": self.game.state.isWin(),
            "is_lose": self.game.state.isLose(),
        }

    def reset(self, seed=None, options=None, layout=None):
        # Seed RNG using gymnasium helper
        super().reset(seed=seed)
        
        # Ensure np_random exists - CRITICAL for layout generation
        if not hasattr(self, 'np_random') or self.np_random is None:
            self.np_random, _ = seeding.np_random(seed)

        if layout is not None:
            self.layout = getLayout(layout)
        else:
            self._choose_layout(random_layout=True)

        self.step_counter = 0
        self.cumulative_reward = 0

        self.terminated = False  # ← CRITICAL: Reset terminated flag
        self.truncated = False   # ← CRITICAL: Reset truncated flag

        self._update_observation_space()

        # we don't want super powerful ghosts
        self.ghosts = [DirectionalGhost( i+1, prob_attack=0.2, prob_scaredFlee=0.2) for i in range(MAX_GHOSTS)]

        # this agent is just a placeholder for graphics to work
        self.pacman = OpenAIAgent()

        self.rules = ClassicGameRules(300)
        self.rules.quiet = False

        self.game = self.rules.newGame(self.layout, self.pacman, self.ghosts,
            self.display, False, False)

        self.game.init()

        self.display.initialize(self.game.state.data)
        self.display.updateView()

        self.location = self.game.state.data.agentStates[0].getPosition()
        self.ghostLocations = [a.getPosition() for a in self.game.state.data.agentStates[1:]]
        self.ghostInFrame = any([np.sum(np.abs(np.array(g) - np.array(self.location))) <= 2 for g in self.ghostLocations])

        self.location_history = [self.location]
        self.orientation = PACMAN_DIRECTIONS.index(self.game.state.data.agentStates[0].getDirection())
        self.orientation_history = [self.orientation]
        self.illegal_move_counter = 0

        self.cumulative_reward = 0

        info = {
            'past_loc': [self.location_history[-1]],
            'curr_loc': [self.location_history[-1]],
            'past_orientation': [[self.orientation_history[-1]]],
            'curr_orientation': [[self.orientation_history[-1]]],
            'illegal_move_counter': [self.illegal_move_counter],
            'ghost_positions': [self.ghostLocations],
            'ghost_in_frame': [self.ghostInFrame],
            'step_counter': [[0]],
        }

        observation = self._get_obs()
        info = self._get_info()
        
        # if self.render_mode == 'human':
        #     self.render()
        
        return observation, info

    def step(self, action):
        if self.terminated or self.truncated:
            print("Warning: step() called after episode has terminated or truncated. Returning last observation.")
            self.step_counter += 1
            obs = self._get_obs() if self.use_dict_obs else np.zeros(self.observation_space.shape, dtype=np.uint8)
            info = self._get_info()
            return obs, 0.0, True, False, info

        pacman_action = PACMAN_ACTIONS[action]

        legal_actions = self.game.state.getLegalPacmanActions()
        illegal_action = False
        if pacman_action not in legal_actions:
            self.illegal_move_counter += 1
            illegal_action = True
            pacman_action = 'Stop' # Stop is always legal

        reward = self.game.step(pacman_action)
        self.cumulative_reward += reward
        # reward shaping for illegal actions
        if illegal_action:
            reward -= 10

        self.terminated = self.game.state.isWin() or self.game.state.isLose()

        self.location = self.game.state.data.agentStates[0].getPosition()
        self.location_history.append(self.location)
        self.ghostLocations = [a.getPosition() for a in self.game.state.data.agentStates[1:]]

        self.orientation = PACMAN_DIRECTIONS.index(self.game.state.data.agentStates[0].getDirection())
        self.orientation_history.append(self.orientation)

        extent = (self.location[0] - 1, self.location[1] - 1), (self.location[0] + 1, self.location[1] + 1)
        self.ghostInFrame = any([g[0] >= extent[0][0] and g[1] >= extent[0][1] and 
                                 g[0] <= extent[1][0] and g[1] <= extent[1][1] 
                                 for g in self.ghostLocations])
        
        self.step_counter += 1
        
        if self.step_counter >= MAX_EP_LENGTH:
            self.truncated = True

        observation = self._get_obs()
        info = self._get_info()
        
        if self.terminated or self.truncated:
            info['episode'] = {'r': self.cumulative_reward, 'l': self.step_counter}

        return observation, reward, self.terminated, self.truncated, info

    def get_action_meanings(self):
        return PACMAN_ACTIONS

    def _get_image(self, image_sz=(84, 84)):
        image = self.display.image
        w, h = image.size
        grid_size_x = w / float(self.layout.width)
        grid_size_y = h / float(self.layout.height)

        extent = [
            grid_size_x * (self.location[0] - 1),
            grid_size_y * (self.layout.height - (self.location[1] + 2.2)),
            grid_size_x * (self.location[0] + 2),
            grid_size_y * (self.layout.height - (self.location[1] - 1.2)),
        ]
        extent = tuple([int(e) for e in extent])
        image = image.crop(extent).resize(image_sz)
        return np.array(image)

    def render(self):
        img = self._get_image()
        if self.render_mode == 'rgb_array':
            return img
        if self.render_mode == 'human' or self.render_mode is None:
            from gymnasium.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)
            return self.viewer.isopen

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
        try:
            self.display.finish()
        except Exception:
            pass

    def __del__(self):
        self.close()
