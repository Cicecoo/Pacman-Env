import gymnasium
from gymnasium import spaces
from gymnasium.utils import seeding
import numpy as np

from .ucb_pacman.graphicsDisplay import PacmanGraphics, NullDisplay, DEFAULT_GRID_SIZE
from .ucb_pacman.game import Actions
from .ucb_pacman.pacman import ClassicGameRules
from .ucb_pacman.layout import getLayout, getRandomLayout
from .ucb_pacman.ghostAgents import DirectionalGhost
from .ucb_pacman.pacmanAgents import OpenAIAgent

from gymnasium.utils import seeding

import json
import os


PACMAN_ACTIONS = ['North', 'South', 'East', 'West', 'Stop']

PACMAN_DIRECTIONS = ['North', 'South', 'East', 'West']


# Load layout params (best-effort, fall back to empty dict if missing)
try:
    base_dir = os.path.dirname(__file__)
    params_path = os.path.abspath(os.path.join(base_dir, '..', '..', 'layout_params.json'))
    layout_params = json.load(open(params_path))
except Exception:
    layout_params = {}

class PacmanEnv(gymnasium.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, use_dict_obs=True, max_ghosts=5, use_graphics=True, episode_length=1000):
        """Initialize the Pacman environment.
        
        Args:
            render_mode: One of ["human", "rgb_array"] or None
            use_dict_obs: If True, use Dict observation space; if False, use image only
            max_ghosts: Maximum number of ghosts in the game
            use_graphics: If False, disable graphics rendering (faster, avoids Ghostscript issues)
        """
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.use_dict_obs = use_dict_obs
        self.max_ghosts = max_ghosts
        self.use_graphics = use_graphics
        
        self.episode_length = episode_length
        
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
        
        # Initialize display - use NullDisplay when graphics are disabled
        if self.use_graphics:
            self.display = PacmanGraphics(1.0)
        else:
            self.display = NullDisplay()

        self.viewer = None
        self.np_random = None

        self.layout = None
        
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
        """Get current observation """
        if not self.use_dict_obs:
            # Simple image-only observation
            return self._get_image()
        
        # Agent info
        agent_position = np.array(self.game.state.data.agentStates[0].getPosition(), dtype=np.float32)
        agent_direction = PACMAN_DIRECTIONS.index(self.game.state.data.agentStates[0].getDirection())
        
        # Ghost info
        ghost_positions = np.zeros((self.max_ghosts, 2), dtype=np.float32)
        ghost_scared = np.zeros(self.max_ghosts, dtype=np.int8)
        for i, agent_state in enumerate(self.game.state.data.agentStates[1:]):
            if i < self.max_ghosts:
                ghost_positions[i] = agent_state.getPosition()
                ghost_scared[i] = 1 if agent_state.scaredTimer > 0 else 0
        
        # Food grid
        food = np.array(self.game.state.data.food.data, dtype=np.uint8).T
        
        # Capsules
        capsules_list = self.game.state.data.capsules
        capsules = np.zeros((10, 2), dtype=np.float32)
        for i, cap in enumerate(capsules_list[:10]):
            capsules[i] = cap
        
        # Walls
        walls = np.array(self.game.state.data.layout.walls.data, dtype=np.uint8).T
        
        # Image observation
        image = self._get_image()
        
        return {
            "agent": agent_position,
            "agent_direction": agent_direction,
            "ghosts": ghost_positions,
            "ghost_scared": ghost_scared,
            "food": food,
            "capsules": capsules,
            "walls": walls,
            "image": image,
        }

    def _get_info(self):
        return {}

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

        self.steps = 0
        self.terminated = False  # ← CRITICAL: Reset terminated flag
        self.truncated = False   # ← CRITICAL: Reset truncated flag

        self._update_observation_space()

        # we don't want super powerful ghosts
        self.ghosts = [DirectionalGhost( i+1, prob_attack=0.2, prob_scaredFlee=0.2) for i in range(self.max_ghosts)]
        # this agent is just a placeholder for graphics to work
        self.pacman = OpenAIAgent()

        self.rules = ClassicGameRules(300)
        self.rules.quiet = False

        self.game = self.rules.newGame(self.layout, self.pacman, self.ghosts,
            self.display, False, False)
        self.game.init()

        # Update display only if graphics are enabled
        if self.use_graphics and self.display is not None:
            self.display.initialize(self.game.state.data)
            self.display.updateView()
        

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def get_legal_actions(self):
        """
        获取当前状态下的合法动作列表（索引）
        Returns:
            list: 合法动作的索引列表，例如 [0, 2, 3, 4] 表示 North, East, West, Stop 合法
        """
        legal_actions_str = self.game.state.getLegalPacmanActions()
        legal_actions_idx = [PACMAN_ACTIONS.index(a) for a in legal_actions_str]
        return legal_actions_idx
    
    def action_masks(self):
        """
        动作掩码
        Returns:
            np.ndarray: 长度为 action_space_size 的数组，1 表示合法，0 表示非法
        """
        mask = np.zeros(len(PACMAN_ACTIONS), dtype=np.int8)
        legal_actions = self.get_legal_actions()
        mask[legal_actions] = 1
        return mask


    def step(self, action):
        if self.terminated or self.truncated:
            print("Warning: step() called after episode has terminated or truncated. Returning last observation.")
            self.steps += 1
            obs = self._get_obs() if self.use_dict_obs else np.zeros(self.observation_space.shape, dtype=np.uint8)
            info = self._get_info()
            return obs, 0.0, True, False, info

        # 直接执行动作，不再检查是否合法（由 Agent 负责）
        reward = self.game.step(PACMAN_ACTIONS[action])
        
        self.steps += 1
        self.terminated = self.game.state.isWin() or self.game.state.isLose()
        self.truncated = self.steps >= self.episode_length


        observation = self._get_obs()
        info = self._get_info()
        
        return observation, reward, self.terminated, self.truncated, info

    def get_action_meanings(self):
        return PACMAN_ACTIONS

    def _get_image(self, image_sz=(84, 84)):
        # Return blank image if graphics are disabled
        if not self.use_graphics or self.display is None:
            return np.zeros((*image_sz, 3), dtype=np.uint8)
        
        # Get the full game image and resize it
        image = self.display.image
        image = image.resize(image_sz)
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
        if self.use_graphics and self.display is not None:
            try:
                self.display.finish()
            except Exception:
                pass

    def __del__(self):
        self.close()
