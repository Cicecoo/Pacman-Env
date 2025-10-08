from gymnasium.envs.registration import register

# 导出PacmanEnv以便直接使用
from pacman_env.envs.pacman_env import PacmanEnv

register(
    id='Pacman-v1',
    entry_point='pacman_env.envs:PacmanEnv',
)

# register(
#     id='Pacman-v0',
#     entry_point='pacman_env.envs:PacmanEnv_copy',
# )

# register(
#     id='ImprovedPacman-v0',
#     entry_point='pacman_env.envs:ImprovedPacmanEnv',
# )
