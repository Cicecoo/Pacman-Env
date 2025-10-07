from gymnasium.envs.registration import register

register(
    id='Pacman-v1',
    entry_point='pacman_env.envs:PacmanEnv',
)

register(
    id='Pacman-v0',
    entry_point='pacman_env.envs:PacmanEnv_copy',
)
