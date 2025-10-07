from PIL import Image
import gymnasium
import pacman_env
import time

env = gymnasium.make('Pacman-v1')
# env.seed(1)


done = False

while True:
    done = False
    env.reset()
    i = 0
    while i < 100:
        i += 1
        action = env.action_space.sample() # 随机采样一个动作
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        