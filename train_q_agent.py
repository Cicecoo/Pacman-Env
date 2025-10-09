
from agents.q_learning_agent import QLearningAgent
from agents.approximate_q_learning_agent import ApproximateQLearningAgent   

from pacman_env import PacmanEnv

PACMAN_ACTIONS = {
    'North': 0,
    'South': 1,
    'East': 2,
    'West': 3,
    'Stop': 4
}

layout = 'smallGrid.lay'

if __name__ == "__main__":
    env = PacmanEnv(use_graphics=False)
    # agent = QLearningAgent(alpha=0.2, epsilon=0.05, gamma=0.8)
    agent = ApproximateQLearningAgent(alpha=0.2, epsilon=0.4, gamma=0.8)
    num_episodes = 2000

    for episode in range(num_episodes):
        obs, info = env.reset(layout=layout)
        state = env.game.state
        agent.register_initial_state(state)
        done = False
        total_reward = 0

        while not done:
            action = agent.get_action(state)
            # print(f"Action taken: {action}")

            next_obs, reward, terminated, truncated, info = env.step(PACMAN_ACTIONS[action])   
            done = terminated or truncated

            next_state = env.game.state
            agent.observation_function(next_state)
            state = next_state
            total_reward += reward 

        agent.reach_terminal_state(state)
        # print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}")

    print("Training completed.")
    agent.save('q_learning_agent.pkl')

    print("Testing the trained agent...")
    agent.set_test_mode()
    test_episodes = 10
    # env.use_graphics = True
    env = PacmanEnv(use_graphics=True)

    for episode in range(test_episodes):
        obs, info = env.reset(layout=layout)
        state = env.game.state
        agent.register_initial_state(state)
        done = False
        total_reward = 0

        while not done:
            action = agent.get_action(state)
            next_obs, reward, terminated, truncated, info = env.step(PACMAN_ACTIONS[action])   
            done = terminated or truncated

            next_state = env.game.state
            agent.observation_function(next_state)
            state = next_state
            total_reward += reward 

        agent.reach_terminal_state(state)
        print(f"Test Episode {episode + 1}/{test_episodes}, Total Reward: {total_reward}")

