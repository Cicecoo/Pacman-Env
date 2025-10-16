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

layout = 'smallClassic.lay'
num_episodes = 500
ckpt_name = f'approx-q_{layout}_{num_episodes}ep'
first_win = True


if __name__ == "__main__":
    env = PacmanEnv(use_graphics=False)
    # agent = QLearningAgent(alpha=0.2, epsilon=0.05, gamma=0.8)
    agent = ApproximateQLearningAgent(alpha=0.2, epsilon=0.1, gamma=0.8)
    
    for episode in range(num_episodes):
        obs, info = env.reset(layout=layout)
        cur_state = env.game.state
        agent.register_initial_state(cur_state)
        done = False
        
        while not done:
            action = agent.choose_action(cur_state)
            # print(f"Action taken: {action}")

            next_obs, reward, terminated, truncated, info = env.step(PACMAN_ACTIONS[action])   
            done = terminated or truncated
            if terminated and info.get('win', False) and first_win:
                print(f"Agent won the game in episode {episode + 1}!")
                first_win = False
                agent.save(ckpt_name+'_first-win{}.pkl'.format(episode + 1))

            new_state = env.game.state
            agent.observe_transition(new_state)
            cur_state = new_state

        agent.reach_terminal_state(cur_state)
        # print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}")

    print("Training completed.")
    agent.save(ckpt_name+'.pkl')

    print("Testing the trained agent...")
    agent.set_test_mode()
    test_episodes = 10
    # env.use_graphics = True
    env = PacmanEnv(use_graphics=True)

    for episode in range(test_episodes):
        obs, info = env.reset(layout=layout)
        cur_state = env.game.state
        agent.register_initial_state(cur_state)
        done = False

        while not done:
            action = agent.choose_action(cur_state)
            next_obs, reward, terminated, truncated, info = env.step(PACMAN_ACTIONS[action])   
            done = terminated or truncated

            new_state = env.game.state
            agent.observe_transition(new_state)
            cur_state = new_state
   
        agent.reach_terminal_state(cur_state)
        print(f"Test Episode {episode + 1}/{test_episodes}, Total Reward: {agent.episode_rewards}")

