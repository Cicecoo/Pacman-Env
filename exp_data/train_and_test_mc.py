"""
训练并测试 MC Learning Agent 在 smallGrid 地图上的性能
如果已有训练好的模型，也可以直接测试
"""

from agents.mc_learning_agent import MCLearningAgent
from pacman_env import PacmanEnv
import json
import os

PACMAN_ACTIONS = {
    'North': 0,
    'South': 1,
    'East': 2,
    'West': 3,
    'Stop': 4
}

# 配置
LAYOUT = 'smallGrid.lay'
TRAIN_EPISODES = 500
TEST_EPISODES = 100
USE_GRAPHICS = False

# 超参数
ALPHA = 0.1
EPSILON = 0.05
GAMMA = 0.8

# 检查点路径
CHECKPOINT_DIR = './checkpoints/mc_learning'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, f'mc_agent_smallGrid_{TRAIN_EPISODES}ep.pkl')

def train_mc_agent():
    """训练 MC Learning Agent"""
    print("="*60)
    print(f"Training MC Learning Agent on {LAYOUT}")
    print(f"Episodes: {TRAIN_EPISODES}")
    print(f"Alpha: {ALPHA}, Epsilon: {EPSILON}, Gamma: {GAMMA}")
    print("="*60 + "\n")
    
    env = PacmanEnv(use_graphics=USE_GRAPHICS)
    agent = MCLearningAgent(alpha=ALPHA, epsilon=EPSILON, gamma=GAMMA)
    
    for episode in range(TRAIN_EPISODES):
        obs, info = env.reset(layout=LAYOUT)
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
        
        if (episode + 1) % 50 == 0:
            score = env.game.state.getScore()
            is_win = env.game.state.isWin()
            print(f"Episode {episode + 1}/{TRAIN_EPISODES}, "
                  f"Score: {score}, Win: {is_win}")
    
    # 保存模型
    agent.save(CHECKPOINT_PATH)
    print(f"\nModel saved to: {CHECKPOINT_PATH}\n")
    
    return agent

def test_mc_agent(agent):
    """测试 MC Learning Agent"""
    print("="*60)
    print(f"Testing MC Learning Agent on {LAYOUT}")
    print(f"Test Episodes: {TEST_EPISODES}")
    print("="*60 + "\n")
    
    env = PacmanEnv(use_graphics=USE_GRAPHICS)
    agent.set_test_mode()  # 设置为测试模式（epsilon=0）
    
    score_list = []
    win_list = []
    
    for episode in range(TEST_EPISODES):
        obs, info = env.reset(layout=LAYOUT)
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
        
        score = env.game.state.getScore()
        is_win = env.game.state.isWin()
        
        score_list.append(score)
        win_list.append(1 if is_win else 0)
        
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}/{TEST_EPISODES}, "
                  f"Score: {score}, Win: {is_win}")
    
    # 统计结果
    highest_score = max(score_list)
    average_score = sum(score_list) / TEST_EPISODES
    win_rate = sum(win_list) / TEST_EPISODES * 100
    
    print("\n" + "="*60)
    print("Testing completed!")
    print(f"Highest Score: {highest_score}")
    print(f"Average Score: {average_score:.1f}")
    print(f"Win Rate: {win_rate:.1f}%")
    print("="*60)
    
    # 保存结果
    results = {
        'info': {
            'layout': LAYOUT,
            'checkpoint': CHECKPOINT_PATH,
            'train_episodes': TRAIN_EPISODES,
            'test_episodes': TEST_EPISODES,
            'agent_type': 'MCLearningAgent',
            'hyperparameters': {
                'alpha': ALPHA,
                'epsilon': EPSILON,
                'gamma': GAMMA
            }
        },
        'scores': score_list,
        'wins': win_list,
        'highest_score': highest_score,
        'average_score': average_score,
        'win_rate': win_rate
    }
    
    result_file = f'test_results_MCLearningAgent_{LAYOUT[:-4]}_{TRAIN_EPISODES}train_{TEST_EPISODES}test.json'
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\nResults saved to: {result_file}")
    print("\n【实验报告数据】")
    print(f"MC Learning & {average_score:.1f} & {win_rate:.1f} & {highest_score:.1f} \\\\")
    
    return results

if __name__ == "__main__":
    # 检查是否已有训练好的模型
    if os.path.exists(CHECKPOINT_PATH):
        print(f"Found existing checkpoint: {CHECKPOINT_PATH}")
        choice = input("Do you want to (T)rain new model or (L)oad existing? [T/L]: ").strip().upper()
        
        if choice == 'L':
            agent = MCLearningAgent(alpha=0, epsilon=0, gamma=GAMMA)
            agent.load(CHECKPOINT_PATH)
            print("Loaded existing model.\n")
        else:
            agent = train_mc_agent()
    else:
        agent = train_mc_agent()
    
    # 测试
    results = test_mc_agent(agent)
