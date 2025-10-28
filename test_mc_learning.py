"""
测试 MC Learning Agent 在 smallGrid 地图上的性能
用于补充实验报告中的数据
"""

from agents.mc_learning_agent import MCLearningAgent
from pacman_env import PacmanEnv
import json

PACMAN_ACTIONS = {
    'North': 0,
    'South': 1,
    'East': 2,
    'West': 3,
    'Stop': 4
}

# 配置
LAYOUT = 'smallGrid.lay'
TEST_EPISODES = 100
USE_GRAPHICS = False  # 测试时不显示图形界面

# 检查点路径 - 需要根据实际情况调整
CHECKPOINT_PATH = './checkpoints/mc_agent.pkl'

if __name__ == "__main__":
    # 创建环境
    env = PacmanEnv(use_graphics=USE_GRAPHICS)
    
    # 创建并加载 agent
    agent = MCLearningAgent(alpha=0, epsilon=0, gamma=0.8)
    agent.set_test_mode()
    
    try:
        agent.load(CHECKPOINT_PATH)
        print(f"Loaded checkpoint from: {CHECKPOINT_PATH}")
    except Exception as e:
        print(f"Warning: Could not load checkpoint: {e}")
        print("Testing with untrained agent...")
    
    # 测试
    score_list = []
    win_list = []
    
    print(f"\nTesting MC Learning Agent on {LAYOUT}")
    print(f"Running {TEST_EPISODES} episodes...\n")
    
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
            'test_episodes': TEST_EPISODES,
            'agent_type': 'MCLearningAgent',
        },
        'scores': score_list,
        'wins': win_list,
        'highest_score': highest_score,
        'average_score': average_score,
        'win_rate': win_rate
    }
    
    result_file = f'test_results_MCLearningAgent_{LAYOUT[:-4]}_{TEST_EPISODES}ep.json'
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\nResults saved to: {result_file}")
    print("\n将以下数据填入实验报告：")
    print(f"MC Learning & {average_score:.1f} & {win_rate:.1f} & {highest_score:.1f} \\\\")
