"""详细诊断ghost停止运动的问题 - 在未terminated时的情况"""
import gymnasium
import pacman_env
import time
import numpy as np

env = gymnasium.make('Pacman-v1')

print("=" * 80)
print("Ghost Movement Diagnostic - Non-Terminated Episodes")
print("=" * 80)

episode_count = 0
max_episodes = 3
max_steps = 50  # 增加步数以观察更多

while episode_count < max_episodes:
    episode_count += 1
    print(f"\n{'='*80}")
    print(f"EPISODE {episode_count}")
    print(f"{'='*80}\n")
    
    obs, info = env.reset()
    
    # 记录初始ghost位置
    initial_ghosts = obs['ghosts'].copy()
    print(f"Initial ghost positions:")
    for i, pos in enumerate(initial_ghosts):
        if not np.all(pos == 0):
            print(f"  Ghost {i}: {pos}")
    
    ghost_stuck_counter = {}  # 记录每个ghost静止的步数
    prev_ghosts = initial_ghosts.copy()
    
    for step in range(max_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        current_ghosts = obs['ghosts']
        pacman_pos = obs['agent']
        
        # 检查每个ghost是否移动
        ghost_moved = []
        for i, (prev_pos, curr_pos) in enumerate(zip(prev_ghosts, current_ghosts)):
            if not np.all(curr_pos == 0):  # 忽略未使用的ghost槽位
                moved = not np.array_equal(prev_pos, curr_pos)
                ghost_moved.append(moved)
                
                # 更新静止计数器
                if i not in ghost_stuck_counter:
                    ghost_stuck_counter[i] = 0
                
                if not moved:
                    ghost_stuck_counter[i] += 1
                else:
                    ghost_stuck_counter[i] = 0
        
        # 详细输出
        print(f"\nStep {step + 1:3d} | Pacman: {pacman_pos} | Reward: {reward:6.1f}")
        print(f"  Terminated: {terminated} | Truncated: {truncated}")
        
        # 显示ghost状态
        all_stuck = True
        for i, pos in enumerate(current_ghosts):
            if not np.all(pos == 0):
                stuck_count = ghost_stuck_counter.get(i, 0)
                moved_str = "MOVED" if i < len(ghost_moved) and ghost_moved[i] else "STUCK"
                print(f"  Ghost {i}: {pos} - {moved_str} (stuck for {stuck_count} steps)")
                if stuck_count < 3:
                    all_stuck = False
        
        # 警告：所有ghost停止移动
        if all_stuck and step > 5:
            print("\n" + "!"*80)
            print(f"⚠️  WARNING: ALL GHOSTS STOPPED MOVING at step {step + 1}!")
            print(f"   Terminated: {terminated}, Truncated: {truncated}")
            print(f"   Game State:")
            print(f"     - gameOver: {env.unwrapped.game.gameOver if hasattr(env.unwrapped.game, 'gameOver') else 'N/A'}")
            print(f"     - isWin: {env.unwrapped.game.state.isWin()}")
            print(f"     - isLose: {env.unwrapped.game.state.isLose()}")
            
            # 检查ghost agents
            print(f"\n   Ghost Agent States:")
            for i, agent_state in enumerate(env.unwrapped.game.state.data.agentStates[1:]):
                print(f"     Ghost {i}: pos={agent_state.getPosition()}, "
                      f"dir={agent_state.getDirection()}, "
                      f"scared={agent_state.scaredTimer}")
            
            print("!"*80)
            
            # 继续观察几步
            print("\n   Continuing for 5 more steps to confirm...")
            for extra_step in range(5):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                current_ghosts = obs['ghosts']
                print(f"   Extra step {extra_step + 1}: ", end="")
                for i, pos in enumerate(current_ghosts):
                    if not np.all(pos == 0):
                        print(f"Ghost {i}={pos} ", end="")
                print()
                if terminated or truncated:
                    print(f"   Episode ended: terminated={terminated}, truncated={truncated}")
                    break
            
            break
        
        # 更新上一步ghost位置
        prev_ghosts = current_ghosts.copy()
        
        # 如果episode结束，跳出
        if terminated or truncated:
            print(f"\n✓ Episode ended naturally at step {step + 1}")
            print(f"  Reason: {'Win' if env.unwrapped.game.state.isWin() else 'Lose' if env.unwrapped.game.state.isLose() else 'Truncated'}")
            break
        
        time.sleep(0.02)
    
    print(f"\nEpisode {episode_count} Summary:")
    print(f"  Total steps: {step + 1}")
    print(f"  Ghost stuck counters: {ghost_stuck_counter}")

print(f"\n{'='*80}")
print(f"Diagnostic completed: {episode_count} episodes")
print(f"{'='*80}")
env.close()
