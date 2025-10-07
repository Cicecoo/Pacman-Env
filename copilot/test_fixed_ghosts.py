"""
Quick test to verify ghost movement fix
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pacman_env.envs.pacman_env import PacmanEnv
import numpy as np

def test_ghost_movement():
    """Test that ghosts move properly without crashes"""
    
    env = PacmanEnv()
    
    print("Testing ghost movement fix...")
    print("Running 10 episodes with 50 steps each")
    print()
    
    total_steps = 0
    crashes = 0
    stuck_episodes = 0
    
    for episode in range(10):
        obs, info = env.reset()
        prev_ghosts = obs['ghosts'].copy()
        stuck_count = 0
        
        for step in range(50):
            action = env.action_space.sample()
            
            try:
                obs, reward, terminated, truncated, info = env.step(action)
                total_steps += 1
                
                # Check if ghosts moved
                curr_ghosts = obs['ghosts']
                if np.array_equal(prev_ghosts, curr_ghosts):
                    stuck_count += 1
                else:
                    stuck_count = 0
                
                prev_ghosts = curr_ghosts.copy()
                
                if terminated or truncated:
                    break
                    
            except Exception as e:
                if "Illegal ghost action" in str(e):
                    print(f"  Episode {episode+1}: CRASHED with '{e}'")
                    crashes += 1
                    break
                else:
                    raise
        
        if stuck_count > 10:
            stuck_episodes += 1
            print(f"  Episode {episode+1}: Ghosts stuck for {stuck_count} steps")
    
    env.close()
    
    print()
    print("=" * 60)
    print("Test Results:")
    print(f"  Total steps executed: {total_steps}")
    print(f"  Crashes (Illegal ghost action): {crashes}")
    print(f"  Episodes with stuck ghosts: {stuck_episodes}")
    
    if crashes == 0 and stuck_episodes == 0:
        print()
        print("✓ SUCCESS: All tests passed!")
        print("  - No 'Illegal ghost action' crashes")
        print("  - Ghosts moving properly")
    else:
        print()
        print("✗ FAILURE: Issues detected")
        if crashes > 0:
            print(f"  - {crashes} crashes occurred")
        if stuck_episodes > 0:
            print(f"  - {stuck_episodes} episodes had stuck ghosts")
    
    print("=" * 60)

if __name__ == "__main__":
    test_ghost_movement()
