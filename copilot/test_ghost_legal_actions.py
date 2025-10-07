"""
Diagnostic script to check why ghosts get stuck - focus on legal actions
"""
import os
import sys
import numpy as np

# Add the parent directory to sys.path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))

from pacman_env.envs.pacman_env import PacmanEnv

def test_stuck_ghost_legal_actions():
    """Check what legal actions ghosts have when they get stuck"""
    
    env = PacmanEnv()
    
    print("="*80)
    print("Ghost Legal Actions Diagnostic")
    print("="*80)
    print()
    
    for episode in range(3):
        print(f"\n{'='*80}")
        print(f"EPISODE {episode + 1}")
        print(f"{'='*80}\n")
        
        obs, info = env.reset()
        
        # Get layout info
        walls = obs['walls']
        print(f"Layout size: {walls.shape}")
        print(f"Walls grid:")
        for y in range(walls.shape[1]-1, -1, -1):  # Top to bottom
            row = ""
            for x in range(walls.shape[0]):
                if walls[x, y]:
                    row += "#"
                else:
                    row += " "
            print(f"  y={y}: {row}")
        print()
        
        # Initial ghost positions
        ghost_locs = obs['ghosts']
        # Find actual number of ghosts (non-zero positions)
        num_ghosts = sum(1 for g in ghost_locs if not np.array_equal(g, [0, 0]))
        ghost_locs = ghost_locs[:num_ghosts]
        
        print(f"Initial ghost positions ({num_ghosts} ghosts):")
        for i, pos in enumerate(ghost_locs):
            print(f"  Ghost {i}: {pos}")
        print()
        
        # Track for stuck detection
        prev_ghost_locs = ghost_locs.copy()
        stuck_counters = {i: 0 for i in range(len(ghost_locs))}
        
        for step in range(1, 51):  # 50 steps
            # Use mostly STOP action to let ghosts move more
            if step % 3 == 0:
                action = env.action_space.sample()
            else:
                action = 4  # STOP
            obs, reward, terminated, truncated, info = env.step(action)
            
            ghost_locs = obs['ghosts'][:num_ghosts]
            
            # Check which ghosts moved
            all_stuck = True
            for i in range(len(ghost_locs)):
                moved = not np.array_equal(ghost_locs[i], prev_ghost_locs[i])
                if moved:
                    stuck_counters[i] = 0
                    all_stuck = False
                else:
                    stuck_counters[i] += 1
            
            # If any ghost stuck for >3 steps, print detailed info
            if step > 5 and any(count > 3 for count in stuck_counters.values()):
                print(f"\nStep {step:3d} | Pacman: {obs['agent']}")
                print(f"  Terminated: {terminated} | Truncated: {truncated}")
                
                for i in range(len(ghost_locs)):
                    ghost_pos = ghost_locs[i]
                    ghost_state = env.game.state.data.agentStates[i+1]
                    
                    # Get legal actions for this ghost
                    legal_actions = env.game.state.getLegalActions(i+1)
                    
                    status = "MOVED" if stuck_counters[i] == 0 else f"STUCK x{stuck_counters[i]}"
                    
                    print(f"\n  Ghost {i}: pos={ghost_pos}, dir={ghost_state.configuration.direction}")
                    print(f"    Status: {status}")
                    print(f"    Legal actions: {legal_actions}")
                    print(f"    Scared: {ghost_state.scaredTimer}")
                    
                    # Check walls around ghost
                    x, y = int(ghost_pos[0]), int(ghost_pos[1])
                    print(f"    Walls around ({x}, {y}):")
                    for direction, (dx, dy) in [('North', (0, 1)), ('South', (0, -1)), 
                                                  ('East', (1, 0)), ('West', (-1, 0))]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < walls.shape[0] and 0 <= ny < walls.shape[1]:
                            is_wall = walls[nx, ny]
                            print(f"      {direction}: ({nx}, {ny}) - {'WALL' if is_wall else 'FREE'}")
                        else:
                            print(f"      {direction}: ({nx}, {ny}) - OUT OF BOUNDS")
                
                # If all stuck, break to next episode
                if all_stuck and all(count > 3 for count in stuck_counters.values()):
                    print(f"\n⚠️  All ghosts stuck!")
                    break
            
            prev_ghost_locs = ghost_locs.copy()
            
            if terminated or truncated:
                print(f"\n✓ Episode ended at step {step}")
                break
        
        print(f"\nEpisode {episode + 1} Summary:")
        print(f"  Final stuck counters: {stuck_counters}")
    
    env.close()
    print("\n" + "="*80)
    print("Diagnostic completed")
    print("="*80)

if __name__ == "__main__":
    test_stuck_ghost_legal_actions()
