"""Diagnostic script to see what's happening with layout changes."""
import gymnasium
import pacman_env
import time

env = gymnasium.make('Pacman-v1')

episode_count = 0
max_episodes = 3

print("Starting diagnostic test...")
print("=" * 70)

while episode_count < max_episodes:
    episode_count += 1
    print(f"\n{'='*70}")
    print(f"EPISODE {episode_count}")
    print(f"{'='*70}")
    
    obs, info = env.reset()
    
    # Record initial layout
    initial_layout = env.unwrapped.layout
    initial_width = initial_layout.width
    initial_height = initial_layout.height
    print(f"Initial layout size: {initial_width} x {initial_height}")
    print(f"Food remaining: {info.get('food_remaining', 'N/A')}")
    
    i = 0
    step_count = 0
    max_steps_per_episode = 20
    
    while i < max_steps_per_episode:
        i += 1
        step_count += 1
        
        # Check if layout changed before step
        current_layout = env.unwrapped.layout
        if current_layout.width != initial_width or current_layout.height != initial_height:
            print(f"\n⚠️  LAYOUT CHANGED BEFORE STEP {i}!")
            print(f"   Old size: {initial_width} x {initial_height}")
            print(f"   New size: {current_layout.width} x {current_layout.height}")
            print(f"   Breaking inner loop...")
            break
        
        # Take action
        action = env.action_space.sample()
        try:
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Step {i}: reward={reward:.1f}, terminated={terminated}, truncated={truncated}, food={info.get('food_remaining', '?')}")
        except Exception as e:
            print(f"\n❌ ERROR in step {i}: {e}")
            import traceback
            traceback.print_exc()
            break
        
        # Check if layout changed after step
        current_layout = env.unwrapped.layout
        if current_layout.width != initial_width or current_layout.height != initial_height:
            print(f"\n⚠️  LAYOUT CHANGED AFTER STEP {i}!")
            print(f"   Old size: {initial_width} x {initial_height}")
            print(f"   New size: {current_layout.width} x {current_layout.height}")
            print(f"   Breaking inner loop...")
            break
        
        # Try render
        try:
            env.render()
        except Exception as e:
            print(f"\n❌ ERROR in render at step {i}: {e}")
            # Don't break, continue without rendering
        
        # Check if layout changed after render
        current_layout = env.unwrapped.layout
        if current_layout.width != initial_width or current_layout.height != initial_height:
            print(f"\n⚠️  LAYOUT CHANGED AFTER RENDER {i}!")
            print(f"   Old size: {initial_width} x {initial_height}")
            print(f"   New size: {current_layout.width} x {current_layout.height}")
            print(f"   Breaking inner loop...")
            break
        
        # Check if episode ended
        if terminated or truncated:
            print(f"\n✓ Episode naturally ended at step {i}")
            print(f"  Reason: {'Won' if env.unwrapped.game.state.isWin() else 'Lost' if env.unwrapped.game.state.isLose() else 'Truncated'}")
            break
        
        time.sleep(0.05)
    
    print(f"\nEpisode {episode_count} completed after {step_count} steps")
    
    if i >= max_steps_per_episode:
        print(f"✓ Reached max steps ({max_steps_per_episode}) without layout change")
    
    time.sleep(0.2)

print(f"\n{'='*70}")
print(f"✓ Test completed: {episode_count} episodes")
print(f"{'='*70}")
env.close()
