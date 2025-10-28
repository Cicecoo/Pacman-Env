"""
测量不同算法的内存占用（Q表/权重向量的存储条目数）
用于补充实验报告中的内存对比数据
"""

import pickle
import sys

def analyze_q_learning_memory(checkpoint_path):
    """分析 Q-Learning 的内存占用"""
    try:
        with open(checkpoint_path, 'rb') as f:
            data = pickle.load(f)
        
        # Q-Learning 存储 Q 值字典
        if hasattr(data, 'q_values'):
            q_values = data.q_values
        elif isinstance(data, dict) and 'q_values' in data:
            q_values = data['q_values']
        else:
            # 直接是字典
            q_values = data
        
        num_entries = len(q_values)
        
        # 估算内存大小（每个条目包含状态、动作、Q值）
        memory_bytes = sys.getsizeof(q_values)
        memory_mb = memory_bytes / (1024 * 1024)
        
        return {
            'entries': num_entries,
            'memory_mb': memory_mb
        }
    except Exception as e:
        print(f"Error loading Q-Learning checkpoint: {e}")
        return None

def analyze_approx_q_memory(checkpoint_path):
    """分析 Approximate Q-Learning 的内存占用"""
    try:
        with open(checkpoint_path, 'rb') as f:
            data = pickle.load(f)
        
        # Approximate Q-Learning 存储权重向量
        if hasattr(data, 'weights'):
            weights = data.weights
        elif isinstance(data, dict) and 'weights' in data:
            weights = data['weights']
        else:
            weights = data
        
        num_weights = len(weights)
        
        # 估算内存大小
        memory_bytes = sys.getsizeof(weights)
        memory_mb = memory_bytes / (1024 * 1024)
        
        return {
            'entries': num_weights,
            'memory_mb': memory_mb
        }
    except Exception as e:
        print(f"Error loading Approx Q checkpoint: {e}")
        return None

if __name__ == "__main__":
    print("="*60)
    print("算法内存占用分析")
    print("="*60 + "\n")
    
    # Q-Learning 检查点
    q_learning_ckpt = './checkpoints/q_learning-smallGrid-2000ep-a0.2-e0.1-g0.8.pkl'
    # 或者使用其他路径
    # q_learning_ckpt = './q_smallGrid_500ep.pkl'
    
    print(f"分析 Q-Learning: {q_learning_ckpt}")
    q_stats = analyze_q_learning_memory(q_learning_ckpt)
    
    if q_stats:
        print(f"  存储条目数: {q_stats['entries']}")
        print(f"  内存占用: {q_stats['memory_mb']:.3f} MB")
        print(f"  (约 {q_stats['memory_mb']*1024:.1f} KB)\n")
    else:
        print("  无法分析，请检查检查点路径\n")
    
    # Approximate Q-Learning 检查点
    approx_q_ckpt = './checkpoints/approx_q_learning_agent-smallClassic-1000ep-a0.2-e0.1-g0.8-EnhenceEx-CanEatCapsule.pkl'
    # 或者
    # approx_q_ckpt = './approx-q_smallClassic_2000ep.pkl'
    
    print(f"分析 Approximate Q-Learning: {approx_q_ckpt}")
    approx_stats = analyze_approx_q_memory(approx_q_ckpt)
    
    if approx_stats:
        print(f"  存储条目数(特征权重数): {approx_stats['entries']}")
        print(f"  内存占用: {approx_stats['memory_mb']:.3f} MB")
        print(f"  (约 {approx_stats['memory_mb']*1024:.1f} KB)\n")
    else:
        print("  无法分析，请检查检查点路径\n")
    
    # 输出对比结果
    if q_stats and approx_stats:
        print("="*60)
        print("对比结果：")
        print("="*60)
        print(f"Q-Learning 存储条目数: {q_stats['entries']}")
        print(f"Approx Q-Learning 存储条目数: {approx_stats['entries']}")
        print(f"条目数减少: {q_stats['entries'] / approx_stats['entries']:.1f}x")
        print(f"\n内存占用减少: {q_stats['memory_mb'] / approx_stats['memory_mb']:.1f}x")
        
        print("\n将以下数据填入实验报告：")
        print(f"Q-Learning & \\texttt{{smallGrid}} & {q_stats['entries']} \\\\")
        print(f"Approx Q-Learning & \\texttt{{smallClassic}} & {approx_stats['entries']}（特征权重） \\\\")
