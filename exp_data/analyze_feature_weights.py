"""
分析 Approximate Q-Learning 训练过程中特征权重的演化
"""
import numpy as np
import json
import matplotlib.pyplot as plt
from matplotlib import rcParams
from pacman_env import PacmanEnv
from agents.approximate_q_learning_agent import ApproximateQLearningAgent

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
rcParams['axes.unicode_minus'] = False

PACMAN_ACTIONS = {
    'North': 0,
    'South': 1,
    'East': 2,
    'West': 3,
    'Stop': 4
}

def train_and_record_weights(layout='smallClassic', n_episodes=1000, record_interval=10):
    """
    训练 Approximate Q-Learning 并记录特征权重的变化
    
    Args:
        layout: 地图名称（不带.lay后缀）
        n_episodes: 训练轮数
        record_interval: 记录权重的间隔
    
    Returns:
        weight_history: 权重历史记录 {episode: {feature_action: weight}}
        feature_names: 特征名称列表
    """
    env = PacmanEnv(use_graphics=False)
    
    # 初始化智能体
    agent = ApproximateQLearningAgent(alpha=0.1, epsilon=0.05, gamma=0.8)
    
    weight_history = {}
    feature_names = None
    
    print(f"开始训练 {layout}，共 {n_episodes} episodes...")
    
    for episode in range(n_episodes):
        obs, info = env.reset(layout=f'{layout}.lay')
        cur_state = env.game.state
        agent.register_initial_state(cur_state)
        done = False
        steps = 0
        max_steps = 1000
        
        while not done and steps < max_steps:
            action = agent.choose_action(cur_state)
            obs, reward, terminated, truncated, info = env.step(PACMAN_ACTIONS[action])
            done = terminated or truncated
            next_state = env.game.state
            agent.observe_transition(next_state)
            cur_state = next_state
            steps += 1
        
        agent.reach_terminal_state(cur_state)
        
        # 记录权重
        if episode % record_interval == 0 or episode == n_episodes - 1:
            # 获取所有特征的权重
            weights_snapshot = {}
            
            # 从agent获取特征名称
            if feature_names is None:
                # 获取一个样本状态的特征
                sample_state = cur_state
                sample_features = agent.feat_extractor.get_features(sample_state, 'North')
                feature_names = list(sample_features.keys())
            
            # 记录所有权重
            for key, value in agent.weights.items():
                weights_snapshot[str(key)] = float(value)
            
            weight_history[episode] = weights_snapshot
            
            if (episode + 1) % 100 == 0:
                print(f"Episode {episode + 1}/{n_episodes} 完成，记录权重快照")
    
    env.close()
    print("训练完成！")
    
    return weight_history, feature_names


def aggregate_weights_by_feature(weight_history, feature_names):
    """
    将权重按特征聚合（对所有动作求平均或求和）
    
    Returns:
        aggregated: {episode: {feature: avg_weight}}
    """
    aggregated = {}
    
    for episode, weights in weight_history.items():
        episode_features = {}
        for feature_name in feature_names:
            # 收集该特征在所有动作上的权重
            feature_weights = []
            for key, value in weights.items():
                if key.startswith(feature_name + "_"):
                    feature_weights.append(abs(value))  # 使用绝对值
            
            if feature_weights:
                # 使用平均绝对值
                episode_features[feature_name] = np.mean(feature_weights)
        
        aggregated[episode] = episode_features
    
    return aggregated


def plot_weight_evolution(weight_history, feature_names, output_path='report/images/weight_evolution.png'):
    """
    绘制特征权重演化图
    """
    # 聚合权重
    aggregated = aggregate_weights_by_feature(weight_history, feature_names)
    
    episodes = sorted(aggregated.keys())
    
    # 选择最重要的特征进行可视化
    important_features = [
        'eats-food',
        'eats-scared-ghost',
        'eats-capsule',
        'closest-food',
        'closest-scared-ghost',
        'closest-capsule'
    ]
    
    # 过滤存在的特征
    important_features = [f for f in important_features if f in feature_names]
    
    plt.figure(figsize=(10, 6))
    
    for feature in important_features:
        values = [aggregated[ep].get(feature, 0) for ep in episodes]
        plt.plot(episodes, values, marker='o', markersize=3, label=feature, linewidth=2)
    
    plt.xlabel('训练轮数 (Episode)', fontsize=12)
    plt.ylabel('权重绝对值平均', fontsize=12)
    plt.title('特征权重随训练轮数的演化', fontsize=14)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"权重演化图已保存到: {output_path}")
    plt.close()


def analyze_weight_changes(weight_history, feature_names):
    """
    分析权重变化的统计信息
    """
    aggregated = aggregate_weights_by_feature(weight_history, feature_names)
    episodes = sorted(aggregated.keys())
    
    if len(episodes) < 2:
        print("权重记录不足，无法分析")
        return
    
    initial_episode = episodes[0]
    final_episode = episodes[-1]
    
    print("\n=== 特征权重变化分析 ===")
    print(f"初始 episode: {initial_episode}, 最终 episode: {final_episode}\n")
    
    # 计算每个特征的变化
    changes = []
    for feature in feature_names:
        initial_val = aggregated[initial_episode].get(feature, 0)
        final_val = aggregated[final_episode].get(feature, 0)
        change = final_val - initial_val
        changes.append({
            'feature': feature,
            'initial': initial_val,
            'final': final_val,
            'change': change,
            'abs_change': abs(change)
        })
    
    # 按变化幅度排序
    changes.sort(key=lambda x: x['abs_change'], reverse=True)
    
    print("特征权重变化排名（按变化幅度）：")
    print(f"{'特征名称':<30} {'初始值':>12} {'最终值':>12} {'变化量':>12}")
    print("-" * 70)
    for item in changes[:10]:  # 显示前10个
        print(f"{item['feature']:<30} {item['initial']:>12.4f} {item['final']:>12.4f} {item['change']:>+12.4f}")
    
    # 分析不同类型特征的权重
    print("\n按特征类型分组：")
    
    flag_features = [f for f in feature_names if 'eats-' in f]
    distance_features = [f for f in feature_names if 'closest-' in f]
    
    print("\n标志位特征 (Flag Features):")
    for feature in flag_features:
        final_val = aggregated[final_episode].get(feature, 0)
        print(f"  {feature:<25}: {final_val:>8.4f}")
    
    print("\n距离特征 (Distance Features):")
    for feature in distance_features:
        final_val = aggregated[final_episode].get(feature, 0)
        print(f"  {feature:<25}: {final_val:>8.4f}")
    
    return changes


def save_results_to_json(weight_history, feature_names, analysis_results, output_path='weight_evolution_results.json'):
    """
    保存结果到 JSON 文件
    """
    results = {
        'weight_history': {str(k): v for k, v in weight_history.items()},
        'feature_names': feature_names,
        'analysis': analysis_results
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n结果已保存到: {output_path}")


if __name__ == '__main__':
    # 训练并记录权重（减少训练轮数以加快速度）
    weight_history, feature_names = train_and_record_weights(
        layout='smallClassic',
        n_episodes=300,  # 减少到300轮以加快速度
        record_interval=15  # 每15个episode记录一次
    )
    
    # 分析权重变化
    analysis_results = analyze_weight_changes(weight_history, feature_names)
    
    # 绘制权重演化图
    plot_weight_evolution(weight_history, feature_names)
    
    # 保存结果
    save_results_to_json(weight_history, feature_names, analysis_results)
    
    print("\n分析完成！")
