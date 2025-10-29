"""
使用已有的权重数据重新绘制特征权重演化图
"""
import json
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
rcParams['axes.unicode_minus'] = False

def load_weight_data(json_path='weight_evolution_results.json'):
    """加载权重数据"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def extract_feature_evolution(weight_history, feature_names):
    """提取每个特征随训练的演化"""
    episodes = sorted([int(ep) for ep in weight_history.keys()])
    
    feature_evolution = {}
    for feature in feature_names:
        values = []
        for episode in episodes:
            ep_weights = weight_history[str(episode)]
            # 对该特征在所有动作上求平均绝对值
            feature_values = [abs(v) for k, v in ep_weights.items() if k.startswith(feature)]
            if feature_values:
                values.append(np.mean(feature_values))
            else:
                values.append(0)
        feature_evolution[feature] = values
    
    return episodes, feature_evolution

def plot_weight_evolution(episodes, feature_evolution, output_path='report/images/weight_evolution.png'):
    """绘制特征权重演化图"""
    
    # 分类特征：大幅度、大幅度、中小幅度
    very_large_features = [
        '#-of-normal-ghosts-1-step-away',
        
    ]
    
    medium_features = [
        'eats-food',
        '#-of-scared-ghosts-1-step-away',
        'eats-scared-ghost',
        'scared-timer'
    ]
    
    small_features = [
        'eats-capsule',
        'closest-food',
        'closest-scared-ghost',
        'closest-capsule',
        # 'bias'
    ]
    
    # 过滤存在的特征
    very_large = [f for f in very_large_features if f in feature_evolution]
    medium = [f for f in medium_features if f in feature_evolution]
    small = [f for f in small_features if f in feature_evolution]
    
    # 使用经典的低饱和度配色（类似学术论文）
    colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3', '#937860', '#DA8BC3', '#8C8C8C', '#CCB974']
    markers = ['o', 's', '^', 'v', 'D', 'p', '*', 'x', 'h']
    
    # 创建三个子图
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))
    
    # 子图1：大幅度变化的特征
    for idx, feature in enumerate(very_large):
        values = feature_evolution[feature]
        ax1.plot(episodes, values, 
                marker=markers[idx % len(markers)], 
                markersize=5, 
                label=feature, 
                linewidth=2.5,
                color=colors[idx % len(colors)],
                markevery=max(1, len(episodes)//15),
                alpha=0.85)
    
    ax1.set_xlabel('训练轮数 (Episode)', fontsize=20)
    ax1.set_ylabel('权重绝对值平均', fontsize=20)
    ax1.set_title('(a) 大幅度变化特征', fontsize=20, loc='left')
    ax1.legend(loc='upper left', fontsize=18, framealpha=0.6)
    ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    
    # 子图2：中等幅度变化的特征
    for idx, feature in enumerate(medium):
        values = feature_evolution[feature]
        ax2.plot(episodes, values, 
                marker=markers[(idx+len(very_large)) % len(markers)], 
                markersize=5, 
                label=feature, 
                linewidth=2.5,
                color=colors[(idx+len(very_large)) % len(colors)],
                markevery=max(1, len(episodes)//15),
                alpha=0.85)
    
    ax2.set_xlabel('训练轮数 (Episode)', fontsize=20)
    ax2.set_ylabel('权重绝对值平均', fontsize=20)
    ax2.set_title('(b) 中等幅度变化特征', fontsize=20, loc='left')
    ax2.legend(loc='upper left', fontsize=18, framealpha=0.6)
    ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    
    # 子图3：小幅度变化的特征
    for idx, feature in enumerate(small):
        values = feature_evolution[feature]
        ax3.plot(episodes, values, 
                marker=markers[(idx+len(very_large)+len(medium)) % len(markers)], 
                markersize=5, 
                label=feature, 
                linewidth=2.5,
                color=colors[(idx+len(very_large)+len(medium)) % len(colors)],
                markevery=max(1, len(episodes)//15),
                alpha=0.85)

    ax3.set_xlabel('训练轮数 (Episode)', fontsize=20)
    ax3.set_ylabel('权重绝对值平均', fontsize=20)
    ax3.set_title('(c) 小幅度变化特征', fontsize=20, loc='left')
    ax3.legend(loc='upper left', fontsize=18, framealpha=0.6)
    ax3.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"权重演化图已保存到: {output_path}")
    plt.close()

if __name__ == '__main__':
    # 加载数据
    data = load_weight_data()
    weight_history = data['weight_history']
    feature_names = data['feature_names']
    
    print(f"加载了 {len(weight_history)} 个时间点的权重数据")
    print(f"原始特征数量: {len(feature_names)}")
    print(f"原始特征列表: {feature_names}")
    
    # 从实际权重数据中获取所有特征名称（更可靠）
    all_features = set()
    for ep_data in weight_history.values():
        all_features.update(ep_data.keys())
    
    feature_names = sorted(list(all_features))
    print(f"\n实际特征数量: {len(feature_names)}")
    print(f"实际特征列表: {feature_names}")
    
    # 提取特征演化
    episodes, feature_evolution = extract_feature_evolution(weight_history, feature_names)
    
    # 调试：打印每个特征是否有数据
    print("\n特征数据检查:")
    for fname in ['eats-food', '#-of-normal-ghosts-1-step-away']:
        if fname in feature_evolution:
            print(f"  {fname}: 有数据，范围 {min(feature_evolution[fname]):.2f} - {max(feature_evolution[fname]):.2f}")
        else:
            print(f"  {fname}: 无数据")
    
    # 绘制图表
    plot_weight_evolution(episodes, feature_evolution)
    
    print("\n完成！")
