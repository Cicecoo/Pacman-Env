"""
Reward 配置对比工具

用于可视化对比不同 reward 配置的效果
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def load_config(filename):
    """加载配置文件"""
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
        return data.get('rewards', {})


def visualize_config(config, title="Reward Configuration"):
    """可视化单个配置"""
    
    # 按类别组织
    categories = {
        '基础事件': [],
        '动作塑造': [],
        '位置塑造': [],
        '距离塑造': [],
        '状态塑造': [],
        '高级塑造': []
    }
    
    # 分类映射
    category_map = {
        'basic_food': '基础事件',
        'basic_capsule': '基础事件',
        'basic_ghost_eat': '基础事件',
        'basic_death': '基础事件',
        'basic_win': '基础事件',
        'basic_time_penalty': '基础事件',
        'stop_penalty': '动作塑造',
        'illegal_action_penalty': '动作塑造',
        'reverse_direction_penalty': '动作塑造',
        'oscillation_penalty': '动作塑造',
        'exploration_bonus': '位置塑造',
        'revisit_penalty': '位置塑造',
        'corner_penalty': '位置塑造',
        'approach_food_bonus': '距离塑造',
        'leave_food_penalty': '距离塑造',
        'approach_capsule_bonus': '距离塑造',
        'ghost_distance_reward': '距离塑造',
        'food_remaining_penalty': '状态塑造',
        'progress_reward': '状态塑造',
        'efficiency_bonus': '状态塑造',
        'potential_based_shaping': '高级塑造'
    }
    
    for key, value in config.items():
        if isinstance(value, dict):
            # 复杂配置（如势函数）
            cat = category_map.get(key, '其他')
            categories[cat].append((key, '复杂配置', 0))
        else:
            cat = category_map.get(key, '其他')
            categories[cat].append((key, value, value))
    
    # 创建图表
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    axes = axes.flatten()
    
    for i, (cat_name, cat_items) in enumerate(categories.items()):
        ax = axes[i]
        
        if not cat_items:
            ax.text(0.5, 0.5, f'{cat_name}\n无配置', 
                   ha='center', va='center', fontsize=12, color='gray')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            continue
        
        # 准备数据
        names = [item[0] for item in cat_items]
        values = [item[2] for item in cat_items]
        colors = ['green' if v > 0 else 'red' if v < 0 else 'gray' for v in values]
        
        # 绘制条形图
        bars = ax.barh(names, values, color=colors, alpha=0.7, edgecolor='black')
        
        # 添加数值标签
        for j, (bar, val) in enumerate(zip(bars, values)):
            if val != 0:
                ax.text(val, j, f' {val:.2f}', 
                       va='center', ha='left' if val > 0 else 'right',
                       fontsize=9, fontweight='bold')
        
        ax.axvline(x=0, color='black', linewidth=1)
        ax.set_title(cat_name, fontweight='bold')
        ax.set_xlabel('Reward Value')
        ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    return fig


def compare_configs(config_files, labels=None):
    """对比多个配置"""
    
    if labels is None:
        labels = [f"Config {i+1}" for i in range(len(config_files))]
    
    configs = [load_config(f) for f in config_files]
    
    # 收集所有配置项
    all_keys = set()
    for config in configs:
        all_keys.update(config.keys())
    
    all_keys = sorted(all_keys)
    
    # 过滤掉复杂配置
    simple_keys = [k for k in all_keys if not isinstance(configs[0].get(k), dict)]
    
    # 创建对比图
    fig, ax = plt.subplots(figsize=(14, max(8, len(simple_keys) * 0.5)))
    
    # 准备数据
    x = np.arange(len(simple_keys))
    width = 0.8 / len(configs)
    
    for i, (config, label) in enumerate(zip(configs, labels)):
        values = [config.get(k, 0) for k in simple_keys]
        offset = (i - len(configs)/2 + 0.5) * width
        bars = ax.barh(x + offset, values, width, label=label, alpha=0.7)
        
        # 添加数值标签
        for bar, val in zip(bars, values):
            if val != 0:
                ax.text(val, bar.get_y() + bar.get_height()/2, 
                       f'{val:.2f}',
                       va='center', ha='left' if val > 0 else 'right',
                       fontsize=8)
    
    ax.set_yticks(x)
    ax.set_yticklabels(simple_keys)
    ax.axvline(x=0, color='black', linewidth=2)
    ax.set_xlabel('Reward Value', fontsize=12)
    ax.set_title('Reward Configuration Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    return fig


def analyze_config(config):
    """分析配置的统计信息"""
    
    print("\n" + "="*70)
    print("  Reward 配置分析")
    print("="*70)
    
    # 分离简单配置和复杂配置
    simple_config = {k: v for k, v in config.items() if not isinstance(v, dict)}
    complex_config = {k: v for k, v in config.items() if isinstance(v, dict)}
    
    # 统计
    positive_rewards = {k: v for k, v in simple_config.items() if v > 0}
    negative_rewards = {k: v for k, v in simple_config.items() if v < 0}
    zero_rewards = {k: v for k, v in simple_config.items() if v == 0}
    
    total_positive = sum(positive_rewards.values())
    total_negative = sum(negative_rewards.values())
    
    # 打印分析
    print(f"\n📊 基本统计:")
    print(f"  总配置项数: {len(config)}")
    print(f"  简单配置: {len(simple_config)}")
    print(f"  复杂配置: {len(complex_config)}")
    
    print(f"\n✅ 正向奖励 ({len(positive_rewards)} 项):")
    for k, v in sorted(positive_rewards.items(), key=lambda x: -x[1]):
        print(f"  {k:30s} = +{v:.2f}")
    print(f"  {'总和':30s} = +{total_positive:.2f}")
    
    print(f"\n❌ 负向奖励 ({len(negative_rewards)} 项):")
    for k, v in sorted(negative_rewards.items(), key=lambda x: x[1]):
        print(f"  {k:30s} = {v:.2f}")
    print(f"  {'总和':30s} = {total_negative:.2f}")
    
    if zero_rewards:
        print(f"\n⚪ 零值奖励 ({len(zero_rewards)} 项):")
        for k in zero_rewards.keys():
            print(f"  {k}")
    
    if complex_config:
        print(f"\n🔧 高级配置:")
        for k, v in complex_config.items():
            print(f"  {k}:")
            for sub_k, sub_v in v.items():
                print(f"    {sub_k}: {sub_v}")
    
    # 风险评估
    print(f"\n⚠️  风险评估:")
    
    risks = []
    
    # 检查 shaping 相对强度
    if 'basic_food' in config:
        food_reward = config['basic_food']
        
        shaping_keys = [
            'stop_penalty', 'exploration_bonus', 'approach_food_bonus',
            'oscillation_penalty', 'revisit_penalty'
        ]
        
        for key in shaping_keys:
            if key in config:
                ratio = abs(config[key] / food_reward)
                if ratio > 0.3:
                    risks.append(f"{key} 过强 ({ratio*100:.1f}% of food reward)")
    
    # 检查冲突
    if 'exploration_bonus' in config and 'revisit_penalty' in config:
        if abs(config['exploration_bonus']) > abs(config['revisit_penalty']):
            risks.append("exploration_bonus 可能导致刷探索分")
    
    if 'approach_food_bonus' in config and 'leave_food_penalty' not in config:
        risks.append("建议添加 leave_food_penalty 与 approach_food_bonus 配对")
    
    if risks:
        for risk in risks:
            print(f"  ⚠️  {risk}")
    else:
        print(f"  ✅ 未发现明显风险")
    
    print("\n" + "="*70)


def main():
    """主函数"""
    
    print("\n" + "="*70)
    print("  Reward 配置对比工具")
    print("="*70)
    
    print("\n功能:")
    print("1. 可视化单个配置")
    print("2. 对比多个配置")
    print("3. 分析配置统计")
    
    choice = input("\n请选择功能 (1-3): ").strip()
    
    if choice == '1':
        # 可视化单个配置
        config_file = input("输入配置文件路径: ").strip()
        if not Path(config_file).exists():
            print(f"❌ 文件不存在: {config_file}")
            return
        
        config = load_config(config_file)
        analyze_config(config)
        
        fig = visualize_config(config, title=f"Reward Configuration: {config_file}")
        plt.savefig(f"{Path(config_file).stem}_visualization.png", dpi=150, bbox_inches='tight')
        print(f"\n✅ 可视化已保存: {Path(config_file).stem}_visualization.png")
        plt.show()
    
    elif choice == '2':
        # 对比多个配置
        num_configs = int(input("输入要对比的配置数量: ").strip())
        
        config_files = []
        labels = []
        
        for i in range(num_configs):
            file_path = input(f"配置 {i+1} 文件路径: ").strip()
            if not Path(file_path).exists():
                print(f"❌ 文件不存在: {file_path}")
                return
            config_files.append(file_path)
            
            label = input(f"配置 {i+1} 标签（可选，回车跳过）: ").strip()
            labels.append(label if label else f"Config {i+1}")
        
        fig = compare_configs(config_files, labels)
        plt.savefig("reward_comparison.png", dpi=150, bbox_inches='tight')
        print(f"\n✅ 对比图已保存: reward_comparison.png")
        plt.show()
    
    elif choice == '3':
        # 分析配置
        config_file = input("输入配置文件路径: ").strip()
        if not Path(config_file).exists():
            print(f"❌ 文件不存在: {config_file}")
            return
        
        config = load_config(config_file)
        analyze_config(config)
    
    else:
        print("❌ 无效选择")


if __name__ == '__main__':
    # 示例：创建几个测试配置
    example_configs = {
        'minimal_config.json': {
            "metadata": {"description": "最小配置"},
            "rewards": {
                "basic_food": 10.0,
                "basic_win": 500.0,
                "basic_time_penalty": -1.0
            }
        },
        'anti_stop_config.json': {
            "metadata": {"description": "反 Stop 配置"},
            "rewards": {
                "basic_food": 10.0,
                "basic_win": 500.0,
                "basic_time_penalty": -1.0,
                "stop_penalty": -1.5
            }
        },
        'guided_config.json': {
            "metadata": {"description": "引导学习配置"},
            "rewards": {
                "basic_food": 10.0,
                "basic_win": 500.0,
                "basic_time_penalty": -1.0,
                "approach_food_bonus": 0.3,
                "leave_food_penalty": -0.3
            }
        }
    }
    
    # 创建示例配置文件
    create_examples = input("是否创建示例配置文件？(y/n): ").strip().lower()
    if create_examples == 'y':
        for filename, data in example_configs.items():
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"✅ 已创建: {filename}")
        print()
    
    main()
