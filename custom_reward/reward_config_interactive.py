"""
交互式 Reward 设计配置器

这个脚本帮助你从头设计 Pacman 环境的 reward 结构。
你可以逐项选择要使用的 reward 组件并设置参数。
"""

import json
import os

# ================== Reward 组件目录 ==================

REWARD_COMPONENTS = {
    # ========== 基础事件奖励 (UCB Pacman 原生) ==========
    "basic_food": {
        "name": "吃 Food 奖励",
        "description": "吃掉一个食物点",
        "default_value": 10.0,
        "ucb_original": 10,
        "category": "基础事件",
        "recommended": True,
        "notes": "UCB Pacman 原生奖励，通常不建议修改"
    },
    
    "basic_capsule": {
        "name": "吃 Capsule 奖励",
        "description": "吃掉能量豆（让 ghost 进入惊吓状态）",
        "default_value": 0.0,
        "ucb_original": 0,
        "category": "基础事件",
        "recommended": False,
        "notes": "UCB 原生为 0，价值体现在可以吃 ghost。可以设置为正值鼓励主动吃能量豆"
    },
    
    "basic_ghost_eat": {
        "name": "吃 Ghost 奖励",
        "description": "吃掉惊吓状态的 ghost",
        "default_value": 200.0,
        "ucb_original": 200,
        "category": "基础事件",
        "recommended": True,
        "notes": "UCB Pacman 原生奖励，建议保持"
    },
    
    "basic_death": {
        "name": "死亡惩罚",
        "description": "被正常状态的 ghost 吃掉",
        "default_value": -500.0,
        "ucb_original": -500,
        "category": "基础事件",
        "recommended": True,
        "notes": "UCB Pacman 原生惩罚，建议保持。在 noGhosts 环境下不会触发"
    },
    
    "basic_win": {
        "name": "胜利奖励",
        "description": "吃完所有 food 时的额外奖励",
        "default_value": 500.0,
        "ucb_original": 500,
        "category": "基础事件",
        "recommended": True,
        "notes": "UCB Pacman 原生奖励，建议保持"
    },
    
    "basic_time_penalty": {
        "name": "时间惩罚",
        "description": "每一步的固定时间惩罚",
        "default_value": -1.0,
        "ucb_original": -1,
        "category": "基础事件",
        "recommended": True,
        "notes": "UCB Pacman 原生，鼓励快速完成任务。可以设为 0 移除"
    },
    
    # ========== 动作相关奖励 ==========
    "stop_penalty": {
        "name": "Stop 动作惩罚",
        "description": "执行 Stop 动作的额外惩罚",
        "default_value": -2.0,
        "ucb_original": None,
        "category": "动作塑造",
        "recommended": False,
        "notes": "解决 agent 喜欢停在原地的问题。建议 -0.5 到 -2.0"
    },
    
    "illegal_action_penalty": {
        "name": "非法动作惩罚",
        "description": "尝试执行非法动作（撞墙）的惩罚",
        "default_value": -5.0,
        "ucb_original": None,
        "category": "动作塑造",
        "recommended": False,
        "notes": "如果在 agent 的 select_action 中已处理，环境中不需要此项"
    },
    
    "reverse_direction_penalty": {
        "name": "反向移动惩罚",
        "description": "180度掉头的惩罚",
        "default_value": -1.0,
        "ucb_original": None,
        "category": "动作塑造",
        "recommended": False,
        "notes": "减少无意义的来回移动。建议 -0.5 到 -1.0"
    },
    
    "oscillation_penalty": {
        "name": "震荡行为惩罚",
        "description": "连续在两个位置之间来回移动",
        "default_value": -5.0,
        "ucb_original": None,
        "category": "动作塑造",
        "recommended": False,
        "notes": "检测 A→B→A→B 模式。建议 -2.0 到 -5.0，检测窗口 4-6 步"
    },
    
    # ========== 位置相关奖励 ==========
    "exploration_bonus": {
        "name": "探索奖励",
        "description": "访问新位置的奖励",
        "default_value": 0.1,
        "ucb_original": None,
        "category": "位置塑造",
        "recommended": False,
        "notes": "鼓励探索地图。建议 0.05-0.2，必须远小于 food 奖励"
    },
    
    "revisit_penalty": {
        "name": "重访惩罚",
        "description": "重复访问已访问过的位置",
        "default_value": -0.5,
        "ucb_original": None,
        "category": "位置塑造",
        "recommended": False,
        "notes": "减少无效循环。建议 -0.2 到 -0.5"
    },
    
    "corner_penalty": {
        "name": "困角惩罚",
        "description": "进入死胡同或角落",
        "default_value": -1.0,
        "ucb_original": None,
        "category": "位置塑造",
        "recommended": False,
        "notes": "减少进入危险位置。建议 -0.5 到 -1.0"
    },
    
    # ========== 距离相关奖励 ==========
    "approach_food_bonus": {
        "name": "接近 Food 奖励",
        "description": "向最近的 food 靠近时的奖励",
        "default_value": 0.3,
        "ucb_original": None,
        "category": "距离塑造",
        "recommended": False,
        "notes": "引导 agent 向 food 移动。建议 0.1-0.5，配合 leave_food_penalty"
    },
    
    "leave_food_penalty": {
        "name": "远离 Food 惩罚",
        "description": "远离最近的 food 时的惩罚",
        "default_value": -0.3,
        "ucb_original": None,
        "category": "距离塑造",
        "recommended": False,
        "notes": "配合 approach_food_bonus 使用。建议相同绝对值"
    },
    
    "approach_capsule_bonus": {
        "name": "接近 Capsule 奖励",
        "description": "当有 ghost 且未惊吓时，向 capsule 靠近",
        "default_value": 0.5,
        "ucb_original": None,
        "category": "距离塑造",
        "recommended": False,
        "notes": "仅在有 ghost 环境中有意义"
    },
    
    "ghost_distance_reward": {
        "name": "Ghost 距离奖励",
        "description": "根据与 ghost 距离动态调整奖励",
        "default_value": {
            "mode": "avoid",  # "avoid" 或 "chase"
            "threshold": 3.0,
            "reward_scale": 1.0
        },
        "ucb_original": None,
        "category": "距离塑造",
        "recommended": False,
        "notes": "mode='avoid': 距离<threshold 时惩罚; mode='chase': ghost 惊吓时追逐奖励"
    },
    
    # ========== 状态相关奖励 ==========
    "food_remaining_penalty": {
        "name": "剩余 Food 惩罚",
        "description": "每步根据剩余 food 数量的惩罚",
        "default_value": -0.01,
        "ucb_original": None,
        "category": "状态塑造",
        "recommended": False,
        "notes": "鼓励快速吃 food。惩罚 = 系数 × 剩余food数"
    },
    
    "progress_reward": {
        "name": "进度奖励",
        "description": "根据已吃 food 比例的奖励",
        "default_value": 1.0,
        "ucb_original": None,
        "category": "状态塑造",
        "recommended": False,
        "notes": "达到里程碑时奖励（如 25%, 50%, 75%）"
    },
    
    "efficiency_bonus": {
        "name": "效率奖励",
        "description": "在较少步数内吃到 food 的奖励",
        "default_value": 0.5,
        "ucb_original": None,
        "category": "状态塑造",
        "recommended": False,
        "notes": "奖励 = 系数 × (1 - steps_since_last_food / threshold)"
    },
    
    # ========== 复合奖励 ==========
    "potential_based_shaping": {
        "name": "势函数塑造",
        "description": "基于势函数的奖励塑造（保证最优策略不变）",
        "default_value": {
            "enabled": True,
            "gamma": 0.99,
            "potential_type": "nearest_food"  # "nearest_food", "all_food", "custom"
        },
        "ucb_original": None,
        "category": "高级塑造",
        "recommended": True,
        "notes": "理论保证：F(s,a,s')=γΦ(s')-Φ(s)，不改变最优策略"
    },
}


# ================== 预设配置方案 ==================

PRESET_CONFIGS = {
    "minimal": {
        "name": "最小配置（仅 UCB 原生）",
        "description": "只使用 UCB Pacman 原生的奖励，无额外塑造",
        "components": {
            "basic_food": 10.0,
            "basic_capsule": 0.0,
            "basic_ghost_eat": 200.0,
            "basic_death": -500.0,
            "basic_win": 500.0,
            "basic_time_penalty": -1.0,
        }
    },
    
    "no_time_penalty": {
        "name": "无时间惩罚",
        "description": "移除每步 -1 惩罚，适合学习探索",
        "components": {
            "basic_food": 10.0,
            "basic_capsule": 0.0,
            "basic_ghost_eat": 200.0,
            "basic_death": -500.0,
            "basic_win": 500.0,
            "basic_time_penalty": 0.0,
        }
    },
    
    "anti_stop": {
        "name": "反 Stop 配置",
        "description": "解决 agent 喜欢停在原地的问题",
        "components": {
            "basic_food": 10.0,
            "basic_capsule": 0.0,
            "basic_ghost_eat": 200.0,
            "basic_death": -500.0,
            "basic_win": 500.0,
            "basic_time_penalty": -1.0,
            "stop_penalty": -1.0,
        }
    },
    
    "anti_oscillation": {
        "name": "反震荡配置",
        "description": "解决来回移动问题",
        "components": {
            "basic_food": 10.0,
            "basic_capsule": 0.0,
            "basic_ghost_eat": 200.0,
            "basic_death": -500.0,
            "basic_win": 500.0,
            "basic_time_penalty": -1.0,
            "oscillation_penalty": -3.0,
            "reverse_direction_penalty": -0.5,
        }
    },
    
    "exploration_focused": {
        "name": "探索导向",
        "description": "鼓励探索新区域",
        "components": {
            "basic_food": 10.0,
            "basic_capsule": 0.0,
            "basic_ghost_eat": 200.0,
            "basic_death": -500.0,
            "basic_win": 500.0,
            "basic_time_penalty": -1.0,
            "exploration_bonus": 0.2,
            "revisit_penalty": -0.3,
        }
    },
    
    "guided_learning": {
        "name": "引导学习",
        "description": "使用距离奖励引导 agent 向 food 移动",
        "components": {
            "basic_food": 10.0,
            "basic_capsule": 0.0,
            "basic_ghost_eat": 200.0,
            "basic_death": -500.0,
            "basic_win": 500.0,
            "basic_time_penalty": -1.0,
            "approach_food_bonus": 0.3,
            "leave_food_penalty": -0.3,
        }
    },
    
    "balanced": {
        "name": "平衡配置",
        "description": "结合多种塑造技术的平衡方案",
        "components": {
            "basic_food": 10.0,
            "basic_capsule": 0.0,
            "basic_ghost_eat": 200.0,
            "basic_death": -500.0,
            "basic_win": 500.0,
            "basic_time_penalty": -1.0,
            "stop_penalty": -0.5,
            "oscillation_penalty": -2.0,
            "exploration_bonus": 0.1,
            "approach_food_bonus": 0.2,
        }
    },
    
    "potential_based": {
        "name": "势函数塑造",
        "description": "理论保证最优策略不变的塑造方法",
        "components": {
            "basic_food": 10.0,
            "basic_capsule": 0.0,
            "basic_ghost_eat": 200.0,
            "basic_death": -500.0,
            "basic_win": 500.0,
            "basic_time_penalty": -1.0,
            "potential_based_shaping": {
                "enabled": True,
                "gamma": 0.99,
                "potential_type": "nearest_food"
            }
        }
    }
}


# ================== 交互式配置函数 ==================

def print_header(text):
    """打印标题"""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70)


def print_component_info(key, component):
    """打印 reward 组件信息"""
    print(f"\n【{component['name']}】")
    print(f"  描述: {component['description']}")
    print(f"  类别: {component['category']}")
    print(f"  默认值: {component['default_value']}")
    if component['ucb_original'] is not None:
        print(f"  UCB 原生: {component['ucb_original']}")
    if component['recommended']:
        print(f"  推荐: ✅ 是")
    else:
        print(f"  推荐: ⚠️  需根据情况选择")
    print(f"  说明: {component['notes']}")


def select_preset():
    """选择预设配置"""
    print_header("选择预设配置方案")
    
    print("\n可用预设:")
    presets = list(PRESET_CONFIGS.keys())
    for i, key in enumerate(presets, 1):
        preset = PRESET_CONFIGS[key]
        print(f"{i}. {preset['name']}")
        print(f"   {preset['description']}")
    
    print(f"{len(presets) + 1}. 自定义配置（从头逐项选择）")
    
    while True:
        try:
            choice = input(f"\n请选择 (1-{len(presets) + 1}): ").strip()
            choice = int(choice)
            if 1 <= choice <= len(presets):
                selected_key = presets[choice - 1]
                print(f"\n✅ 已选择预设: {PRESET_CONFIGS[selected_key]['name']}")
                return PRESET_CONFIGS[selected_key]['components']
            elif choice == len(presets) + 1:
                print("\n✅ 开始自定义配置")
                return None
            else:
                print("❌ 无效选择，请重试")
        except:
            print("❌ 无效输入，请输入数字")


def interactive_configure():
    """交互式配置 reward"""
    print_header("Pacman 环境 Reward 交互式配置器")
    print("\n欢迎使用 Reward 配置器！")
    print("你可以选择预设方案，或从头逐项设计 reward 结构。")
    
    # 选择预设或自定义
    config = select_preset()
    
    if config is None:
        # 自定义配置
        config = custom_configure()
    else:
        # 使用预设，询问是否修改
        print("\n是否要修改此预设配置？")
        modify = input("输入 'y' 修改，其他键跳过: ").strip().lower()
        if modify == 'y':
            config = modify_config(config)
    
    # 显示最终配置
    print_header("最终配置")
    print_config(config)
    
    # 保存配置
    save_config(config)
    
    return config


def custom_configure():
    """从头自定义配置"""
    config = {}
    
    categories = {}
    for key, comp in REWARD_COMPONENTS.items():
        cat = comp['category']
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(key)
    
    for category, keys in categories.items():
        print_header(f"配置类别: {category}")
        
        for key in keys:
            component = REWARD_COMPONENTS[key]
            print_component_info(key, component)
            
            # 询问是否启用
            enable = input(f"\n启用此项？(y/n，默认 {'y' if component['recommended'] else 'n'}): ").strip().lower()
            
            if enable == 'y' or (enable == '' and component['recommended']):
                # 询问数值
                if isinstance(component['default_value'], dict):
                    print(f"使用默认配置: {component['default_value']}")
                    use_default = input("使用默认配置？(y/n，默认 y): ").strip().lower()
                    if use_default != 'n':
                        config[key] = component['default_value']
                    else:
                        # TODO: 复杂配置的详细设置
                        config[key] = component['default_value']
                else:
                    value_input = input(f"输入数值（默认 {component['default_value']}）: ").strip()
                    if value_input == '':
                        config[key] = component['default_value']
                    else:
                        try:
                            config[key] = float(value_input)
                            print(f"✅ 已设置: {key} = {config[key]}")
                        except:
                            print(f"❌ 无效输入，使用默认值 {component['default_value']}")
                            config[key] = component['default_value']
    
    return config


def modify_config(config):
    """修改现有配置"""
    print_header("修改配置")
    print("\n当前配置项:")
    
    keys = list(config.keys())
    for i, key in enumerate(keys, 1):
        component = REWARD_COMPONENTS[key]
        print(f"{i}. {component['name']}: {config[key]}")
    
    print(f"{len(keys) + 1}. 添加新项")
    print(f"{len(keys) + 2}. 完成修改")
    
    while True:
        choice = input(f"\n选择要修改的项 (1-{len(keys) + 2}): ").strip()
        
        if choice == str(len(keys) + 2):
            break
        
        try:
            choice = int(choice)
            if 1 <= choice <= len(keys):
                key = keys[choice - 1]
                component = REWARD_COMPONENTS[key]
                print_component_info(key, component)
                
                new_value = input(f"输入新值（当前: {config[key]}，留空删除此项）: ").strip()
                if new_value == '':
                    del config[key]
                    print(f"✅ 已删除: {key}")
                else:
                    try:
                        config[key] = float(new_value)
                        print(f"✅ 已更新: {key} = {config[key]}")
                    except:
                        print("❌ 无效输入")
            elif choice == len(keys) + 1:
                # 添加新项
                add_new_component(config)
            else:
                print("❌ 无效选择")
        except:
            print("❌ 无效输入")
    
    return config


def add_new_component(config):
    """添加新的 reward 组件"""
    print("\n可添加的组件:")
    available = [k for k in REWARD_COMPONENTS.keys() if k not in config]
    
    for i, key in enumerate(available, 1):
        component = REWARD_COMPONENTS[key]
        print(f"{i}. {component['name']} ({component['category']})")
    
    choice = input(f"\n选择要添加的组件 (1-{len(available)}): ").strip()
    try:
        choice = int(choice)
        if 1 <= choice <= len(available):
            key = available[choice - 1]
            component = REWARD_COMPONENTS[key]
            print_component_info(key, component)
            
            value_input = input(f"输入数值（默认 {component['default_value']}）: ").strip()
            if value_input == '':
                config[key] = component['default_value']
            else:
                config[key] = float(value_input)
            print(f"✅ 已添加: {key} = {config[key]}")
    except:
        print("❌ 无效输入")


def print_config(config):
    """打印配置"""
    print("\n当前 Reward 配置:")
    print("-" * 70)
    
    total_positive = 0
    total_negative = 0
    
    for key, value in config.items():
        component = REWARD_COMPONENTS[key]
        print(f"  {component['name']:30s} = {value}")
        
        if isinstance(value, (int, float)):
            if value > 0:
                total_positive += value
            elif value < 0:
                total_negative += value
    
    print("-" * 70)
    print(f"  最大正奖励总和: {total_positive:.2f}")
    print(f"  最大负奖励总和: {total_negative:.2f}")


def save_config(config):
    """保存配置到 JSON 文件"""
    print("\n保存配置到文件？")
    print("输入文件名（默认 reward_config.json，直接回车使用默认）")
    filename = input(">>> ").strip()
    if filename == '':
        filename = 'reward_config.json'
    
    if not filename.endswith('.json'):
        filename += '.json'
    
    # 添加元数据
    config_with_meta = {
        "metadata": {
            "description": "Pacman 环境 Reward 配置",
            "version": "1.0"
        },
        "rewards": config
    }
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(config_with_meta, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 配置已保存到: {filename}")


def load_config(filename='reward_config.json'):
    """从 JSON 文件加载配置"""
    if os.path.exists(filename):
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data.get('rewards', {})
    return None


# ================== 主程序 ==================

if __name__ == '__main__':
    config = interactive_configure()
    
    print_header("配置完成")
    print("\n接下来你可以:")
    print("1. 使用这个配置创建自定义环境")
    print("2. 修改 pacman_env.py 或创建新的环境类")
    print("3. 在 train_mc.py 中使用新环境进行训练")
    print("\n示例代码:")
    print("""
from pacman_env.envs.custom_reward_env import CustomRewardPacmanEnv

env = CustomRewardPacmanEnv(
    reward_config='reward_config.json',
    use_graphics=False,
    episode_length=100
)
""")
