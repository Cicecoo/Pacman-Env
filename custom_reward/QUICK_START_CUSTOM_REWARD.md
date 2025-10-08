# 🎯 自定义 Reward 系统 - 快速开始

## 📦 你现在拥有的工具

### 1️⃣ **交互式配置器** (`reward_config_interactive.py`)
- ✅ 逐条列出所有可选 reward 组件
- ✅ 提供 8 个预设方案
- ✅ 支持完全自定义配置
- ✅ 生成 JSON 配置文件

### 2️⃣ **自定义环境类** (`custom_reward_env.py`)
- ✅ 支持加载配置文件
- ✅ 实现了所有 reward shaping 逻辑
- ✅ 提供 reward 分解信息（base + shaped）
- ✅ 完全兼容现有训练代码

### 3️⃣ **使用示例** (`example_custom_reward.py`)
- ✅ 6 个完整示例
- ✅ 展示不同配置的效果
- ✅ 包含训练示例

### 4️⃣ **完整文档**
- ✅ `REWARD_DESIGN_GUIDE.md` - 设计原则和指南
- ✅ `CAPSULE_AND_GHOST_ANALYSIS.md` - UCB Pacman 机制分析

---

## 🚀 快速开始（3 步）

### Step 1: 设计你的 Reward

运行交互式配置器：
```bash
python reward_config_interactive.py
```

**推荐流程**：
1. 先选择一个预设方案（如"最小配置"或"平衡配置"）
2. 根据需要微调参数
3. 保存为配置文件（如 `my_reward.json`）

**8 个预设方案**：
- `最小配置` - 仅 UCB 原生，验证基础学习 ⭐
- `无时间惩罚` - 移除每步 -1，探索友好
- `反 Stop 配置` - 解决停在原地问题
- `反震荡配置` - 解决来回移动问题
- `探索导向` - 鼓励访问新位置
- `引导学习` - 接近 food 有奖励 ⭐
- `平衡配置` - 综合多种技术 ⭐
- `势函数塑造` - 理论保证最优策略不变 ⭐⭐⭐

---

### Step 2: 测试你的配置

运行示例脚本测试效果：
```bash
python example_custom_reward.py
```

选择对应的示例查看效果：
- 示例 1: 最小配置演示
- 示例 2: 反 Stop 效果
- 示例 3: 引导学习效果（看 reward breakdown）
- 示例 4: 势函数塑造效果
- 示例 5: 从配置文件加载
- 示例 6: 训练 MC agent

---

### Step 3: 用于训练

**方法 A: 修改 `train_mc.py`**

```python
# 在 train_mc.py 顶部导入
from pacman_env.envs.custom_reward_env import CustomRewardPacmanEnv

# 在 train_agent() 函数中替换环境创建
def train_agent(...):
    # 旧代码:
    # env = PacmanEnv(use_graphics=render, episode_length=max_steps)
    
    # 新代码:
    env = CustomRewardPacmanEnv(
        reward_config_file='my_reward.json',  # 你的配置文件
        use_graphics=render,
        episode_length=max_steps
    )
    
    # 其余代码不变...
```

**方法 B: 创建新的训练脚本**

```python
# train_custom_reward.py
from pacman_env.envs.custom_reward_env import CustomRewardPacmanEnv
from agents.mc_agent import MCAgent

# 定义配置
config = {
    'basic_food': 10.0,
    'basic_win': 500.0,
    'basic_time_penalty': -1.0,
    'stop_penalty': -1.0,
    'approach_food_bonus': 0.3,
}

# 创建环境
env = CustomRewardPacmanEnv(
    reward_config=config,
    use_graphics=False,
    episode_length=100
)

# 训练（与原来完全一样）
agent = MCAgent(...)
for episode in range(num_episodes):
    obs, info = env.reset(layout='smallClassic.lay')
    for step in range(max_steps):
        legal_actions = env.get_legal_actions()
        action = agent.select_action(obs, legal_actions)
        # ... 其余训练逻辑 ...
```

---

## 📊 对比实验建议

### 实验 1: 验证 Stop 问题是否解决

```bash
# 基线: 最小配置
python reward_config_interactive.py  # 选择 1 (最小配置)
# 保存为 baseline.json
python train_mc.py --config baseline.json --episodes 500

# 对比: 反 Stop 配置
python reward_config_interactive.py  # 选择 3 (反 Stop)
# 保存为 anti_stop.json
python train_mc.py --config anti_stop.json --episodes 500
```

**观察指标**：
- Stop 动作比例是否下降
- 平均 episode reward 是否提高
- Win rate 是否提高

---

### 实验 2: 不同引导强度对比

测试不同的 `approach_food_bonus` 值：

```python
# 配置 A: 无引导
{'basic_food': 10.0, 'basic_win': 500.0}

# 配置 B: 弱引导
{'basic_food': 10.0, 'basic_win': 500.0, 'approach_food_bonus': 0.1}

# 配置 C: 中等引导
{'basic_food': 10.0, 'basic_win': 500.0, 'approach_food_bonus': 0.3}

# 配置 D: 强引导
{'basic_food': 10.0, 'basic_win': 500.0, 'approach_food_bonus': 0.5}
```

**观察指标**：
- 学习速度（前 100 episodes 的平均 reward）
- 震荡问题是否出现
- 最终策略质量

---

### 实验 3: 势函数 vs 手动塑造

```python
# 配置 A: 手动塑造
{
    'basic_food': 10.0,
    'basic_win': 500.0,
    'stop_penalty': -1.0,
    'approach_food_bonus': 0.3,
    'exploration_bonus': 0.1
}

# 配置 B: 势函数塑造
{
    'basic_food': 10.0,
    'basic_win': 500.0,
    'potential_based_shaping': {
        'enabled': True,
        'gamma': 0.99,
        'potential_type': 'nearest_food'
    }
}
```

**理论预测**：
- 势函数保证最优策略不变
- 手动塑造可能更快但有风险

---

## 🔍 查看 Reward 分解

`CustomRewardPacmanEnv` 在 `info` 中提供 reward 分解：

```python
obs, reward, terminated, truncated, info = env.step(action)

if 'reward_breakdown' in info:
    print(f"Base reward: {info['reward_breakdown']['base']}")      # UCB 原生
    print(f"Shaped reward: {info['reward_breakdown']['shaped']}")  # 额外塑造
    print(f"Total reward: {info['reward_breakdown']['total']}")    # 总和
```

**用于调试**：
- 检查 shaping 是否生效
- 观察哪些步骤获得了 shaping reward
- 验证配置是否正确

---

## 📝 完整的 Reward 组件列表

### 基础事件（建议保持 UCB 原值）
- `basic_food`: +10
- `basic_capsule`: 0
- `basic_ghost_eat`: +200
- `basic_death`: -500
- `basic_win`: +500
- `basic_time_penalty`: -1

### 动作塑造（解决行为问题）
- `stop_penalty`: -0.5 ~ -2.0
- `reverse_direction_penalty`: -0.5 ~ -1.0
- `oscillation_penalty`: -2.0 ~ -5.0

### 位置塑造（探索相关）
- `exploration_bonus`: +0.05 ~ +0.2
- `revisit_penalty`: -0.2 ~ -0.5
- `corner_penalty`: -0.5 ~ -1.0

### 距离塑造（引导学习）
- `approach_food_bonus`: +0.1 ~ +0.5
- `leave_food_penalty`: -0.1 ~ -0.5
- `approach_capsule_bonus`: +0.5
- `ghost_distance_reward`: 动态

### 状态塑造（效率相关）
- `food_remaining_penalty`: -0.01
- `progress_reward`: +1.0
- `efficiency_bonus`: +0.5

### 高级塑造（理论保证）
- `potential_based_shaping`: 理论最优 ⭐⭐⭐

---

## ⚠️ 设计原则提醒

### ✅ 黄金法则
1. **Shaping reward < 10% × Basic reward**
2. **保持稀疏性** - 基础事件是主信号
3. **避免冲突** - 不同 shaping 应一致
4. **逐步调整** - 一次只改一项

### ❌ 常见陷阱
1. `exploration_bonus` 过高 → 刷探索分
2. `approach_food_bonus` 过高 → 震荡
3. `stop_penalty` 过强 → 撞墙不停
4. 过度塑造 → 学到 shaping 而非任务

---

## 🎓 推荐学习路径

### 第 1 周：基线验证
- 使用**最小配置**训练 1000 episodes
- 记录问题（Stop 偏好？震荡？）
- 不要急于添加 shaping

### 第 2 周：针对性改进
- 根据问题选择预设方案
- 对比实验（baseline vs improved）
- 分析 reward breakdown

### 第 3 周：精细调参
- 测试不同 shaping 强度
- 找到最佳平衡点
- 在多个 layout 上验证

### 第 4 周：理论方法
- 尝试**势函数塑造**
- 对比手动塑造 vs 势函数
- 总结最佳实践

---

## 📚 相关文档

- `REWARD_DESIGN_GUIDE.md` - 完整设计指南
- `CAPSULE_AND_GHOST_ANALYSIS.md` - UCB Pacman 机制
- `STOP_ACTION_ANALYSIS_SUMMARY.md` - Stop 问题分析
- `OSCILLATION_PROBLEM_ANALYSIS.md` - 震荡问题分析

---

## 💡 常见问题

### Q1: 配置文件放在哪里？
A: 与 `train_mc.py` 同目录即可，或使用绝对路径

### Q2: 如何知道哪个配置最好？
A: 运行对比实验，观察：
- Win rate
- 平均 reward
- 训练曲线稳定性
- 最终策略质量

### Q3: 势函数塑造真的更好吗？
A: 理论上是的，但实践中取决于：
- 势函数设计是否合理
- γ 是否与 agent 一致
- 任务复杂度

### Q4: 可以同时用多种 shaping 吗？
A: 可以，但要注意：
- 避免冲突（如同时惩罚和奖励同一行为）
- 控制总量（总 shaping 不超过 basic reward）
- 逐步添加，验证效果

---

## 🎉 开始你的实验！

```bash
# Step 1: 设计配置
python reward_config_interactive.py

# Step 2: 测试效果
python example_custom_reward.py

# Step 3: 开始训练
python train_mc.py  # 修改后的版本
```

祝训练顺利！🚀
