# Reward 设计交互式配置指南

## 📋 概述

`reward_config_interactive.py` 是一个交互式工具，帮助你从头设计 Pacman 环境的 reward 结构。

## 🎯 Reward 组件完整列表

### 1️⃣ 基础事件奖励（UCB Pacman 原生）

| 组件 | 名称 | UCB 原值 | 推荐值 | 说明 |
|------|------|----------|--------|------|
| `basic_food` | 吃 Food 奖励 | +10 | +10 | ✅ 核心奖励，建议保持 |
| `basic_capsule` | 吃 Capsule 奖励 | 0 | 0 ~ +10 | UCB 原生为 0，可设正值鼓励主动吃 |
| `basic_ghost_eat` | 吃 Ghost 奖励 | +200 | +200 | ✅ 仅有 ghost 时有效 |
| `basic_death` | 死亡惩罚 | -500 | -500 | ✅ 仅有 ghost 时触发 |
| `basic_win` | 胜利奖励 | +500 | +500 | ✅ 吃完所有 food |
| `basic_time_penalty` | 时间惩罚 | -1 | -1 ~ 0 | 可设为 0 移除时间压力 |

---

### 2️⃣ 动作相关奖励（Reward Shaping）

| 组件 | 名称 | 推荐值 | 说明 |
|------|------|--------|------|
| `stop_penalty` | Stop 动作惩罚 | -0.5 ~ -2.0 | 解决 agent 喜欢停在原地 |
| `illegal_action_penalty` | 非法动作惩罚 | -5.0 | ⚠️ 如在 agent 中处理，环境中不需要 |
| `reverse_direction_penalty` | 反向移动惩罚 | -0.5 ~ -1.0 | 减少 180° 掉头 |
| `oscillation_penalty` | 震荡行为惩罚 | -2.0 ~ -5.0 | 检测 A→B→A→B 模式 |

**设计原则**：
- 惩罚值应**远小于 food 奖励**（10）
- 过强的惩罚可能导致 agent 过度保守
- 建议: `stop_penalty = -0.5 到 -1.0`（food 的 5-10%）

---

### 3️⃣ 位置相关奖励

| 组件 | 名称 | 推荐值 | 说明 |
|------|------|--------|------|
| `exploration_bonus` | 探索奖励 | +0.05 ~ +0.2 | 访问新位置 |
| `revisit_penalty` | 重访惩罚 | -0.2 ~ -0.5 | 重复访问已访问位置 |
| `corner_penalty` | 困角惩罚 | -0.5 ~ -1.0 | 进入死胡同 |

**设计原则**：
- `exploration_bonus` 必须远小于 food（建议 1-2%）
- 过高会导致 agent 刷探索分而不吃 food

---

### 4️⃣ 距离相关奖励（引导学习）

| 组件 | 名称 | 推荐值 | 说明 |
|------|------|--------|------|
| `approach_food_bonus` | 接近 Food 奖励 | +0.1 ~ +0.5 | 向最近 food 靠近 |
| `leave_food_penalty` | 远离 Food 惩罚 | -0.1 ~ -0.5 | 远离最近 food |
| `approach_capsule_bonus` | 接近 Capsule 奖励 | +0.5 | 仅有 ghost 时有效 |
| `ghost_distance_reward` | Ghost 距离奖励 | 动态 | 避开或追逐 ghost |

**设计原则**：
- 接近/远离奖励应**成对使用**且**绝对值相同**
- 总和必须远小于吃到 food 的奖励
- 避免 agent 在两个 food 之间来回获取接近奖励

---

### 5️⃣ 状态相关奖励

| 组件 | 名称 | 推荐值 | 说明 |
|------|------|--------|------|
| `food_remaining_penalty` | 剩余 Food 惩罚 | -0.01 | 惩罚 = 系数 × 剩余food数 |
| `progress_reward` | 进度奖励 | +1.0 | 达到里程碑（25%, 50%, 75%） |
| `efficiency_bonus` | 效率奖励 | +0.5 | 快速吃到 food 的奖励 |

---

### 6️⃣ 高级奖励（理论保证）

| 组件 | 名称 | 说明 |
|------|------|------|
| `potential_based_shaping` | 势函数塑造 | ✅ **理论保证最优策略不变** |

**势函数塑造原理**：
```
F(s, a, s') = γ × Φ(s') - Φ(s)
```
- **Φ(s)**: 状态势函数（如到最近 food 的负距离）
- **γ**: 折扣因子（与 agent 一致）
- **理论保证**: 不改变最优策略，只加速学习

**常见势函数**：
1. `nearest_food`: Φ(s) = -distance_to_nearest_food
2. `all_food`: Φ(s) = -sum_of_all_food_distances
3. `weighted`: 根据状态复杂度加权

---

## 🎨 预设配置方案

### 方案 1: 最小配置（推荐起点）
```json
{
  "basic_food": 10.0,
  "basic_ghost_eat": 200.0,
  "basic_death": -500.0,
  "basic_win": 500.0,
  "basic_time_penalty": -1.0
}
```
- ✅ **适用场景**: 验证基础学习能力，无干扰
- ✅ **优点**: 信号清晰，易于调试
- ⚠️ **缺点**: 学习速度慢，可能出现 Stop 偏好

---

### 方案 2: 无时间惩罚
```json
{
  "basic_food": 10.0,
  "basic_win": 500.0,
  "basic_time_penalty": 0.0
}
```
- ✅ **适用场景**: 探索学习，无时间压力
- ⚠️ **注意**: episode_length 仍然有效，会 truncate

---

### 方案 3: 反 Stop 配置
```json
{
  "basic_food": 10.0,
  "basic_win": 500.0,
  "basic_time_penalty": -1.0,
  "stop_penalty": -1.0
}
```
- ✅ **适用场景**: Agent 总是停在原地
- ✅ **效果**: 强制 agent 移动

---

### 方案 4: 反震荡配置
```json
{
  "basic_food": 10.0,
  "basic_win": 500.0,
  "basic_time_penalty": -1.0,
  "oscillation_penalty": -3.0,
  "reverse_direction_penalty": -0.5
}
```
- ✅ **适用场景**: Agent 来回移动（East → West → East）
- ✅ **效果**: 检测并惩罚重复模式

---

### 方案 5: 引导学习
```json
{
  "basic_food": 10.0,
  "basic_win": 500.0,
  "basic_time_penalty": -1.0,
  "approach_food_bonus": 0.3,
  "leave_food_penalty": -0.3
}
```
- ✅ **适用场景**: MC 学习信用分配困难
- ✅ **效果**: 提供密集反馈，引导向 food 移动
- ⚠️ **风险**: 可能在两个 food 之间震荡

---

### 方案 6: 平衡配置（推荐）
```json
{
  "basic_food": 10.0,
  "basic_win": 500.0,
  "basic_time_penalty": -1.0,
  "stop_penalty": -0.5,
  "oscillation_penalty": -2.0,
  "exploration_bonus": 0.1,
  "approach_food_bonus": 0.2
}
```
- ✅ **适用场景**: 综合解决多个问题
- ✅ **优点**: 结合多种技术，平衡各项指标
- ⚠️ **注意**: 需要仔细调参，避免冲突

---

### 方案 7: 势函数塑造（理论最优）⭐
```json
{
  "basic_food": 10.0,
  "basic_win": 500.0,
  "basic_time_penalty": -1.0,
  "potential_based_shaping": {
    "enabled": true,
    "gamma": 0.99,
    "potential_type": "nearest_food"
  }
}
```
- ✅ **适用场景**: 需要理论保证
- ✅ **优点**: 加速学习且不改变最优策略
- ✅ **推荐**: 学术研究或生产环境

---

## 🚀 使用流程

### 步骤 1: 运行交互式配置器

```bash
cd d:\WorkSpace\Projects\RLdemo\repos\Pacman-Env
python reward_config_interactive.py
```

### 步骤 2: 选择配置方式

**选项 A: 使用预设方案**
1. 输入 1-8 选择预设
2. 决定是否修改
3. 保存配置文件

**选项 B: 自定义配置**
1. 输入 9 进入自定义模式
2. 按类别逐项选择：
   - 基础事件
   - 动作塑造
   - 位置塑造
   - 距离塑造
   - 状态塑造
   - 高级塑造
3. 对每一项：
   - 查看详细说明
   - 决定是否启用（y/n）
   - 输入数值（或使用默认值）

### 步骤 3: 保存配置

- 输入文件名（如 `my_reward.json`）
- 或直接回车使用 `reward_config.json`

### 步骤 4: 使用配置

配置文件格式：
```json
{
  "metadata": {
    "description": "Pacman 环境 Reward 配置",
    "version": "1.0"
  },
  "rewards": {
    "basic_food": 10.0,
    "basic_win": 500.0,
    "stop_penalty": -1.0
  }
}
```

---

## 🛠️ 集成到环境

### 方法 1: 创建自定义环境类

我接下来会为你创建 `CustomRewardPacmanEnv` 类，可以加载配置文件。

### 方法 2: 直接修改 step() 方法

在 `pacman_env.py` 中添加 reward shaping 逻辑：

```python
def step(self, action):
    # ... 原有代码 ...
    reward = self.game.step(pacman_action)
    
    # 应用 reward shaping
    if self.reward_config.get('stop_penalty') and action == 4:  # Stop action
        reward += self.reward_config['stop_penalty']
    
    # ... 继续处理 ...
```

---

## 📊 Reward 设计原则总结

### ✅ 黄金法则

1. **奖励稀疏性**: 基础事件（food, win）应该是主要信号
2. **比例原则**: Shaping reward 应远小于基础 reward
   - 建议: `shaping_reward < 10% × basic_reward`
3. **信号一致性**: 所有 shaping 都应引导向正确目标
4. **避免冲突**: 不同 shaping 不应相互矛盾

### ⚠️ 常见陷阱

1. **Exploration bonus 过高** → Agent 刷探索分不吃 food
2. **Approach/leave 不平衡** → Agent 在两个 food 之间震荡
3. **Stop penalty 过强** → Agent 撞墙也不停
4. **过度塑造** → Agent 学到 shaping 而非任务本身

### 🎯 推荐流程

1. **从最小配置开始** → 验证基础学习能力
2. **识别问题** → 使用 `diagnose_agent.py` 分析行为
3. **逐步添加 shaping** → 一次只改一项
4. **A/B 测试** → 对比不同配置的效果
5. **最终验证** → 在多个 layout 上测试泛化能力

---

## 🔧 下一步

1. **运行配置器**生成你的 reward 配置
2. **创建自定义环境**（我接下来会帮你实现）
3. **训练对比**不同配置的效果
4. **分析结果**使用诊断工具

你想先尝试哪个预设方案？或者想自定义配置？
