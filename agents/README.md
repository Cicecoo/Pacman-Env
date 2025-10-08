# Monte Carlo Learning Agent for Pacman

这是一个为 Pacman 环境实现的 Monte Carlo Learning Agent，可作为强化学习的学习示例。

## 特性

✅ **First-Visit Monte Carlo** - 使用首次访问 MC 方法估计状态-动作价值  
✅ **Epsilon-Greedy 策略** - 平衡探索与利用  
✅ **特征提取** - 从观察中提取关键特征以实现泛化  
✅ **模型持久化** - 支持保存和加载训练的模型  
✅ **训练/推理模式** - 易于切换训练和评估模式  
✅ **统计追踪** - 完整的训练统计和可视化  

## 文件结构

```
agents/
  ├── __init__.py           # 包初始化
  └── mc_agent.py           # MC Agent 实现

train_mc.py                 # 完整的训练和评估脚本
simple_mc_example.py        # 简单示例（类似 run_test.py）
```

## 快速开始

### 1. 简单示例（推荐用于学习）

```bash
# 训练 50 个 episode
python simple_mc_example.py

# 测试训练好的 agent
python simple_mc_example.py test
```

### 2. 完整训练脚本

```bash
# 训练 1000 个 episode
python train_mc.py --mode train --episodes 1000

# 评估训练好的 agent
python train_mc.py --mode eval --episodes 100

# 自定义参数
python train_mc.py --mode train --episodes 500 --max-steps 150 --model-path my_model.pkl
```

## 如何使用 MC Agent

### 基本使用流程

```python
from pacman_env.envs.pacman_env import PacmanEnv
from agents.mc_agent import MCAgent

# 1. 创建环境和 agent
env = PacmanEnv()
agent = MCAgent(
    action_space_size=5,
    gamma=0.99,
    epsilon=0.5,
    epsilon_decay=0.995,
    epsilon_min=0.05
)

# 2. 训练模式
agent.train()

# 3. 运行 episode
obs, info = env.reset()
while True:
    # 选择动作
    action = agent.select_action(obs)
    
    # 存储转移（用于 MC 学习）
    agent.store_transition(obs, action, 0)
    
    # 执行动作
    obs, reward, terminated, truncated, info = env.step(action)
    
    # 更新最后一个转移的奖励
    state, act, _ = agent.episode_buffer[-1]
    agent.episode_buffer[-1] = (state, act, reward)
    
    if terminated or truncated:
        break

# 4. Episode 结束 - 更新 Q 值
agent.end_episode()

# 5. 保存模型
agent.save("model/my_agent.pkl")
```

### 推理（评估）模式

```python
# 1. 加载训练好的模型
agent = MCAgent()
agent.load("model/my_agent.pkl")

# 2. 切换到评估模式（不探索）
agent.eval()

# 3. 运行测试
obs, info = env.reset()
while True:
    action = agent.select_action(obs)  # 贪婪选择
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
```

## MC Agent 参数说明

### 初始化参数

- `action_space_size`: 动作空间大小（Pacman 为 5）
- `gamma`: 折扣因子 γ，范围 [0, 1]，控制未来奖励的权重
- `epsilon`: 初始探索率，ε-greedy 策略的 ε 值
- `epsilon_decay`: ε 衰减率，每个 episode 后 ε = ε × decay
- `epsilon_min`: 最小 ε 值，防止完全停止探索
- `learning_rate`: Q 值更新的学习率（可选，用于增量式更新）

### 主要方法

#### `select_action(obs, legal_actions=None)`
根据 ε-greedy 策略选择动作。训练模式下会探索，评估模式下贪婪选择。

#### `store_transition(obs, action, reward)`
将转移存储到 episode buffer 中，用于 episode 结束后的 MC 更新。

#### `end_episode()`
处理完成的 episode，使用 First-Visit MC 更新 Q 值。

#### `train()` / `eval()`
切换训练/评估模式。

#### `save(filepath)` / `load(filepath)`
保存/加载模型。

#### `get_stats()` / `print_stats()`
获取/打印训练统计信息。

## 特征提取

Agent 从观察中提取以下特征来表示状态：

1. **Agent 方向** - 4 个方向（North, South, East, West）
2. **到最近 Ghost 的距离** - 离散化为 4 个区间（很近、近、中、远）
3. **最近 Ghost 的方向** - 4 个方向
4. **Ghost 是否被吓到** - 布尔值
5. **相邻格子的食物** - 4 位掩码（上下左右）
6. **相邻格子的墙** - 4 位掩码
7. **剩余食物数量** - 上限为 10

这种特征提取允许 agent 泛化到未见过的状态。

## Monte Carlo 学习原理

### First-Visit MC

1. **收集 Episode**: 从开始到结束收集完整的轨迹
2. **计算回报**: 对每个时间步 t，计算 G_t = Σ γ^k × r_{t+k}
3. **更新 Q 值**: 对于首次访问的 (s, a) 对，更新 Q(s,a) 为观察到的平均回报

### ε-Greedy 策略改进

- 以概率 ε 随机选择动作（探索）
- 以概率 1-ε 选择 Q 值最大的动作（利用）
- ε 随训练逐渐衰减，从探索转向利用

## 训练建议

1. **初始探索率**: 从较高的 ε (0.5-1.0) 开始
2. **衰减率**: 使用 0.995-0.999，让 agent 逐渐减少探索
3. **最小探索**: 保持小的 ε_min (0.01-0.05)，避免陷入局部最优
4. **Episode 数量**: 至少 500-1000 个 episode
5. **折扣因子**: γ = 0.99 适合 Pacman（长期规划重要）

## 性能监控

训练脚本会生成可视化曲线：

- **Episode Rewards**: 每个 episode 的总奖励
- **Episode Length**: 每个 episode 的步数
- **Win Rate**: 100 个 episode 窗口的胜率
- **Reward Distribution**: 奖励分布直方图

## 扩展建议

1. **改进特征**: 添加更多状态特征（如食物密度、Ghost 行为模式等）
2. **函数逼近**: 使用神经网络代替 Q-table
3. **其他 MC 变体**: 
   - Every-Visit MC
   - Off-policy MC with importance sampling
4. **其他算法**: 
   - TD Learning (SARSA, Q-Learning)
   - Actor-Critic
   - PPO, DQN 等深度强化学习算法

## 故障排除

### Q-table 增长过快
- 减少特征维度
- 增加特征的离散化粒度

### 训练不收敛
- 调整 γ 值
- 增加训练 episode
- 调整 ε 衰减率

### 性能不佳
- 改进特征工程
- 增加训练时间
- 尝试不同的超参数

## 参考资料

- Sutton & Barto, "Reinforcement Learning: An Introduction"
- Chapter 5: Monte Carlo Methods
- [UCB Pacman Projects](http://ai.berkeley.edu/project_overview.html)
