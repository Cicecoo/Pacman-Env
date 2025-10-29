# 实验数据补充指南

本文档说明如何运行测试脚本来补充实验报告中缺失的数据。

## 需要补充的数据

根据 `report/body/expt.tex` 中的标记，需要补充以下数据：

1. ✅ **MC Learning 在 smallGrid 上的性能** (`train_and_test_mc.py`)
   - 平均分数
   - 胜率
   - 最高分数

2. ✅ **表格式特征表示 vs 线性函数近似对比** (`test_tabular_feature.py`)
   - 表格式特征表示的性能
   - 对比泛化能力

3. ⚪ **内存占用对比**（可选，`analyze_memory.py`）
   - Q-Learning 的存储条目数
   - Approximate Q-Learning 的存储条目数

4. ⚪ **仅距离特征的消融实验**（可选）
5. ⚪ **特征权重演化分析**（可选）

## 快速开始

### 方法 1：一键运行所有测试

```bash
python run_all_tests.py
```

这个脚本会：
- 按顺序运行所有测试
- 提示你选择要运行的测试
- 汇总所有结果
- 生成可以直接填入报告的 LaTeX 代码

### 方法 2：单独运行测试

#### 1. 测试 MC Learning

如果已有训练好的模型：
```bash
python test_mc_learning.py
```

如果需要训练新模型：
```bash
python train_and_test_mc.py
```

**输出文件：**
- `test_results_MCLearningAgent_smallGrid_500train_100test.json`

**预期结果示例：**
```
MC Learning & 450.0 & 95.0 & 500.0 \\
```

#### 2. 表格式特征表示对比实验

```bash
python test_tabular_feature.py
```

这个脚本会：
1. 训练表格式特征表示 agent (1000 episodes)
2. 测试表格式特征表示 agent (100 episodes)
3. 加载已训练的线性函数近似 agent 进行对比

**输出文件：**
- `comparison_tabular_vs_approx.json`

**预期结果示例：**
```
表格式特征表示 & 800.0 & 60.0 & 差 \\
线性函数近似 & 1421.6 & 91.0 & 好 \\
```

#### 3. 分析内存占用（可选）

```bash
python analyze_memory.py
```

需要修改脚本中的检查点路径：
- `q_learning_ckpt`: Q-Learning 的检查点路径
- `approx_q_ckpt`: Approximate Q-Learning 的检查点路径

## 检查点文件位置

确保以下检查点文件存在：

```
./checkpoints/
├── mc_learning/
│   └── mc_agent_smallGrid_500ep.pkl          # MC Learning (需要训练)
├── q_learning-smallGrid-2000ep-a0.2-e0.1-g0.8.pkl  # Q-Learning
└── approx_q_learning_agent-smallClassic-1000ep-*.pkl  # Approx Q
```

## 训练时间估计

- **MC Learning (500 episodes)**: 约 5-10 分钟
- **表格式特征表示 (1000 episodes)**: 约 10-20 分钟
- **测试 (100 episodes)**: 约 1-2 分钟

## 填写实验报告

测试完成后，脚本会输出可直接复制到 LaTeX 的代码：

### 1. 表格式方法性能表 (表 3.1)

```latex
MC Learning & [平均分数] & [胜率] & [最高分数] \\
Q-Learning & 436.0 & 94.0 & 500.0 \\
```

### 2. 表格式特征表示对比表 (表 3.6)

```latex
表格式特征表示 & [平均分数] & [胜率] & 差 \\
线性函数近似 & 1421.6 & 91.0 & 好 \\
```

## 常见问题

### Q: 找不到检查点文件

**A:** 
- MC Learning: 运行 `train_and_test_mc.py` 训练新模型
- Q-Learning: 检查 `./checkpoints/` 或 `./` 目录
- Approximate Q: 检查 `./checkpoints/` 目录

### Q: 测试结果不理想

**A:** 
- 确保使用了训练好的模型（不是随机初始化）
- 检查超参数设置是否正确
- 确保测试时 epsilon=0 (纯贪心策略)

### Q: 如何调整训练轮次

**A:** 修改脚本中的常量：
```python
TRAIN_EPISODES = 500  # 修改为需要的值
TEST_EPISODES = 100   # 修改测试轮次
```

## 脚本说明

### train_and_test_mc.py
- 功能：训练并测试 MC Learning Agent
- 输入：无（可配置超参数）
- 输出：JSON 结果文件 + LaTeX 代码
- 时间：约 15 分钟

### test_tabular_feature.py
- 功能：对比表格式特征表示和线性函数近似
- 输入：Approximate Q 检查点（用于对比）
- 输出：JSON 结果文件 + LaTeX 代码
- 时间：约 20 分钟

### analyze_memory.py
- 功能：分析算法内存占用
- 输入：检查点文件路径
- 输出：存储条目数和内存占用
- 时间：< 1 分钟

### run_all_tests.py
- 功能：按顺序运行所有测试
- 输入：交互式选择
- 输出：汇总所有结果
- 时间：约 30-40 分钟

## 结果验证

测试完成后，检查结果是否合理：

- MC Learning 胜率应该 > 80%（在 smallGrid 上）
- 表格式特征表示胜率应该 < 线性函数近似
- Q-Learning 存储条目数应该 >> Approximate Q (10个权重)

## 下一步

1. 运行 `python run_all_tests.py`
2. 等待测试完成
3. 复制生成的 LaTeX 代码到报告中
4. 替换 `[待补充]` 标记
5. 检查数据合理性
6. 完成报告！

## 联系

如有问题，检查：
1. Python 环境是否正确
2. 依赖包是否安装
3. 检查点文件路径是否正确
