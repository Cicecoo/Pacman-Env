# 修复：Episode在第一步就终止的问题

## 问题描述

用户报告：使用 `pacman_env.py` 时，几个正常的episode后，突然开始layout不断变化，pacman还没动就改变了。

## 问题诊断过程

### 初步假设（错误）
一开始怀疑是：
1. ❌ `reset()` 时每次都重新选择layout
2. ❌ `_choose_layout()` 被频繁调用
3. ❌ `np_random` 未初始化导致layout生成异常

### 实际问题定位

通过诊断脚本发现真正的问题：

**Episode 1**: 正常运行4-6步，然后Pacman died
**Episode 2和3**: **第一步就 `terminated=True`**，导致立即返回并触发新的reset

这导致了快速的reset循环，**看起来像layout在不断刷新**！

## 根本原因

### 代码分析

**`reset()` 方法**（第199-201行）：
```python
self.step_counter = 0
self.cumulative_reward = 0
self.done = False  # ← 只重置了 done
# ❌ 没有重置 terminated 和 truncated！
```

**`step()` 方法**（第253行）：
```python
def step(self, action):
    if self.terminated or self.truncated:  # ← 检查这两个变量
        return obs, 0.0, True, False, info  # 立即返回
```

### 问题流程

1. **Episode 1结束时**：`self.terminated = True` 被设置（因为Pacman died）
2. **Episode 2调用reset()**：
   - ✅ `self.done = False` 被重置
   - ❌ `self.terminated = True` **没有被重置**
   - ❌ `self.truncated` **没有被重置**
3. **Episode 2第一步**：
   - `step()` 检查 `if self.terminated or self.truncated:`
   - 条件为 `True`（因为 `self.terminated` 仍然是 `True`）
   - **立即返回 `terminated=True`**，不执行任何游戏逻辑
4. **外层循环检测到terminated**：
   - 没有break语句，所以继续内层循环
   - 但每次 `step()` 都立即返回 `terminated=True`
   - 100步瞬间完成（都是无效步）
5. **内层循环结束，调用新的reset()**：
   - 新的layout被选择
   - **看起来像layout在不断刷新**

## 解决方案

在 `reset()` 方法中正确重置所有episode状态标志：

```python
self.step_counter = 0
self.cumulative_reward = 0
self.done = False
self.terminated = False  # ← 添加：重置terminated标志
self.truncated = False   # ← 添加：重置truncated标志
```

## 测试结果

### 修复前
```
Episode 1: 4 steps ✓
Episode 2: 1 step (terminated=True 立即返回) ❌
Episode 3: 1 step (terminated=True 立即返回) ❌
→ 看起来像layout在疯狂刷新
```

### 修复后
```
Episode 1: 6 steps ✓
Episode 2: 2 steps ✓
Episode 3: 3 steps ✓
所有episode都能正常运行，layout保持一致
```

## 对比 `pacman_env_copy.py`

让我们看看为什么 `pacman_env_copy.py` 没有这个问题：

**`pacman_env_copy.py` 中**（在 `__init__` 中初始化）：
```python
self.terminated = False
self.truncated = False
```

**并且在 `reset()` 中**：
```python
self.done = False
# terminated 和 truncated 在 step() 中是局部变量
```

**在 `step()` 中**（第205行）：
```python
done = self.game.state.isWin() or self.game.state.isLose()
terminated = done  # ← 局部变量
truncated = self.step_counter + 1 >= MAX_EP_LENGTH  # ← 局部变量
# ...
return observation, reward, terminated, truncated, info
```

`pacman_env_copy.py` 使用**局部变量**而不是实例变量来表示 `terminated` 和 `truncated`，所以不会有状态残留问题。

## 关键教训

### Gymnasium API 最佳实践

当实现 `reset()` 和 `step()` 时，需要确保：

1. **`reset()` 必须重置所有episode状态**：
   - `self.step_counter = 0`
   - `self.cumulative_reward = 0`
   - `self.done = False`
   - **`self.terminated = False`** ← 关键
   - **`self.truncated = False`** ← 关键

2. **两种设计模式**：
   - **模式A（使用实例变量）**：
     ```python
     def step(self, action):
         if self.terminated or self.truncated:  # 检查实例变量
             return ...
         # ...
         self.terminated = ...  # 更新实例变量
         self.truncated = ...
         return ..., self.terminated, self.truncated, ...
     ```
     ⚠️ **必须在 `reset()` 中重置这些实例变量**
   
   - **模式B（使用局部变量）**：
     ```python
     def step(self, action):
         # 不检查实例变量
         # ...
         terminated = ...  # 局部变量
         truncated = ...   # 局部变量
         return ..., terminated, truncated, ...
     ```
     ✓ 不需要在 `reset()` 中重置

3. **一致性检查**：
   - 如果 `step()` 开头检查 `self.terminated`，则 `reset()` 必须重置它
   - 如果使用实例变量，所有地方都应该使用实例变量
   - 如果使用局部变量，不要在 `step()` 开头检查实例变量

## 修改的文件

- `pacman_env/envs/pacman_env.py` - 在 `reset()` 中添加 `self.terminated = False` 和 `self.truncated = False`

## 相关问题

如果遇到以下现象，可能是相同的根本原因：
- ✓ Episode第一步就terminated
- ✓ Layout看起来在疯狂刷新
- ✓ Reset循环非常快
- ✓ 无法正常训练（因为episode太短）
- ✓ 第一个episode正常，后续episode异常

检查：
```python
# 在 reset() 后立即检查
obs, info = env.reset()
print(f"terminated: {env.unwrapped.terminated}")  # 应该是 False
print(f"truncated: {env.unwrapped.truncated}")    # 应该是 False
```
