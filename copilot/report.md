## 问题修复总结

### 问题1: Ghost 停止移动
**现象**: 在游戏进行过程中（未 terminated/truncated），ghost 在数个 step 后停止运动

**根本原因**: 
- 随机生成的 layout 可能包含死胡同（dead-end）
- Ghost 进入死胡同后，`Actions.getPossibleActions()` 只返回 `['Stop']`
- 原始代码中 ghost 不断执行 'Stop' 导致冻结

### 问题2: Exception: Illegal ghost action Stop
**现象**: 程序抛出 "Illegal ghost action Stop" 异常并崩溃

**根本原因**:
- 修改 `getLegalActions()` 移除 'Stop' 后，某些极端情况下可能返回空列表
- `GhostAgent.getAction()` 在分布为空时返回 `Directions.STOP`
- 但 'Stop' 不在合法动作中，导致 `applyAction()` 验证失败

### 最终修复方案

#### 1. 修改 pacman.py 的 `GhostRules.getLegalActions()`:

```python
def getLegalActions( state, ghostIndex ):
    """
    Ghosts cannot stop, and cannot turn around unless they
    reach a dead end, but can turn 90 degrees at intersections.
    """
    conf = state.getGhostState( ghostIndex ).configuration
    possibleActions = Actions.getPossibleActions( conf, state.data.layout.walls )
    
    # Remove 'Stop' action - ghosts should always move
    # But only if there are other actions available
    if 'Stop' in possibleActions and len( possibleActions ) > 1:
        possibleActions.remove( 'Stop' )
    
    # Remove reverse direction only if ghost has other options
    # If ghost is in a dead-end (only one possible direction), allow reversal
    reverse = Actions.reverseDirection( conf.direction )
    if reverse in possibleActions and len( possibleActions ) > 1:
        possibleActions.remove( reverse )
    
    return possibleActions
```

**关键点**:
- **只在有其他选择时移除 'Stop'**: 避免返回空列表
- **允许死胡同反转**: Ghost 可以在死胡同中掉头
- **顺序很重要**: 先处理 Stop，再处理反向，确保死胡同检测正确

#### 2. 修改 ghostAgents.py 的 `GhostAgent.getAction()`:

```python
def getAction( self, state ):
    dist = self.getDistribution(state)
    if len(dist) == 0:
        # Fallback: get any legal action (should not happen with proper layouts)
        legalActions = state.getLegalActions(self.index)
        if len(legalActions) > 0:
            return legalActions[0]
        # Last resort: STOP (this indicates a serious problem)
        return Directions.STOP
    else:
        return chooseFromDistribution( dist )
```

**关键点**:
- **多层后备机制**: 确保即使在异常情况下也不会崩溃
- **优先使用合法动作**: 从实际合法动作列表中选择

### 测试结果

✓ **10 个 episode，无崩溃**
✓ **Ghost 持续移动，无卡死**  
✓ **正常的游戏终止（Pacman died / victorious）**

所有问题已完全解决！

Made changes.