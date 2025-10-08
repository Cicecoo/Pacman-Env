"""
特征编码器集合 - 为DQN Agent设计不同的状态表示方案

包含多种特征编码方案：
1. DistanceDirectionEncoder - 基于距离和方向的轻量特征
2. EightDirectionEncoder - 八方向扫描特征（类似MC Agent）
3. CompactEncoder - 紧凑型特征（最小维度）
4. OriginalEncoder - 原始的固定尺寸编码器
"""

import numpy as np


class DistanceDirectionEncoder:
    """
    基于距离和方向的特征编码器
    
    核心思想：
    - 不直接使用坐标，而是使用相对距离和方向
    - 使用最近的K个对象，避免固定槽位的浪费
    - 添加更多语义特征（如是否被包围、逃生方向等）
    
    特征维度：约50-80维（比461维小很多）
    """
    
    def __init__(self, 
                 top_k_foods=5,
                 top_k_ghosts=5,
                 max_capsules=4):
        self.top_k_foods = top_k_foods
        self.top_k_ghosts = top_k_ghosts
        self.max_capsules = max_capsules
        self.feature_dim = self._calculate_feature_dim()
    
    def _calculate_feature_dim(self):
        """计算特征维度"""
        dim = 0
        
        # 1. Agent基础信息：方向(4 one-hot)
        dim += 4
        
        # 2. Ghost信息：每个最近的Ghost
        #    - 曼哈顿距离(1) + x方向(1) + y方向(1) + scared(1) = 4
        dim += self.top_k_ghosts * 4
        
        # 3. 食物信息：每个最近的食物
        #    - 曼哈顿距离(1) + x方向(1) + y方向(1) = 3
        dim += self.top_k_foods * 3
        
        # 4. Capsule信息：每个capsule
        #    - 曼哈顿距离(1) + x方向(1) + y方向(1) = 3
        dim += self.max_capsules * 3
        
        # 5. 统计信息
        dim += 6  # 总食物数、总capsule数、scared ghost数、
                  # 最近食物距离、最近ghost距离、最近scared ghost距离
        
        # 6. 危险度信息（四个方向的危险度）
        dim += 4
        
        return dim
    
    def encode(self, obs_dict):
        """编码观测为特征向量"""
        features = []
        
        # Agent位置
        agent_pos = np.array(obs_dict["agent"])
        agent_dir = obs_dict["agent_direction"]
        
        # 1. Agent方向（one-hot编码）
        direction_onehot = [0.0] * 4
        direction_onehot[agent_dir] = 1.0
        features.extend(direction_onehot)
        
        # 2. Ghost信息（按距离排序，取最近的K个）
        ghosts = obs_dict["ghosts"]
        ghost_scared = obs_dict["ghost_scared"]
        
        # 过滤有效ghost（坐标不为0）
        valid_ghosts = []
        for i, ghost_pos in enumerate(ghosts):
            if not (ghost_pos[0] == 0 and ghost_pos[1] == 0):
                valid_ghosts.append({
                    'pos': ghost_pos,
                    'scared': ghost_scared[i]
                })
        
        # 按曼哈顿距离排序
        if len(valid_ghosts) > 0:
            for ghost in valid_ghosts:
                ghost['distance'] = abs(ghost['pos'][0] - agent_pos[0]) + \
                                  abs(ghost['pos'][1] - agent_pos[1])
            valid_ghosts.sort(key=lambda g: g['distance'])
        
        # 提取最近的K个ghost特征
        for i in range(self.top_k_ghosts):
            if i < len(valid_ghosts):
                ghost = valid_ghosts[i]
                dist = ghost['distance']
                dx = ghost['pos'][0] - agent_pos[0]
                dy = ghost['pos'][1] - agent_pos[1]
                scared = float(ghost['scared'])
                
                # 归一化距离（假设地图最大尺寸约30）
                features.extend([
                    dist / 30.0,
                    np.clip(dx / 30.0, -1, 1),
                    np.clip(dy / 30.0, -1, 1),
                    scared
                ])
            else:
                features.extend([0.0, 0.0, 0.0, 0.0])  # 填充
        
        # 3. 食物信息（按距离排序，取最近的K个）
        food_grid = obs_dict["food"]
        food_positions = np.argwhere(food_grid > 0)
        
        if len(food_positions) > 0:
            # 计算距离并排序
            distances = np.abs(food_positions[:, 0] - agent_pos[0]) + \
                       np.abs(food_positions[:, 1] - agent_pos[1])
            sorted_indices = np.argsort(distances)[:self.top_k_foods]
            nearest_foods = food_positions[sorted_indices]
            nearest_distances = distances[sorted_indices]
            
            for i in range(self.top_k_foods):
                if i < len(nearest_foods):
                    dist = nearest_distances[i]
                    dx = nearest_foods[i][0] - agent_pos[0]
                    dy = nearest_foods[i][1] - agent_pos[1]
                    
                    features.extend([
                        dist / 30.0,
                        np.clip(dx / 30.0, -1, 1),
                        np.clip(dy / 30.0, -1, 1)
                    ])
                else:
                    features.extend([0.0, 0.0, 0.0])
        else:
            features.extend([0.0, 0.0, 0.0] * self.top_k_foods)
        
        # 4. Capsule信息
        capsules = obs_dict["capsules"]
        valid_capsules = [cap for cap in capsules if not (cap[0] == 0 and cap[1] == 0)]
        
        # 按距离排序
        if len(valid_capsules) > 0:
            capsule_data = []
            for cap in valid_capsules:
                dist = abs(cap[0] - agent_pos[0]) + abs(cap[1] - agent_pos[1])
                capsule_data.append({
                    'pos': cap,
                    'distance': dist
                })
            capsule_data.sort(key=lambda c: c['distance'])
            
            for i in range(self.max_capsules):
                if i < len(capsule_data):
                    cap = capsule_data[i]
                    dist = cap['distance']
                    dx = cap['pos'][0] - agent_pos[0]
                    dy = cap['pos'][1] - agent_pos[1]
                    
                    features.extend([
                        dist / 30.0,
                        np.clip(dx / 30.0, -1, 1),
                        np.clip(dy / 30.0, -1, 1)
                    ])
                else:
                    features.extend([0.0, 0.0, 0.0])
        else:
            features.extend([0.0, 0.0, 0.0] * self.max_capsules)
        
        # 5. 统计信息
        total_food = len(food_positions)
        total_capsules = len(valid_capsules)
        scared_ghost_count = sum([1 for g in valid_ghosts if g['scared'] > 0])
        
        min_food_dist = min([g['distance'] for g in 
                            [{'distance': d} for d in distances[:1]]], 
                           default=30) if len(food_positions) > 0 else 30
        min_ghost_dist = min([g['distance'] for g in valid_ghosts], 
                            default=30) if len(valid_ghosts) > 0 else 30
        min_scared_dist = min([g['distance'] for g in valid_ghosts if g['scared'] > 0], 
                             default=30) if scared_ghost_count > 0 else 30
        
        features.extend([
            total_food / 100.0,
            total_capsules / 10.0,
            scared_ghost_count / 5.0,
            min_food_dist / 30.0,
            min_ghost_dist / 30.0,
            min_scared_dist / 30.0
        ])
        
        # 6. 危险度评估（四个方向）
        # North, South, East, West
        walls_grid = obs_dict["walls"]
        danger_scores = self._calculate_danger_scores(
            agent_pos, valid_ghosts, walls_grid
        )
        features.extend(danger_scores)
        
        return np.array(features, dtype=np.float32)
    
    def _calculate_danger_scores(self, agent_pos, ghosts, walls_grid):
        """计算四个方向的危险度"""
        directions = [
            (0, 1),   # North
            (0, -1),  # South
            (1, 0),   # East
            (-1, 0)   # West
        ]
        
        danger_scores = []
        map_height, map_width = walls_grid.shape
        
        for dx, dy in directions:
            danger = 0.0
            
            # 检查该方向是否有墙
            new_x = int(agent_pos[0] + dx)
            new_y = int(agent_pos[1] + dy)
            
            if (new_x < 0 or new_x >= map_height or 
                new_y < 0 or new_y >= map_width or 
                walls_grid[new_x, new_y] > 0):
                danger = 1.0  # 墙壁=最高危险
            else:
                # 检查该方向是否有危险的ghost
                for ghost in ghosts:
                    if ghost['scared'] == 0:  # 只考虑非scared的ghost
                        ghost_dx = ghost['pos'][0] - new_x
                        ghost_dy = ghost['pos'][1] - new_y
                        ghost_dist = abs(ghost_dx) + abs(ghost_dy)
                        
                        if ghost_dist <= 3:  # 3步内的ghost被认为是危险的
                            danger = max(danger, 1.0 - ghost_dist / 3.0)
            
            danger_scores.append(danger)
        
        return danger_scores


class EightDirectionEncoder:
    """
    八方向扫描特征编码器（类似MC Agent的特征）
    
    在8个方向上扫描：
    - 到墙的距离
    - 到最近食物的距离
    - 到最近ghost的距离
    - ghost是否scared
    - 到最近capsule的距离
    
    特征维度：8方向 × 5特征 + 额外统计 = 40-50维
    """
    
    def __init__(self, max_scan_distance=10):
        self.max_scan_distance = max_scan_distance
        self.directions = [
            (0, 1),    # North
            (1, 1),    # NE
            (1, 0),    # East
            (1, -1),   # SE
            (0, -1),   # South
            (-1, -1),  # SW
            (-1, 0),   # West
            (-1, 1)    # NW
        ]
        self.feature_dim = self._calculate_feature_dim()
    
    def _calculate_feature_dim(self):
        """计算特征维度"""
        # 8个方向 × 5个特征（墙、食物、ghost、scared、capsule）
        # + 4个方向one-hot + 6个统计信息
        return 8 * 5 + 4 + 6
    
    def encode(self, obs_dict):
        """编码观测为特征向量"""
        features = []
        
        agent_pos = obs_dict["agent"]
        agent_dir = obs_dict["agent_direction"]
        walls_grid = obs_dict["walls"]
        food_grid = obs_dict["food"]
        ghosts = obs_dict["ghosts"]
        ghost_scared = obs_dict["ghost_scared"]
        capsules = obs_dict["capsules"]
        
        # 1. Agent方向（one-hot）
        direction_onehot = [0.0] * 4
        direction_onehot[agent_dir] = 1.0
        features.extend(direction_onehot)
        
        # 2. 八方向扫描
        for dx, dy in self.directions:
            # 在这个方向上扫描
            scan_result = self._scan_direction(
                agent_pos, (dx, dy), 
                walls_grid, food_grid, ghosts, ghost_scared, capsules
            )
            features.extend(scan_result)
        
        # 3. 统计信息
        food_positions = np.argwhere(food_grid > 0)
        valid_capsules = [c for c in capsules if not (c[0] == 0 and c[1] == 0)]
        valid_ghosts = [(i, g) for i, g in enumerate(ghosts) 
                       if not (g[0] == 0 and g[1] == 0)]
        
        total_food = len(food_positions)
        total_capsules = len(valid_capsules)
        scared_count = sum([ghost_scared[i] for i, _ in valid_ghosts])
        
        # 计算最小距离
        min_food_dist = 30
        if len(food_positions) > 0:
            distances = np.abs(food_positions[:, 0] - agent_pos[0]) + \
                       np.abs(food_positions[:, 1] - agent_pos[1])
            min_food_dist = np.min(distances)
        
        min_ghost_dist = 30
        min_scared_dist = 30
        for i, g in valid_ghosts:
            dist = abs(g[0] - agent_pos[0]) + abs(g[1] - agent_pos[1])
            min_ghost_dist = min(min_ghost_dist, dist)
            if ghost_scared[i] > 0:
                min_scared_dist = min(min_scared_dist, dist)
        
        features.extend([
            total_food / 100.0,
            total_capsules / 10.0,
            scared_count / 5.0,
            min_food_dist / 30.0,
            min_ghost_dist / 30.0,
            min_scared_dist / 30.0
        ])
        
        return np.array(features, dtype=np.float32)
    
    def _scan_direction(self, agent_pos, direction, walls_grid, food_grid, 
                       ghosts, ghost_scared, capsules):
        """在一个方向上扫描特征"""
        dx, dy = direction
        
        # 初始化特征
        wall_dist = self.max_scan_distance
        food_dist = self.max_scan_distance
        ghost_dist = self.max_scan_distance
        ghost_is_scared = 0.0
        capsule_dist = self.max_scan_distance
        
        map_height, map_width = walls_grid.shape
        
        # 沿方向扫描
        for step in range(1, self.max_scan_distance + 1):
            x = int(agent_pos[0] + dx * step)
            y = int(agent_pos[1] + dy * step)
            
            # 检查边界
            if x < 0 or x >= map_height or y < 0 or y >= map_width:
                wall_dist = step
                break
            
            # 检查墙
            if walls_grid[x, y] > 0:
                wall_dist = step
                break
            
            # 检查食物（只记录第一个）
            if food_dist == self.max_scan_distance and food_grid[x, y] > 0:
                food_dist = step
            
            # 检查capsule
            for cap in capsules:
                if cap[0] == x and cap[1] == y:
                    if capsule_dist == self.max_scan_distance:
                        capsule_dist = step
                    break
            
            # 检查ghost
            for i, ghost in enumerate(ghosts):
                if abs(ghost[0] - x) < 0.5 and abs(ghost[1] - y) < 0.5:
                    if ghost_dist == self.max_scan_distance:
                        ghost_dist = step
                        ghost_is_scared = float(ghost_scared[i])
                    break
        
        # 归一化
        return [
            wall_dist / self.max_scan_distance,
            food_dist / self.max_scan_distance,
            ghost_dist / self.max_scan_distance,
            ghost_is_scared,
            capsule_dist / self.max_scan_distance
        ]


class CompactEncoder:
    """
    紧凑型特征编码器 - 最小化特征维度
    
    只使用最关键的信息：
    - 最近的1个食物
    - 最近的2个ghost
    - 最近的1个capsule
    - 危险度评估
    
    特征维度：约20-30维
    """
    
    def __init__(self):
        self.feature_dim = 22  # 固定维度: 4+3+8+3+4=22
    
    def encode(self, obs_dict):
        """编码观测为紧凑特征向量"""
        features = []
        
        agent_pos = np.array(obs_dict["agent"])
        agent_dir = obs_dict["agent_direction"]
        
        # 1. Agent方向 (4维 one-hot)
        direction_onehot = [0.0] * 4
        direction_onehot[agent_dir] = 1.0
        features.extend(direction_onehot)
        
        # 2. 最近的食物 (3维: 距离, dx, dy)
        food_grid = obs_dict["food"]
        food_positions = np.argwhere(food_grid > 0)
        
        if len(food_positions) > 0:
            distances = np.linalg.norm(food_positions - agent_pos, axis=1)
            nearest_idx = np.argmin(distances)
            nearest_food = food_positions[nearest_idx]
            dist = distances[nearest_idx]
            dx = nearest_food[0] - agent_pos[0]
            dy = nearest_food[1] - agent_pos[1]
            features.extend([dist / 30.0, np.clip(dx / 30.0, -1, 1), 
                           np.clip(dy / 30.0, -1, 1)])
        else:
            features.extend([1.0, 0.0, 0.0])
        
        # 3. 最近的2个ghost (每个4维: 距离, dx, dy, scared)
        ghosts = obs_dict["ghosts"]
        ghost_scared = obs_dict["ghost_scared"]
        valid_ghosts = []
        
        for i, ghost_pos in enumerate(ghosts):
            if not (ghost_pos[0] == 0 and ghost_pos[1] == 0):
                dist = abs(ghost_pos[0] - agent_pos[0]) + \
                       abs(ghost_pos[1] - agent_pos[1])
                valid_ghosts.append({
                    'pos': ghost_pos,
                    'distance': dist,
                    'scared': ghost_scared[i]
                })
        
        valid_ghosts.sort(key=lambda g: g['distance'])
        
        for i in range(2):
            if i < len(valid_ghosts):
                ghost = valid_ghosts[i]
                dx = ghost['pos'][0] - agent_pos[0]
                dy = ghost['pos'][1] - agent_pos[1]
                features.extend([
                    ghost['distance'] / 30.0,
                    np.clip(dx / 30.0, -1, 1),
                    np.clip(dy / 30.0, -1, 1),
                    float(ghost['scared'])
                ])
            else:
                features.extend([1.0, 0.0, 0.0, 0.0])
        
        # 4. 最近的capsule (3维: 距离, dx, dy)
        capsules = obs_dict["capsules"]
        valid_capsules = [c for c in capsules if not (c[0] == 0 and c[1] == 0)]
        
        if len(valid_capsules) > 0:
            capsule_data = [(abs(c[0] - agent_pos[0]) + abs(c[1] - agent_pos[1]), c) 
                           for c in valid_capsules]
            capsule_data.sort(key=lambda x: x[0])
            dist, cap = capsule_data[0]
            dx = cap[0] - agent_pos[0]
            dy = cap[1] - agent_pos[1]
            features.extend([dist / 30.0, np.clip(dx / 30.0, -1, 1), 
                           np.clip(dy / 30.0, -1, 1)])
        else:
            features.extend([1.0, 0.0, 0.0])
        
        # 5. 统计信息 (4维)
        total_food = len(food_positions)
        total_capsules = len(valid_capsules)
        scared_count = len([g for g in valid_ghosts if g['scared'] > 0])
        danger_score = self._calculate_danger(agent_pos, valid_ghosts)
        
        features.extend([
            total_food / 100.0,
            total_capsules / 10.0,
            scared_count / 5.0,
            danger_score
        ])
        
        return np.array(features, dtype=np.float32)
    
    def _calculate_danger(self, agent_pos, ghosts):
        """计算当前位置的危险度"""
        danger = 0.0
        for ghost in ghosts:
            if ghost['scared'] == 0 and ghost['distance'] <= 5:
                danger = max(danger, 1.0 - ghost['distance'] / 5.0)
        return danger


# 保持原始编码器用于向后兼容
class FixedSizeStateEncoder:
   
    def __init__(self, 
                 max_map_size=20,
                 max_ghosts=5,
                 max_capsules=10,
                 top_k_foods=10,
                 use_grid_encoding=True):
        self.max_map_size = max_map_size
        self.max_ghosts = max_ghosts
        self.max_capsules = max_capsules
        self.top_k_foods = top_k_foods
        self.use_grid_encoding = use_grid_encoding
        self.feature_dim = self._calculate_feature_dim()
    
    def _calculate_feature_dim(self):
        """计算固定的特征向量维度"""
        dim = 3 + self.max_ghosts * 3 + self.max_capsules * 2 + \
              self.top_k_foods * 2 + 1 + 2
        if self.use_grid_encoding:
            dim += self.max_map_size * self.max_map_size
        return dim
    
    def encode(self, obs_dict):
        """原始编码方法（与dqn_agent.py中的FixedSizeStateEncoder相同）"""
        # 这里可以复制原来的实现，或者直接从dqn_agent导入
        from agents.dqn_agent import FixedSizeStateEncoder
        encoder = FixedSizeStateEncoder(
            self.max_map_size, self.max_ghosts, 
            self.max_capsules, self.top_k_foods, 
            self.use_grid_encoding
        )
        return encoder.encode(obs_dict)
