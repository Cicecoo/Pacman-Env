# featureExtractors.py
# --------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"Feature extractors for Pacman game states"

from pacman_env.envs.ucb_pacman.game import Directions, Actions
import pacman_env.envs.ucb_pacman.util as util

class FeatureExtractor:
    def getFeatures(self, state, action):
        """
          Returns a dict from features to counts
          Usually, the count will just be 1.0 for
          indicator functions.
        """
        util.raiseNotDefined()

class IdentityExtractor(FeatureExtractor):
    def getFeatures(self, state, action):
        feats = util.Counter()
        feats[(state,action)] = 1.0
        return feats

class CoordinateExtractor(FeatureExtractor):
    def getFeatures(self, state, action):
        feats = util.Counter()
        feats[state] = 1.0
        feats['x=%d' % state[0]] = 1.0
        feats['y=%d' % state[0]] = 1.0
        feats['action=%s' % action] = 1.0
        return feats

def closestFood(pos, food, walls):
    """
    closestFood -- this is similar to the function that we have
    worked on in the search project; here its all in one place
    """
    fringe = [(pos[0], pos[1], 0)]
    expanded = set()
    while fringe:
        pos_x, pos_y, dist = fringe.pop(0)
        if (pos_x, pos_y) in expanded:
            continue
        expanded.add((pos_x, pos_y))
        # if we find a food at this location then exit
        if food[pos_x][pos_y]:
            return dist
        # otherwise spread out from the location to its neighbours
        nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
        for nbr_x, nbr_y in nbrs:
            fringe.append((nbr_x, nbr_y, dist+1))
    # no food found
    return None

class SimpleExtractor(FeatureExtractor):
    """
    Returns simple features for a basic reflex Pacman:
    - whether food will be eaten
    - how far away the next food is
    - whether a ghost collision is imminent
    - whether a ghost is one step away
    """

    def get_features(self, state, action):
        # extract the grid of food and wall locations and get the ghost locations
        food = state.getFood()
        walls = state.getWalls()
        ghosts = state.getGhostPositions()

        features = util.Counter()

        features["bias"] = 1.0

        # compute the location of pacman after he takes the action
        x, y = state.getPacmanPosition()
        dx, dy = Actions.directionToVector(action)
        next_x, next_y = int(x + dx), int(y + dy)

        # count the number of ghosts 1-step away
        features["#-of-ghosts-1-step-away"] = sum((next_x, next_y) in Actions.getLegalNeighbors(g, walls) for g in ghosts)

        # if there is no danger of ghosts then add the food feature
        if not features["#-of-ghosts-1-step-away"] and food[next_x][next_y]:
            features["eats-food"] = 1.0

        dist = closestFood((next_x, next_y), food, walls)
        if dist is not None:
            # make the distance a number less than one otherwise the update
            # will diverge wildly
            features["closest-food"] = float(dist) / (walls.width * walls.height)
        features.divideAll(10.0)
        return features


def closestCapsule(pos, capsules, walls):
    """
    Find the Manhattan distance to the closest capsule.
    """
    if not capsules:
        return None
    
    # Calculate Manhattan distance to all capsules
    distances = [abs(pos[0] - cap[0]) + abs(pos[1] - cap[1]) for cap in capsules]
    return min(distances)


def closestScaredGhost(pos, scared_ghosts, walls):
    """
    Find the Manhattan distance to the closest scared ghost.
    """
    if not scared_ghosts:
        return None
    
    # Calculate Manhattan distance to all scared ghosts
    distances = [abs(pos[0] - g[0]) + abs(pos[1] - g[1]) for g in scared_ghosts]
    return min(distances)


class EnhancedSimpleExtractor(FeatureExtractor):
    """
    Enhanced feature extractor that includes capsule and scared ghost mechanics:
    - Capsule位置和距离
    - Scared ghost的状态和位置
    - 区分普通ghost和scared ghost的威胁/机会
    - Scared timer剩余时间
    
    这样agent可以学习：
    1. 在危险时寻找capsule
    2. 吃capsule后主动追击scared ghost获取高分
    3. 利用scared状态期间安全收集食物
    """

    def get_features(self, state, action):
        # Extract game state information
        food = state.getFood()
        walls = state.getWalls()
        capsules = state.getCapsules()
        ghost_states = state.getGhostStates()
        
        features = util.Counter()
        features["bias"] = 1.0

        # Compute next position after action
        x, y = state.getPacmanPosition()
        dx, dy = Actions.directionToVector(action)
        next_x, next_y = int(x + dx), int(y + dy)
        next_pos = (next_x, next_y)

        # Separate normal ghosts and scared ghosts
        normal_ghosts = []
        scared_ghosts = []
        max_scared_timer = 0
        
        for ghost in ghost_states:
            ghost_pos = ghost.getPosition()
            ghost_x, ghost_y = int(ghost_pos[0]), int(ghost_pos[1])
            
            if ghost.scaredTimer > 0:
                scared_ghosts.append((ghost_x, ghost_y))
                max_scared_timer = max(max_scared_timer, ghost.scaredTimer)
            else:
                normal_ghosts.append((ghost_x, ghost_y))

        # Feature 1: Normal ghost danger (negative reward expected)
        features["#-of-normal-ghosts-1-step-away"] = sum(
            next_pos in Actions.getLegalNeighbors(g, walls) for g in normal_ghosts
        )

        # Feature 2: Scared ghost opportunity (positive reward expected)
        features["#-of-scared-ghosts-1-step-away"] = sum(
            next_pos in Actions.getLegalNeighbors(g, walls) for g in scared_ghosts
        )

        # Feature 3: Eating scared ghost (high positive reward expected)
        # Check if next position collides with a scared ghost
        features["eats-scared-ghost"] = 1.0 if next_pos in scared_ghosts else 0.0

        # Feature 4: Distance to closest scared ghost (when available)
        if scared_ghosts:
            dist = closestScaredGhost(next_pos, scared_ghosts, walls)
            if dist is not None:
                features["closest-scared-ghost"] = float(dist) / (walls.width * walls.height)
        
        # Feature 5: Scared timer remaining (normalized)
        # 让agent知道还有多少时间可以追击scared ghost
        if max_scared_timer > 0:
            features["scared-timer"] = float(max_scared_timer) / 40.0  # SCARED_TIME = 40

        # Feature 6: Eating capsule (triggers scared mode)
        features["eats-capsule"] = 1.0 if next_pos in capsules else 0.0

        # Feature 7: Distance to closest capsule
        if capsules:
            dist = closestCapsule(next_pos, capsules, walls)
            if dist is not None:
                features["closest-capsule"] = float(dist) / (walls.width * walls.height)

        # Feature 8: Food features
        # Only eat food when safe (no normal ghosts nearby) or when ghosts are scared
        is_safe = (features["#-of-normal-ghosts-1-step-away"] == 0 or 
                   max_scared_timer > 0)
        
        if is_safe and food[next_x][next_y]:
            features["eats-food"] = 1.0

        # Feature 9: Distance to closest food
        dist = closestFood(next_pos, food, walls)
        if dist is not None:
            features["closest-food"] = float(dist) / (walls.width * walls.height)

        # Normalize all features
        features.divideAll(10.0)
        return features


