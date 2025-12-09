# my_team.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import random
import contest.util as util
from contest.capture_agents import CaptureAgent
from contest.game import Directions
from contest.util import nearest_point
from collections import deque

#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='MyOffensiveAgent', second='MyDefensiveAgent', num_training=0):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --red_opts and --blue_opts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """
    return [eval(first)(first_index), eval(second)(second_index)]

##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that choose score-maximizing actions
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)

    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = game_state.get_legal_actions(self.index)

        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [self.evaluate(game_state, a) for a in actions]
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        food_left = len(self.get_food(game_state).as_list())

        if food_left <= 2:
            best_dist = 9999
            best_action = None
            for action in actions:
                successor = self.get_successor(game_state, action)
                pos2 = successor.get_agent_position(self.index)
                dist = self.get_maze_distance(self.start, pos2)
                if dist < best_dist:
                    best_action = action
                    best_dist = dist
            return best_action

        return random.choice(best_actions)

    def get_successor(self, game_state, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearest_point(pos):
            # Only half a grid position was covered
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def evaluate(self, game_state, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    def get_features(self, game_state, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        features['successor_score'] = self.get_score(successor)
        return features

    def get_weights(self, game_state, action):
        """
        Normally, weights do not depend on the game state.  They can be either
        a counter or a dictionary.
        """
        return {'successor_score': 1.0}

class OffensiveReflexAgent(ReflexCaptureAgent):
    """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        food_list = self.get_food(successor).as_list()
        features['successor_score'] = -len(food_list)  # self.getScore(successor)

        # Compute distance to the nearest food

        if len(food_list) > 0:  # This should always be True,  but better safe than sorry
            my_pos = successor.get_agent_state(self.index).get_position()
            min_distance = min([self.get_maze_distance(my_pos, food) for food in food_list])
            features['distance_to_food'] = min_distance
        return features

    def get_weights(self, game_state, action):
        return {'successor_score': 100, 'distance_to_food': -1}

class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that keeps its side Pacman-free. Again,
    this is to give you an idea of what a defensive agent
    could be like.  It is not the best or only way to make
    such an agent.
    """

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)

        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # Computes whether we're on defense (1) or offense (0)
        features['on_defense'] = 1
        if my_state.is_pacman: features['on_defense'] = 0

        # Computes distance to invaders we can see
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            features['invader_distance'] = min(dists)

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        return features

    def get_weights(self, game_state, action):
        return {'num_invaders': -1000, 'on_defense': 100, 'invader_distance': -10, 'stop': -100, 'reverse': -2}

###########################################################################################################

class MyOffensiveAgent(ReflexCaptureAgent):

    def register_initial_state(self, game_state):
        super().register_initial_state(game_state)
        self.layout = game_state.data.layout
        self.walls = self.layout.walls
        self.width = self.layout.width
        self.height = self.layout.height
        self.dead_ends = self.compute_dead_ends()
        self.food_clusters = self.compute_food_clusters(game_state)
        self.last_positions = deque(maxlen=4)
        self.escape_cooldown = 0
        self.carry_limit = 5
        self.danger_threshold = 2
        self.chase_scared_threshold = 5
        self.start = game_state.get_agent_position(self.index)

    def compute_dead_ends(self):
        width, height = self.width, self.height

        def neighbors(pos):
            x, y = pos
            result = []
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < width and 0 <= ny < height and not self.walls[nx][ny]:
                    result.append((nx, ny))
            return result

        degree = {}
        for x in range(width):
            for y in range(height):
                if not self.walls[x][y]:
                    degree[(x, y)] = len(neighbors((x, y)))

        dead = set()
        depths = {}
        q = util.Queue()
        for pos, d in degree.items():
            if d <= 1:
                dead.add(pos)
                depths[pos] = 1
                q.push(pos)

        while not q.is_empty():
            cx, cy = q.pop()
            current_depth = depths[(cx, cy)]
            for n in neighbors((cx, cy)):
                if n not in dead:
                    degree[n] -= 1
                    if degree[n] <= 1:
                        dead.add(n)
                        # Depth propagates +1 from parent
                        depths[n] = current_depth + 1
                        q.push(n)
        return depths

    def compute_food_clusters(self, game_state):
        food = set(self.get_food(game_state).as_list())
        clusters = {}
        visited = set()

        for fx, fy in food:
            if (fx, fy) in visited:
                continue

            cluster = []
            for x, y in food:
                if (x, y) not in visited and self.get_maze_distance((fx, fy), (x, y)) <= 2:
                    visited.add((x, y))
                    cluster.append((x, y))

            if len(cluster) >= 3:
                # centroid (float)
                cx = sum(x for (x, _) in cluster) / len(cluster)
                cy = sum(y for (_, y) in cluster) / len(cluster)

                # round to nearest integer tile
                cx = int(round(cx))
                cy = int(round(cy))

                clusters[(cx, cy)] = len(cluster)

        return clusters

    def choose_action(self, game_state):
        actions = game_state.get_legal_actions(self.index)
        my_pos = game_state.get_agent_position(self.index)
        my_state = game_state.get_agent_state(self.index)

        # Recompute clusters if respawned
        if my_pos == self.start:
            self.food_clusters = self.compute_food_clusters(game_state)

        # Opponents
        opponents = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        opp_positions = [a.get_position() for a in opponents if a.get_position()]

        # Dangerous ghosts
        dangerous = [a for a in opponents if
                     a.get_position() and not a.is_pacman and a.scared_timer <= self.danger_threshold]
        dg_dist = min([self.get_maze_distance(my_pos, a.get_position()) for a in dangerous], default=9999)

        # Escape if necessary
        runaway = False
        if self.escape_cooldown > 0:
            runaway = True
            self.escape_cooldown -= 1
        elif my_state.is_pacman and dg_dist <= 2:
            runaway = True
            self.escape_cooldown = 4

        if runaway or my_state.num_carrying >= self.carry_limit:
            return self.go_home_safely(game_state, actions)

        values = []
        for a in actions:
            succ = self.get_successor(game_state, a)
            succ_pos = succ.get_agent_position(self.index)
            val = self.evaluate(game_state, a)
            values.append(val)

        max_val = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_val]

        chosen = random.choice(best_actions)
        self.last_positions.append(my_pos)
        return chosen

    def go_home_safely(self, game_state, actions):
        best = None
        best_dist = 999999
        my_pos = game_state.get_agent_position(self.index)
        for a in actions:
            succ = self.get_successor(game_state, a)
            pos = succ.get_agent_position(self.index)
            if pos in self.dead_ends:
                continue
            d = self.get_maze_distance(pos, self.start)
            if d < best_dist:
                best_dist = d
                best = a
        if not best:
            best = random.choice(actions)
        self.last_positions.append(my_pos)
        return best

    def get_features(self, game_state, action):
        features = util.Counter()
        succ = self.get_successor(game_state, action)
        my_pos = succ.get_agent_position(self.index)
        my_state = succ.get_agent_state(self.index)

        # Food
        food = self.get_food(succ).as_list()
        features['successor_score'] = -len(food)
        if food:
            features['dist_food'] = min(self.get_maze_distance(my_pos, f) for f in food)

        # Capsules
        caps = self.get_capsules(succ)
        if caps:
            features['dist_capsule'] = min(self.get_maze_distance(my_pos, c) for c in caps)

        # Anti-oscillation
        if my_pos in self.last_positions:
            features['oscillation_penalty'] = 1

        # Dead-end avoidance
        x, y = my_pos
        if ((not self.red and x < self.width // 2) or (self.red and x >= self.width // 2)) and my_pos in self.dead_ends:
            prev_pos = game_state.get_agent_position(self.index)
            if prev_pos not in self.dead_ends or self.dead_ends[prev_pos] < self.dead_ends[my_pos]:
                features['dead_end_penalty'] = self.dead_ends[my_pos]

        # STOP only if forced
        if action == 'Stop':
            features['stop_penalty'] = 1

        # Food clusters
        if self.food_clusters:
            features['food_cluster_dist'] = min(self.get_maze_distance(my_pos, c) for c in self.food_clusters)

        # Opponents
        opponents = [succ.get_agent_state(i) for i in self.get_opponents(succ)]
        for opp in opponents:
            if opp.get_position():
                dist = self.get_maze_distance(my_pos, opp.get_position())
                if not opp.is_pacman:
                    if opp.scared_timer <= self.danger_threshold:
                        features['dangerous_ghost_dist'] = dist
                    elif opp.scared_timer >= self.chase_scared_threshold:
                        features['chase_scared'] = 1.0 / max(1, dist)
        return features

    def get_weights(self, game_state, action):
        return {
            'successor_score': 110,
            'dist_food': -4,
            'dist_capsule': -8,
            'dead_end_penalty': -20,
            'food_cluster_dist': -2,
            'dangerous_ghost_dist': 100,
            'chase_scared': 40,
            'oscillation_penalty': -100,
            'stop_penalty': -200,
        }

class MyDefensiveAgent(ReflexCaptureAgent):

    def register_initial_state(self, game_state):
        super().register_initial_state(game_state)
        width = game_state.data.layout.width
        self.mid_x = width // 2
        self.walls = game_state.get_walls()
        self.height = game_state.data.layout.height
        self.hover_x = self.mid_x - 3 if self.red else self.mid_x + 3  # Hover 3 tiles inside own territory
        self.food_clusters = self.compute_food_clusters(game_state)
        self.last_positions = deque(maxlen=4)

    def compute_food_clusters(self, game_state):
        food = self.get_food_you_are_defending(game_state).as_list()
        clusters = []
        for fx, fy in food:
            cluster = [(x, y) for (x, y) in food if abs(x - fx) + abs(y - fy) <= 3]
            if len(cluster) >= 4:
                clusters.append((fx, fy))
        return clusters

    def distance_to_hover(self, my_pos, game_state):
        best = float('inf')
        walls = game_state.get_walls()

        for y in range(self.height):
            if not walls[self.hover_x][y]:
                d = self.get_maze_distance(my_pos, (self.hover_x, y))
                best = min(best, d)

        return best

    def get_features(self, game_state, action):
        features = util.Counter()
        succ = self.get_successor(game_state, action)
        my_pos = succ.get_agent_position(self.index)
        my_state = succ.get_agent_state(self.index)

        features['on_defense'] = 1
        if my_state.is_pacman:
            features['on_defense'] = 0

        enemies = [succ.get_agent_state(i) for i in self.get_opponents(succ)]
        invaders = [e for e in enemies if e.is_pacman and e.get_position()]
        enemy_ghosts = [e for e in enemies if not e.is_pacman and e.get_position()]

        # Priority 1: Active invaders (enemy already crossed)
        features['num_invaders'] = len(invaders)
        if invaders:
            dists = [self.get_maze_distance(my_pos, e.get_position()) for e in invaders]
            features['invader_distance'] = min(dists)

            # Intercept bias
            ex, ey = invaders[0].get_position()
            features['intercept_bias'] = abs(ex - self.mid_x)

        # Priority 2: Threats (enemy ghosts close to crossing)
        threats = []
        for e in enemy_ghosts:
            ex, ey = e.get_position()
            # Maze distance to any mid-boundary point
            dist_to_mid = min(
                self.get_maze_distance((ex, ey),
                                       (min(max(self.mid_x - 1 if self.red else self.mid_x + 1, self.height - 1), 1),
                                        y))
                for y in range(1, self.height - 1)
            )
            if dist_to_mid <= 4:  # threat threshold
                threats.append((e, dist_to_mid))

        if threats and not invaders:
            closest_threat, threat_dmid = min(threats, key=lambda t: t[1])
            threat_pos = closest_threat.get_position()
            features['threat_distance'] = self.get_maze_distance(my_pos, threat_pos)

            # Don't overcommit far past hover line
            if abs(my_pos[0] - self.hover_x) > 4:
                features['threat_penalty'] = 1

        # Scared avoidance
        my_scared = (my_state.scared_timer > 2)
        if my_scared and invaders:
            closest_inv = min(invaders, key=lambda e: self.get_maze_distance(my_pos, e.get_position()))
            dist = self.get_maze_distance(my_pos, closest_inv.get_position())
            features['avoid_when_scared'] = max(0, 5 - dist)

        # Opportunistic offense
        scared_enemy_ghosts = [e for e in enemy_ghosts if e.scared_timer > 5]
        if scared_enemy_ghosts:
            if abs(my_pos[0] - self.mid_x) <= 2:
                features['opportunistic_offense'] = 1

        # Hover mechanics
        center_y = self.height // 2
        features['hover_distance'] = self.distance_to_hover(my_pos, game_state)
        features['vertical_centering'] = abs(my_pos[1] - center_y)

        if self.food_clusters:
            features['cluster_distance'] = min(
                self.get_maze_distance(my_pos, c) for c in self.food_clusters
            )

        if my_pos in self.last_positions:
            features['oscillation_penalty'] = 1

        if action == Directions.STOP:
            features['stop_penalty'] = 1

        reverse = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == reverse:
            features['reverse_penalty'] = 1

        self.last_positions.append(my_pos)
        return features

    def get_weights(self, game_state, action):
        return {
            'on_defense': 50,
            'num_invaders': -10000,
            'invader_distance': -60,
            'intercept_bias': -20,
            'threat_distance': -40,
            'threat_penalty': -15,
            'avoid_when_scared': 80,
            'opportunistic_offense': 50,
            'hover_distance': -25,
            'vertical_centering': -10,
            'cluster_distance': -4,
            'oscillation_penalty': -150,
            'stop_penalty': -100,
            'reverse_penalty': -3,
        }
