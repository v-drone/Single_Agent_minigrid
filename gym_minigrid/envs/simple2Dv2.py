import random
import itertools
import numpy as np
from gym_minigrid.envs.simple2D import Simple2D
from gym_minigrid.minigrid import Ball, Box


class Simple2Dv2(Simple2D):

    def __init__(self, width=100, height=100, agent_view=5, roads=1, max_step=None, fault_rate=0.3, tf=True):
        super().__init__(width, height, agent_view, roads, max_step, fault_rate, tf)

    def _build_rewards(self):
        rewards = []
        roads = set()
        for i in self.grid.grid:
            if i is not None and i.type == "ball":
                rewards.append(0)
                roads.add(i.cur_pos)
            elif i is not None and i.type == "box" and self.memory[i.cur_pos[0]][i.cur_pos[1]] > 0:
                rewards.append(0)
                roads.add(i.cur_pos)
            else:
                rewards.append(-1)
        for i in self.gen_obs_grid()[0].grid:
            if i is not None and i.type == "box":
                roads.add(i.cur_pos)
        rewards = np.array(rewards).reshape(20, 20).T
        for i in list(itertools.product(*[list(range(self.width)), list(range(self.height))])):
            rewards[i[0]][i[1]] = - min([abs(j[0] - i[0]) + abs(j[1] - i[1]) for j in roads]) + rewards[i[0]][i[1]]
        for i in roads:
            rewards[i[0]][i[1]] = 0
        return rewards

    def _reward(self):
        return self._build_rewards()[self.agent_pos[0]][self.agent_pos[1]]

    def _l_reward(self):
        roads = set()
        walkways = set()
        for i in self.grid.grid:
            if i is not None and i.type == "ball":
                roads.add(i.cur_pos)
            elif i is not None and i.type == "box":
                walkways.add(i.cur_pos)
        roads_arrival = 0
        for i in roads:
            if self.memory[i[0]][i[1]] > 0:
                roads_arrival += 1
        walkway_arrival = 0
        for i in walkways:
            if self.memory[i[0]][i[1]] > 0:
                walkway_arrival += 1
        return roads_arrival / len(roads), walkway_arrival, len(walkways)

    def _check_finish(self):
        if self.step_count >= self.max_steps or self.battery == 0:
            return -1
        elif self._l_reward()[0] == 1:
            return 1
        else:
            return 0

    def _gen_grid(self, width, height):
        roads = self._gent_basic_grid(width, height)
        # add roads
        for i in roads:
            self.put_obj(Ball(color="blue"), *i)
        # add roads not in the map
        walkway = []
        boundary = random.randint(0, 4)
        start = random.randint(1, self.fault_rate)
        if boundary == 0:
            _width = random.randint(3, width - 3)
            for j in range(start, width - 1):
                walkway.append((_width, j))
        elif boundary == 1:
            _width = random.randint(3, width - 3)
            for j in range(width - start - 1, 0, -1):
                walkway.append((_width, j))
        elif boundary == 2:
            _he = random.randint(3, height - 3)
            for j in range(start, height - 1):
                walkway.append((j, _he))
        else:
            _he = random.randint(3, height - 3)
            for j in range(height - start - 1, 0, -1):
                walkway.append((j, _he))
        for i in walkway:
            if i not in roads:
                self.put_obj(Box(color="green"), *i)
