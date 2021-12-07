import random
from gym_minigrid.envs.simple2D import Simple2D
from gym_minigrid.minigrid import Ball, Box


class Simple2Dv2(Simple2D):

    def __init__(self, width=100, height=100, agent_view=5, roads=1, max_step=None, fault_rate=0.3, tf=True):
        super().__init__(width, height, agent_view, roads, max_step, fault_rate, tf)

    def _extrinsic_reward(self):
        roads = set()
        walkways = set()
        for i in self.grid.grid:
            if i is not None and i.type == "ball":
                roads.add(i.cur_pos)
            elif i is not None and i.type == "box":
                walkways.add(i.cur_pos)
        roads_arrival = 0
        history = set(self.history)
        for i in roads:
            if i in history:
                roads_arrival += 1
        walkway_arrival = 0
        for i in walkways:
            if i in history:
                walkway_arrival += 1
        # walkway_arrival / len(walkways)
        return roads_arrival / len(roads), roads_arrival / len(roads)

    def _gen_grid(self, width, height):
        roads = self._gent_basic_grid(width, height)
        # add roads
        for i in roads:
            self.put_obj(Ball(color="blue"), *i)
        # add roads not in the map
        # walkway = []
        # start = random.randint(1, self.fault_rate)
        # boundary = random.random()
        # if list(roads)[0][0] != list(roads)[1][0]:
        #     if boundary >= 0.5:
        #         _width = random.randint(3, width - 3)
        #         for j in range(start, width - 1):
        #             walkway.append((_width, j))
        #     else:
        #         _width = random.randint(3, width - 3)
        #         for j in range(width - start - 1, 0, -1):
        #             walkway.append((_width, j))
        # else:
        #     if boundary >= 0.5:
        #         _height = random.randint(3, height - 3)
        #         for j in range(start, height - 1):
        #             walkway.append((j, _height))
        #     else:
        #         _height = random.randint(3, height - 3)
        #         for j in range(height - start - 1, 0, -1):
        #             walkway.append((j, _height))
        # for i in walkway:
        #     if i not in roads:
        #         self.put_obj(Box(color="green"), *i)
