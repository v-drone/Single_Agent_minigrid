from gym_minigrid.envs.simple2D import Simple2D
from gym_minigrid.minigrid import Ball, Box
import random


class Simple2Dv2(Simple2D):
    def __init__(self, width=100, height=100, agent_view=7, roads=1,
                 max_step=None, road_rate=0.3, tf=True):
        self.roads = roads
        self.road_rate = int(road_rate * min([width, height]))
        super().__init__(width, height, agent_view, roads, max_step, road_rate,
                         tf)

    def _gen_grid(self, width, height):
        roads = self.gent_basic_grid(width, height)
        # add roads
        for i in roads:
            self.roadmap[i[0]][i[1]] = 2
            self.put_obj(Ball(color="blue"), *i)
        # add roads not in the map
        unavailable = []
        boundary = random.randint(0, 4)
        start = random.randint(1, self.road_rate)
        if boundary == 0:
            _width = random.randint(3, width - 3)
            for j in range(start, width - 1):
                unavailable.append((_width, j))
        elif boundary == 1:
            _width = random.randint(3, width - 3)
            for j in range(width - start - 1, 0, -1):
                unavailable.append((_width, j))
        elif boundary == 2:
            _he = random.randint(3, height - 3)
            for j in range(start, height - 1):
                unavailable.append((j, _he))
        else:
            _he = random.randint(3, height - 3)
            for j in range(height - start - 1, 0, -1):
                unavailable.append((j, _he))
        for i in unavailable:
            if i not  in roads:
                self.roadmap[i[0]][i[1]] = 2
                self.put_obj(Box(color="green"), *i)

    def set_reward_map(self):
        pass
