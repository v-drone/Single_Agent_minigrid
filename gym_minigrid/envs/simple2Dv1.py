from gym_minigrid.envs.simple2D import Simple2D
from gym_minigrid.minigrid import Grid, Key, Ball, Box
import random
import numpy as np


class Simple2Dv1(Simple2D):
    def __init__(self, width=100, height=100, agent_view=7, roads=1,
                 max_step=None, road_rate=0.3, tf=True):
        self.roads = roads

        self.road_rate = int(road_rate * min([width, height]))
        super().__init__(tf, width, height, agent_view, max_step)

    def _gen_grid(self, width, height):
        roads = self.gent_basic_grid(width, height)
        # add faults
        np.random.shuffle(roads)
        pos = 100
        faults = set()
        for i in roads:
            if pos >= 50:
                self.roadmap[i[0]][i[1]] = 1
                self.put_obj(Box(color="green"), *i)
                faults.add(tuple(i))
                pos -= 50
            else:
                self.roadmap[i[0]][i[1]] = 2
                self.put_obj(Ball(color="blue"), *i)
                pos += random.randint(5, 10)

    def set_reward_map(self):
        pass
