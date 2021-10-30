from gym_minigrid.envs.simple2D import Simple2D
from gym_minigrid.minigrid import Box
import random
import numpy as np


class Simple2Dv1(Simple2D):
    def __init__(self, width=100, height=100, agent_view=7, roads=1,
                 max_step=None, fault_rate=0.3, tf=True):
        super().__init__(width, height, agent_view, roads, max_step,
                         fault_rate, tf)

    def _gen_grid(self, width, height):
        roads = self._gent_basic_grid(width, height)
        # add faults
        np.random.shuffle(roads)
        pos = 100
        faults = set()
        for i in roads:
            if pos >= 50:
                self.put_obj(Box(color="green"), *i)
                faults.add(tuple(i))
                pos -= 50
            else:
                pos += random.randint(5, 10)

    def _reward(self):
        raise NotImplementedError

    def _l_reward(self):
        raise NotImplementedError

    def _check_finish(self):
        raise NotImplementedError

    def _build_rewards(self):
        raise NotImplementedError
