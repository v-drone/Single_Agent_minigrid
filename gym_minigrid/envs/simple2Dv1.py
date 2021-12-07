from gym_minigrid.envs.simple2D import Simple2D
from gym_minigrid.minigrid import Ball


class Simple2Dv1(Simple2D):
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
        return roads_arrival / len(roads), roads_arrival / len(roads)

    def _gen_grid(self, width, height):
        roads = self._gent_basic_grid(width, height)
        # add roads
        for i in roads:
            self.put_obj(Ball(color="blue"), *i)
