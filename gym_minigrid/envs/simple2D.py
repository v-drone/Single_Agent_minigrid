from gym_minigrid.envs.search import SearchEnv
from gym_minigrid.minigrid import Grid, Key, Ball, Box
import random
import numpy as np


class Simple2D(SearchEnv):
    def __init__(self, width=100, height=100, agent_view=7, roads=1,
                 max_step=None, road_rate=0.3, tf=True):
        self.roads = roads
        self.road_rate = int(road_rate * min([width, height]))
        super().__init__(tf, width, height, agent_view, max_step)

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)
        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)
        # random create road
        roads = []
        for i in range(self.roads):
            choice = random.randint(0, 4)
            start = random.randint(1, self.road_rate)
            if choice == 0:
                #
                _width = random.randint(2, width - 2)
                for j in range(start, width - 1):
                    roads.append((_width, j))
            elif choice == 1:
                _width = random.randint(2, width - 2)
                for j in range(width - start - 1, 0, -1):
                    roads.append((_width, j))
            elif choice == 2:
                _he = random.randint(2, height - 2)
                for j in range(start, height - 1):
                    roads.append((j, _he))
            else:
                _he = random.randint(2, height - 2)
                for j in range(height - start - 1, 0, -1):
                    roads.append((j, _he))
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
        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()
        self.put_obj(Key(), *self.agent_pos)
        self.mission = "go to ball as much as possible"
