from gym_minigrid.minigrid import MiniGridEnv, Grid, Key, Ball, Box
from enum import IntEnum
import random
from utils import to_numpy, object_map, agent_dir, to_one_hot
import numpy as np
import itertools
from gym_minigrid.window import Window


class SearchEnv(MiniGridEnv):
    """
    Distributional shift environment.
    """

    # Enumeration of possible actions
    class Actions(IntEnum):
        # Done completing task
        done = 0
        # Turn left, turn right, move forward
        left = 1
        right = 2
        forward = 3

    def __init__(self, width=100, height=100, agent_view_size=5, goals=20,
                 max_step=None):
        self.width = width
        self.height = height
        self.goals = goals
        if max_step is None:
            max_step = 2 * (width + height)
        self.faults = set()
        self.faults_count = 0
        self.history = []
        self.agent_start_pos = None
        self.agent_start_dir = None
        self.memory = None
        self.view_size = agent_view_size
        super().__init__(width=width, height=height, max_steps=max_step,
                         agent_view_size=agent_view_size,
                         see_through_walls=False)
        # Action enumeration for this environment
        self.reset()
        self.actions = self.Actions

    def reset(self):
        self.agent_start_pos = np.array([random.randint(1, self.width - 2),
                                         random.randint(1, self.height - 2)])
        self.agent_start_dir = random.randint(0, 3)
        self.history = []
        self.faults = set()
        self.faults_count = 0
        self.memory = np.zeros([self.width, self.height])
        super(SearchEnv, self).reset()

    def reward(self):
        find_keys = (self.faults_count - len(self.faults)) / self.faults_count
        travel_area = self.memory.sum() / (self.width * self.height)
        return find_keys, travel_area

    def step(self, action, battery_cost=1):
        self.step_count += 1
        self.agent_battery -= battery_cost
        done = False
        # # # Move
        # Get the position in front of the agent
        fwd_pos = self.front_pos
        # Get the contents of the cell in front of the agent
        fwd_cell = self.grid.get(*fwd_pos)
        # Rotate left
        if action == self.actions.left:
            self.agent_dir -= 1
            if self.agent_dir < 0:
                self.agent_dir += 4
        # Rotate right
        elif action == self.actions.right:
            self.agent_dir = (self.agent_dir + 1) % 4
        # Move forward
        elif action == self.actions.forward:
            if fwd_cell is None or fwd_cell.can_overlap():
                self.agent_pos = fwd_pos
            elif fwd_cell is not None and fwd_cell.type == 'lava':
                done = True
        elif action == self.actions.done:
            done = True
        # save history pod
        self.memory[self.agent_pos[0]][self.agent_pos[1]] = 1
        self.history.append(self.agent_pos)
        if self.grid.get(*self.agent_pos) is not None:
            if self.grid.get(*self.agent_pos).type == 'key':
                self.agent_battery = self.full_agent_battery
            if self.grid.get(*self.agent_pos).type == 'ball':
                if tuple(self.agent_pos) in self.faults:
                    self.faults.remove(tuple(self.agent_pos))
        # check done
        if self.step_count >= self.max_steps or self.agent_battery == 0:
            done = True
        return self.state(), done

    def check_history(self):
        cur = self.history[-1]
        loc = -2
        same = 0
        while len(self.history) > 1 and ~loc < len(self.history) and np.equal(
                self.history[loc], cur).all:
            same += 1
            loc -= 1
        return same

    def get_memory(self, tf):
        if tf:
            attitude = self.agent_dir
        else:
            attitude = '*'
        allow = ["wall", "key", ">", "<", "^", "V", "*"]
        # allow = list(object_map.keys())
        agent = np.array(list(self.agent_pos) + [attitude])
        whole_map = to_one_hot(to_numpy(self.grid, allow, agent).T,
                               len(object_map))
        memory = np.expand_dims(self.memory.T, -1)
        return np.concatenate([whole_map, memory], axis=-1)

    def get_view(self, tf):
        view, vis = self.gen_obs_grid()
        allow = list(object_map.keys())
        if tf:
            agent = [self.view_size - 1, int(self.view_size / 2), 3]
            view = to_numpy(view, allow, agent, vis)

        else:
            view = to_numpy(view, allow, None, vis)
        return to_one_hot(view, len(object_map))

    def state(self, tf=True):

        return {
            "agent_view": self.get_view(tf),
            "whole_map": self.get_memory(tf),
            "attitude": self.agent_dir,
            "reward": self.reward()
        }

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)
        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)
        # Place goals square in the bottom-right corner
        # random
        _width = [random.randint(1, 18) for i in
                  range(random.randint(0, int(20 // 2)))]
        _width = [[(i, j) for j in range(0, 20)] for i in _width]
        _height = [random.randint(1, 18 - 1) for i in
                   range(random.randint(0, int(20 // 2)))]
        _height = [[(j, i) for j in range(1, 18)] for i in _height]
        _points = []
        for i in _width:
            start = random.randint(1, 18)
            if random.randint(0, 1) > 0:
                end = random.randint(start, 18)
            else:
                end = 18
            _points.append(i[start:end])
        for i in _height:
            start = random.randint(1, 18)
            if random.randint(0, 1) > 0:
                end = random.randint(start, 18)
            else:
                end = 18
            _points.append(i[start:end])
        _points = list(itertools.chain.from_iterable(_points))
        for i in _points:
            if random.randint(0, 10) > 8:
                self.put_obj(Box(color="green"), *i)
                self.faults.add(tuple(i))
            else:
                self.put_obj(Ball(color="blue"), *i)

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()
        self.put_obj(Key(), *self.agent_pos)
        self.faults_count = len(self.faults)
        self.mission = "go to ball as much as possible"


if __name__ == '__main__':
    env = SearchEnv(20, 20, max_step=200, goals=50, agent_view_size=7)
    env.reset()
    env.render('human')


    def key_handler(self, event):
        if event.key == 'left':
            self.step(0)
            return
        if event.key == 'right':
            self.step(1)
            return
        if event.key == 'up':
            self.step(2)
            return


    window = Window('GYM_MiniGrid')
    window.reg_key_handler(key_handler)
    window.show(True)
    env.step(3)
    env.step(3)
    env.step(3)
    env.step(3)
    env.step(3)
    env.render('human')
    env.step(1)
    env.render('human')
    env.step(3)
    env.step(3)
    env.step(3)
    env.render('human')
    _map = env.get_memory(True)
    _memory = env.memory.T