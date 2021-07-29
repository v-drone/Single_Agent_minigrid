from gym_minigrid.minigrid import MiniGridEnv
from enum import IntEnum
import random
from utils import to_numpy, agent_dir, to_one_hot, get_goal
import numpy as np


# Detect
class Detect(Exception):
    def __init__(self):
        pass


class SearchEnv(MiniGridEnv):
    """
    Distributional shift environment.
    """

    # Enumeration of possible actions
    class Actions(IntEnum):
        # Turn left, turn right, move forward
        forward = 0
        left = 1
        right = 2

    def __init__(self, tf=True, width=100, height=100, agent_view=7,
                 max_step=None):
        self.tf = tf
        self.width = width
        self.height = height
        if max_step is None:
            max_step = 4 * (width + height)
        self.agent_start_pos = None
        self.agent_start_dir = None
        self.memory = np.zeros([self.width, self.height])
        self.roadmap = np.zeros([self.width, self.height])
        self.history = []
        self.view_pos = [agent_view - 1, int(agent_view / 2), 3]
        self.reward_map = np.zeros(shape=(width, height))
        super().__init__(width=width, height=height, max_steps=max_step,
                         agent_view_size=agent_view,
                         see_through_walls=False)
        # Action enumeration for this environment
        self.reset()
        self.actions = self.Actions
        self.hit = False
        self.to_goal = 999

    def reset(self):
        self.agent_start_pos = np.array([random.randint(1, self.width - 2),
                                         random.randint(1, self.height - 2)])
        self.agent_start_dir = random.randint(0, 3)
        self.memory = np.zeros([self.width, self.height])
        self.roadmap = np.zeros([self.width, self.height])
        super(SearchEnv, self).reset()

    def reward(self):
        memory = self.memory.flatten()
        roadmap = self.roadmap.flatten()
        n_road = np.greater_equal(roadmap, 1).sum()
        n_faults = np.equal(roadmap, 1).sum()
        d_road = 0
        d_faults = 0
        for i, j in zip(memory, roadmap):
            if i == 1 and j == 1:
                d_faults += 1
                d_road += 1
            elif i == 1 and j == 2:
                d_road += 1
        # cover rate, road cover rate, faults cover rate
        return [memory.sum() / len(memory), d_road / n_road,
                d_faults / n_faults]

    def step(self, action, battery_cost=1):
        self.hit = False
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
            elif fwd_cell is not None and fwd_cell.type in ('lava', "wall"):
                self.hit = True
        # save history pod
        if self.grid.get(*self.agent_pos) is not None and self.grid.get(
                *self.agent_pos).type == 'box':
            if self.memory[self.agent_pos[0]][self.agent_pos[1]] < 5:
                self.memory[self.agent_pos[0]][self.agent_pos[1]] += 1
        else:
            self.memory[self.agent_pos[0]][self.agent_pos[1]] = 1
        self.history.append(self.agent_pos)
        # check done
        if self.step_count >= self.max_steps or self.agent_battery == 0:
            done = True
        return self.state(tf=self.tf), done

    def on_road(self):
        if self.grid.get(*self.agent_pos) is not None and self.grid.get(
                *self.agent_pos).type in ("box", "ball"):
            return True
        else:
            return False

    def check_history(self):
        cur = self.history[-1]
        same = 0
        for i in reversed(self.history[:-1]):
            if np.equal(cur, i).all():
                same += 1
            else:
                break
        return same

    def get_whole_map(self):
        allow = ["wall", "key", "ball", "box", ">", "<", "^", "V"]
        allow = {k: v + 1 for v, k in enumerate(allow)}
        agent = np.array(
            [self.agent_pos[1], self.agent_pos[0], self.agent_dir])
        whole_map = to_numpy(self.grid, allow, agent)
        whole_map = np.where(whole_map == allow["box"], allow["ball"],
                             whole_map)
        goal = get_goal(whole_map, agent)
        # whole_map = to_one_hot(whole_map, len(allow))
        # whole_map = np.transpose(whole_map, [2, 0, 1])
        whole_map = np.expand_dims(whole_map, 0)
        return np.concatenate([whole_map, np.expand_dims(self.memory.T, 0),
                               np.expand_dims(goal, 0)], axis=0)

    def get_view(self, tf):
        view, vis = self.gen_obs_grid()
        allow = ["wall", "key", "ball", "box", "^"]
        allow = {k: v + 1 for v, k in enumerate(allow)}
        if tf:
            tf = [self.agent_view_size - 1, int(self.agent_view_size / 2), 3]
        else:
            tf = None
        view = to_numpy(view, allow, tf, vis)
        goal = get_goal(view, tf)
        # view = to_one_hot(view, len(allow))
        # view = np.transpose(view, (2, 0, 1))
        view = np.expand_dims(view, 0)
        return np.concatenate([np.expand_dims(goal, 0), view], axis=0)

    def state(self, tf=True):
        whole_map = self.get_whole_map()
        view = self.get_view(tf)
        data = {
            "whole_map": whole_map,
            "agent_view": view,
            "battery": self. agent_battery,
            "reward": self.reward_map[self.agent_pos[0]][self.agent_pos[1]],
        }
        return data

    def _gen_grid(self, width, height):
        raise NotImplementedError
