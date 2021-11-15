from gym_minigrid.minigrid import MiniGridEnv
from enum import IntEnum
from utils import to_numpy, agent_dir
import numpy as np
import random


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
        self.history = []
        super().__init__(width=width, height=height, max_steps=max_step, agent_view_size=agent_view,
                         see_through_walls=False)
        # Action enumeration for this environment
        self.reset()
        self.actions = self.Actions
        self.to_goal = 999

    def reset(self):
        self.agent_start_pos = np.array([random.randint(1, self.width - 2), random.randint(1, self.height - 2)])
        self.agent_start_dir = random.randint(0, 3)
        self.memory = np.zeros([self.width, self.height])
        self.history = []
        super(SearchEnv, self).reset()

    def state(self, tf=True):
        finish = self._check_finish()
        data = {
            "whole_map": self._get_whole_map(),
            "agent_view": self._get_view(tf),
            "battery": self.battery,
            "reward": self._reward(),
            "history": self._get_history(),
            "finish": finish
        }
        if finish:
            data["l_reward"] = self._l_reward()
        else:
            data["l_reward"] = None
        return data

    def build_memory(self):
        self.memory += 1
        self.memory[self.agent_pos[0]][self.agent_pos[1]] = 0

    def step(self, action, battery_cost=1):
        self.step_count += 1
        done = False
        self.battery -= battery_cost
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
        # save history
        self.build_memory()
        self.history.append(tuple([self.agent_pos[0], self.agent_pos[1]]))
        # check done
        if self._check_finish():
            done = True
        return self.state(tf=self.tf), done

    def check_history(self):
        cur = self.history[-1]
        same = 0
        for i in reversed(self.history[:-1]):
            if np.equal(cur, i).all():
                same += 1
            else:
                break
        return same

    def _get_whole_map(self):
        allow = ["wall", "key", "ball", ">", "<", "^", "V"]
        allow = {k: v + 1 for v, k in enumerate(allow)}
        whole_map = to_numpy(self.grid, allow, None)
        memory = self.memory.T
        memory = np.where(memory > 0, 1, memory)
        memory[self.agent_pos[1]][self.agent_pos[0]] = self.agent_dir + 2
        whole_map = np.expand_dims(whole_map, 0)
        memory = np.expand_dims(memory, 0)
        return np.concatenate([whole_map, memory], axis=0)

    def _get_view(self, tf):
        allow = ["wall", "key", "ball", "box", ">", "<", "^", "V"]
        allow = {k: v + 1 for v, k in enumerate(allow)}
        view, vis = self.gen_obs_grid()
        view = to_numpy(view, allow, None, vis)
        view = np.expand_dims(view, 0)
        if tf:
            agent = np.zeros_like(view[0])
            agent[self.agent_view_size - 1][int(self.agent_view_size / 2)] = allow[agent_dir[3]]
            agent = np.expand_dims(agent, 0)
            view = np.concatenate([view, agent], axis=0)
        return view

    def _get_history(self):
        history = np.zeros(shape=(self.max_steps, 2))
        for i, j in enumerate(history):
            if len(self.history) > i:
                history[i] = self.history[i]
        return history

    def _gen_grid(self, width, height):
        raise NotImplementedError

    def _reward(self):
        raise NotImplementedError

    def _l_reward(self):
        raise NotImplementedError

    def _check_finish(self):
        raise NotImplementedError
