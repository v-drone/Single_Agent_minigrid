from gym_minigrid.minigrid import MiniGridEnv
from enum import IntEnum
from utils import to_numpy, agent_dir
import numpy as np
import random
from PIL import Image


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
        self.memory = []
        self.history = []
        self.battery_history = []
        # Action enumeration for this environment
        self.actions = self.Actions
        self.to_goal = 999
        self.render_size = 5

        super().__init__(width=width, height=height, max_steps=max_step, agent_view_size=agent_view,
                         see_through_walls=False)

    def reset(self):
        self.agent_start_pos = np.array([random.randint(1, self.width - 2), random.randint(1, self.height - 2)])
        self.agent_start_dir = random.randint(0, 3)
        super(SearchEnv, self).reset()
        self.memory = [self._get_whole_map()] * 10
        self.battery_history = [self.full_battery] * 10
        self.history = []
        self.history.append(tuple([self.agent_start_pos[0], self.agent_start_pos[1]]))

    def state(self):
        finish = self._check_finish()
        data = {
            # "whole_map": self._get_whole_map(),
            # "agent_view": self._get_view(),
            "memory": self._get_memory(),
            "battery": self._get_battery(),
            "reward": self._reward(),
            "history": self._get_history(),
            "finish": finish,
            "hidden": None
        }
        if finish:
            data["l_reward"] = self._extrinsic_reward()
        else:
            data["l_reward"] = None
        return data

    def build_memory(self):
        self.memory.append(self._get_whole_map())
        self.memory = self.memory[-10:]
        self.battery_history.append(self.battery)
        self.battery_history = self.battery_history[-10:]

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
        return self.state(), done

    def check_history(self):
        cur = self.history[-1]
        same = 0
        for i in reversed(self.history[:-1]):
            if np.equal(cur, i).all():
                same += 1
            else:
                break
        return same

    def get_agent_obs_locations(self):
        agent_obs = self.gen_obs_grid()[0]
        return set([i.cur_pos if i is not None else None for i in agent_obs.grid])

    def _get_whole_map(self):
        agent_obs = self.get_agent_obs_locations()
        _ = self.grid.copy()
        _.grid = [None if i is not None and i.type == "box" and i.cur_pos not in agent_obs else i for i in _.grid]
        return _.render(self.render_size, self.agent_pos, self.agent_dir)

    def _get_view(self):
        agent_obs = self.get_agent_obs_locations()
        _ = self.grid.copy()
        _.grid = [i if i is not None and (i.type == "wall" or i.cur_pos in agent_obs) else None for i in _.grid]
        return _.render(self.render_size, self.agent_pos, self.agent_dir)

    def _get_history(self):
        history = np.zeros(shape=(self.max_steps, 2))
        for i, j in enumerate(history):
            if len(self.history) > i:
                history[i] = self.history[i]
        return history

    def _get_memory(self):
        return np.concatenate([np.expand_dims(i, 0) for i in self.memory])

    def _get_battery(self):
        return np.concatenate([np.expand_dims(i, 0) for i in self.battery_history[-10:]])

    def _extrinsic_reward(self):
        raise NotImplementedError

    def _gen_grid(self, width, height):
        raise NotImplementedError

    def _reward(self):
        raise NotImplementedError

    def _check_finish(self):
        raise NotImplementedError
