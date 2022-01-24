from minigrid.minigrid import MiniGridEnv
from enum import IntEnum
from gym import spaces
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
        self.history_length = 5
        self.width = width
        self.height = height
        if max_step is None:
            max_step = 4 * (width + height)
        self.agent_start_pos = [9,9]
        self.agent_start_dir = 0
        self.memory = []
        self.history = []
        self.battery_history = []
        self.to_goal = 999
        self.render_size = 10
        self.previous_reward = 0
        self.reward = 0
        super().__init__(width=width, height=height, max_steps=max_step, agent_view_size=agent_view,
                         see_through_walls=False)
        # Action enumeration for this environment
        self.actions = self.Actions
        self.action_space = spaces.Discrete(len(self.actions))


    def reset(self):
        self.agent_start_pos = np.array([random.randint(1, self.width - 2), random.randint(1, self.height - 2)])
        self.agent_start_dir = random.randint(0, 3)
        super(SearchEnv, self).reset()
        self.memory = [self._get_whole_map()] * self.history_length
        self.battery_history = [self.full_battery] * self.history_length
        self.history = []
        self.history.append(tuple([self.agent_start_pos[0], self.agent_start_pos[1]]))
        state = self.state()
        return state["data"]

    def state(self):
        finish = self._check_finish()
        data = {
            "data": self._get_whole_map(),
            "battery": self._get_battery()[-1],
            "finish": finish,
        }
        if finish:
            data["reward"] = self._reward() + sum(self._extrinsic_reward())
        else:
            data["reward"] = self._reward()
        return data

    def build_memory(self):
        self.memory.append(self._get_whole_map())
        self.memory = self.memory[-self.history_length:]
        self.battery_history.append(self.battery)
        self.battery_history = self.battery_history[-self.history_length:]

    def step(self, action, battery_cost=1):
        self.previous_reward = self.reward
        self.step_count += 1
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
        self.reward = sum(self._extrinsic_reward())
        state = self.state()
        return state["data"], state["reward"], state["finish"], state["battery"]

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
        return np.concatenate([np.expand_dims(i, 0) for i in self.battery_history[-self.history_length:]])

    def _extrinsic_reward(self):
        raise NotImplementedError

    def _gen_grid(self, width, height):
        raise NotImplementedError

    def _reward(self):
        raise NotImplementedError

    def _check_finish(self):
        raise NotImplementedError
