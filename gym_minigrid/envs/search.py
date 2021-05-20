from gym_minigrid.minigrid import MiniGridEnv, Grid, Key, Ball
from enum import IntEnum
import random
from utils import to_numpy
import numpy as np


def translate_state(state):
    # agent_view = get_pad(state["agent_view"])
    # whole_map = get_pad(state["whole_map"])
    agent_view = state["agent_view"]
    whole_map = state["whole_map"]
    return agent_view, whole_map


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
        self.agent_start_pos = np.array([random.randint(1, width - 2),
                                         random.randint(1, height - 2)])
        self.agent_start_dir = 0
        self.goals = goals
        if max_step is None:
            max_step = 2 * (width + height)
        self.faults = set()
        self.faults_count = 0
        self.history = []
        super().__init__(width=width, height=height, max_steps=max_step,
                         agent_view_size=agent_view_size,
                         see_through_walls=True)
        # Action enumeration for this environment
        self.actions = self.Actions
        self.memory = np.zeros([self.width, self.height])

    def reset(self):
        self.history = []
        self.faults = set()
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

    def state(self):
        attitude = self.agent_dir
        grid, vis_mask = self.gen_obs_grid()
        agent = np.array(list(self.agent_pos) + [self.agent_dir])
        view = to_numpy(grid, [3, 6, agent[-1]], vis_mask)[::-1]
        view = np.flip(view[::-1], 1)
        whole_map = to_numpy(self.grid, agent, None).T
        return {
            "agent_view": view,
            "whole_map": whole_map,
            "attitude": attitude,
            "pos": self.memory,
        }

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)
        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)
        # Place goals square in the bottom-right corner
        for i in range(random.randint(self.goals, self.goals * 2)):
            _ = (random.randint(1, self.width - 2),
                 random.randint(1, self.height - 2))
            self.put_obj(Ball(), *_)
            self.faults.add(tuple(_))
        self.put_obj(Key(), *(random.randint(1, self.width - 2),
                              random.randint(1, self.height - 2)))

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()
        self.faults_count = len(self.faults)
        self.mission = "go to ball as much as possible"
