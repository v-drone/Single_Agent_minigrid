from gym_minigrid.minigrid import MiniGridEnv, Grid, Goal
from gym_minigrid.super_lava import Lava
import random


class EmptyEnv(MiniGridEnv):
    """
    Distributional shift environment.
    """

    def __init__(
            self,
            width=9,
            height=6,
            max_step=None,
    ):
        if max_step is None:
            max_step = 2 * width * height
        if random.randint(0, 1) != 0:
            self.agent_start_pos = (1, random.randint(1, height - 2))
            self.agent_start_dir = 0
        else:
            self.agent_start_pos = (width - 2, random.randint(1, height - 2))
            self.agent_start_dir = 1
        self.goal_pos = (random.randint(1, width - 2), random.randint(1, height - 2))
        while self.goal_pos == self.agent_start_pos:
            self.goal_pos = (random.randint(1, width - 2), random.randint(1, height - 2))
        super().__init__(
            width=width,
            height=height,
            max_steps=max_step,
            # Set this to True for maximum speed
            see_through_walls=True
        )

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place a goal square in the bottom-right corner
        self.put_obj(Goal(), *self.goal_pos)

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = "get to the green goal square"
