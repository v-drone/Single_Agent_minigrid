from gym_minigrid.minigrid import MiniGridEnv, Grid, Goal
from gym_minigrid.super_lava import Lava
import random


class LavaGapEnv(MiniGridEnv):
    """
    Environment with one wall of lava with a small gap to cross through
    This environment is similar to LavaCrossing but simpler in structure.
    """

    def __init__(self, size, obstacle_type=Lava):
        self.obstacle_type = obstacle_type
        self.size = size
        super().__init__(
            grid_size=size,
            max_steps=4 * size * size,
            # Set this to True for maximum speed
            see_through_walls=False,
            seed=None
        )

    def _gen_grid(self, width, height):
        if random.randint(0, 1) != 0:
            self.agent_pos = (1, random.randint(1, self.size - 2))
            self.goal_pos = (self.size - 2, random.randint(1, self.size - 2))
        else:
            self.agent_pos = (self.size - 2, random.randint(1, self.size - 2))
            self.goal_pos = (1, random.randint(1, self.size - 2))
        assert width >= 5 and height >= 5

        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place the agent in the top-left corner
        self.agent_dir = 0

        # Place a goal square in the bottom-right corner
        self.put_obj(Goal(), *self.goal_pos)

        # Generate and store random gap position
        # Place the obstacle wall
        samples = set(random.sample(range(2, width - 2), random.randint(1, width // 2)))
        for i in samples:
            self.grid.vert_wall(i, 1, height - 2, self.obstacle_type)
            # Put a hole in the wall
            for j in set(random.sample(range(2, height - 2), random.randint(1, height // 1.5))):
                self.grid.set(i, random.randint(1, height - 2), None)

        self.mission = (
            "avoid the lava and get to the green goal square"
            if self.obstacle_type == Lava
            else "find the opening and get to the green goal square"
        )
