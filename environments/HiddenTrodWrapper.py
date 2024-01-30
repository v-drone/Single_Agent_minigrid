import numpy as np
from minigrid.wrappers import ObservationWrapper
from environments.MutilRoadWithTrodEnv import RouteWithTrodEnv, TrodTile
from minigrid.core.grid import Grid


class HiddenTrodWrapper(ObservationWrapper):
    def __init__(self, env: RouteWithTrodEnv):
        super().__init__(env)

    def observation(self, obs):
        # Compute which cells are visible to the agent
        _, vis_mask = self.gen_obs_grid()

        # Compute the world coordinates of the bottom-left corner
        # of the agent's view area
        f_vec = self.dir_vec
        r_vec = self.right_vec
        top_left = (
                self.agent_pos
                + f_vec * (self.agent_view_size - 1)
                - r_vec * (self.agent_view_size // 2)
        )
        # Mask of which cells to highlight
        highlight_mask = np.zeros(shape=(self.width, self.height), dtype=bool)

        # For each cell in the visibility mask
        for vis_j in range(0, self.agent_view_size):
            for vis_i in range(0, self.agent_view_size):
                # If this cell is not visible, don't highlight it
                if not vis_mask[vis_i, vis_j]:
                    continue

                # Compute the world coordinates of this cell
                abs_i, abs_j = top_left - (f_vec * vis_j) + (r_vec * vis_i)

                if abs_i < 0 or abs_i >= self.width:
                    continue
                if abs_j < 0 or abs_j >= self.height:
                    continue

                # Mark this cell to be highlighted
                highlight_mask[abs_i, abs_j] = True

        img = obs.copy()
        # Render the grid
        for j in range(0, self.height):
            for i in range(0, self.width):
                cell = self.grid.get(i, j)
                if isinstance(cell, TrodTile) and cell.color != "purple" and not self.in_view(i, j):
                    agent_here = np.array_equal(self.agent_pos, (i, j))
                    assert highlight_mask is not None
                    tile_img = Grid.render_tile(
                        None,
                        agent_dir=self.agent_dir if agent_here else None,
                        highlight=highlight_mask[i, j],
                        tile_size=self.tile_size,
                    )
                    ymin = j * self.tile_size
                    ymax = (j + 1) * self.tile_size
                    xmin = i * self.tile_size
                    xmax = (i + 1) * self.tile_size
                    img[ymin:ymax, xmin:xmax, :] = tile_img
        return img
