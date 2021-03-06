import os
import numpy as np
from mxnet import nd


def copy_params(offline, online):
    layer = list(offline.collect_params().values())
    for i in layer:
        _1 = online.collect_params().get("_".join(i.name.split("_")[1:])).data().asnumpy()
        online.collect_params().get("_".join(i.name.split("_")[1:])).set_data(i.data())
        _2 = online.collect_params().get("_".join(i.name.split("_")[1:])).data().asnumpy()


def replace_self(grid, attitude):
    on_off = (grid == 6).astype(int)
    on_off *= attitude
    return grid + on_off


def check_dir(i):
    # create required path
    if not os.path.exists("./{}/".format(i)):
        os.mkdir("./{}/".format(i))


def to_numpy(grid, agent=None, vis_mask=None):
    """
    Produce a pretty string of the environment's grid along with the agent.
    A grid cell is represented by 2-character string, the first one for
    the object and the second one for the color.
    """
    map_img = np.zeros((grid.width, grid.height), dtype='uint8')
    if vis_mask is None:
        vis_mask = np.ones((grid.width, grid.height), dtype=bool)
    agent_dir = {
        0: '>',
        1: 'V',
        2: '<',
        3: '^'
    }
    # Map of object types to numbers
    object_map = {
        'empty': 0,
        'goal': 1,
        'wall': 2,
        'lava': 3,
        '>': 6,
        '<': 7,
        '^': 8,
        'V': 9,
    }
    for i in range(grid.width):
        for j in range(grid.height):
            if agent is not None and i == agent[0] and j == agent[1]:
                map_img[i, j] = object_map[agent_dir[agent[2]]]
            elif vis_mask[i, j]:
                v = grid.get(i, j)
                if v is None:
                    map_img[i, j] = object_map['empty']
                else:
                    map_img[i, j] = object_map[v.type]
    return map_img


def get_pad(src, size=15):
    _ = np.array([size, size]) - src.shape
    first_half = np.array(_ / 2).astype(int)
    second_half = _ - first_half
    return np.pad(src, ((first_half[0], second_half[0]), (first_half[1], second_half[1])), mode="constant")


def translate_state(state):
    # agent_view = get_pad(state["agent_view"])
    # whole_map = get_pad(state["whole_map"])
    agent_view = state["agent_view"]
    whole_map = state["whole_map"]
    # whole_map.flatten(),
    return np.concatenate([agent_view.flatten(), state["relative_position"], [state["attitude"]]])
