import os
import sys
import gc
import ray
import yaml
import gymnasium
import numpy as np
import torch.nn as nn
from gymnasium import spaces
from mpu.ml import indices2one_hot
from typing import Dict, Tuple, Union
from environments.MutilRoadWithTrodEnv import RouteWithTrodEnv
from environments.MutilRoadEnv import RouteEnv
from environments.AddBatteryWrapper import AddBatteryWrapper
from environments.AddEmptyWrapper import AddEmptyWrapper
from environments.AddRewardRenderWrapper import AddRewardRenderWrapper
from environments.HiddenTrodWrapper import HiddenTrodWrapper
from environments.SmallNegWrapper import SmallNegativeWrapper
from environments.DistanceBouns import CloserWrapper
from minigrid.wrappers import RGBImgObsWrapper, ImgObsWrapper
from gymnasium.wrappers import ResizeObservation, TimeLimit

agent_dir = {
    0: '>',
    1: 'V',
    2: '<',
    3: '^',
}


def display_feature_map_info(model_conv, input_size: tuple):
    """
    Display the size of the feature maps and the receptive field after each layer.

    Parameters:
    - model: The neural network model containing the layers.
    - input_size: The size of the input image in (C, H, W) format.

    Note: This function assumes the model's convolutional layers are wrapped in nn.Sequential.
    """
    current_size = input_size
    receptive_field = 1
    stride_accumulate = 1  # Accumulated stride

    for layer in model_conv:
        if isinstance(layer, nn.Conv2d):
            kernel_size = layer.kernel_size[0]
            stride = layer.stride[0]
            padding = layer.padding[0]
            current_size = (
                current_size[0],
                (current_size[1] + 2 * padding - kernel_size) // stride + 1,
                (current_size[2] + 2 * padding - kernel_size) // stride + 1,
            )

            receptive_field += (kernel_size - 1) * stride_accumulate
            stride_accumulate *= stride

        elif isinstance(layer, nn.AdaptiveMaxPool2d):
            current_size = (current_size[0],) + layer.output_size

        print(f"After layer {layer}: Feature map size = {current_size}, Receptive field = {receptive_field}")


def minigrid_env_creator(env_config):
    if env_config["id"] == "RouteWithTrod":
        env = RouteWithTrodEnv(**env_config)
        env = SmallNegativeWrapper(env)
        env = CloserWrapper(env)
        env = RGBImgObsWrapper(env, tile_size=env_config["tile_size"])
        env = ImgObsWrapper(env)
        env = HiddenTrodWrapper(env)
        env = AddRewardRenderWrapper(env)
        env = ResizeObservation(env, (env_config["img_size"], env_config["img_size"]))
        env = AddBatteryWrapper(env)
        env = TimeLimit(env, max_episode_steps=env_config["max_steps"])
    elif env_config["id"] == "Route":
        env = RouteEnv(**env_config)
        env = SmallNegativeWrapper(env)
        env = CloserWrapper(env)
        env = RGBImgObsWrapper(env, tile_size=env_config["tile_size"])
        env = ImgObsWrapper(env)
        env = HiddenTrodWrapper(env)
        env = AddRewardRenderWrapper(env)
        env = ResizeObservation(env, (env_config["img_size"], env_config["img_size"]))
        env = AddBatteryWrapper(env)
        env = TimeLimit(env, max_episode_steps=env_config["max_steps"])
    else:
        env = gymnasium.make(env_config["id"], render_mode="rgb_array")
        env = RGBImgObsWrapper(env, tile_size=env_config["tile_size"])
        env = ImgObsWrapper(env)
        env = ResizeObservation(env, (env_config["img_size"], env_config["img_size"]))
        env = AddEmptyWrapper(env)
        env = TimeLimit(env, max_episode_steps=env_config["max_steps"])

    return env


def to_one_hot(array, classes):
    shape = list(array.shape) + [-1]
    array = array.flatten()
    return np.array(indices2one_hot(array, nb_classes=classes + 1)).reshape(shape)


def translate_state(state):
    return state["agent_view"], state["whole_map"], state["battery"]


def copy_params(offline, online):
    layer = list(offline.collect_params().values())
    for i in layer:
        _1 = online.collect_params().get(
            "_".join(i.name.split("_")[1:])).data().asnumpy()
        online.collect_params().get("_".join(i.name.split("_")[1:])).set_data(
            i.data())
        _2 = online.collect_params().get(
            "_".join(i.name.split("_")[1:])).data().asnumpy()


def check_dir(i):
    # create required path
    if not os.path.exists("./{}/".format(i)):
        os.mkdir("./{}/".format(i))


def get_goal(array, agent):
    _min = 999
    _location = np.zeros([array.shape[0], array.shape[1]])
    for i, row in enumerate(array):
        for j, value in enumerate(row):
            if value in (3, 4):
                _dis = sum(np.abs(np.array(agent[:2]) - np.array((i, j))))
                if _dis < _min:
                    _location = np.zeros([array.shape[0], array.shape[1]])
                    _location[i][j] = 1
                    _min = _dis
    return _location


def to_numpy(grid, allow, agent, vis_mask=None):
    """
    Produce a pretty string of the environment.txt's grid along with the agent.
    A grid cell is represented by 2-character string, the first one for
    the object and the second one for the color.
    """
    shape = (grid.width, grid.height)
    grid = grid.grid
    if vis_mask is None:
        vis_mask = np.ones(len(grid), dtype=bool)
    else:
        vis_mask = vis_mask.flatten()
    map_img = []
    for i, j in zip(grid, vis_mask):
        if i is not None and i.type in allow.keys() and j:
            map_img.append(allow[i.type])
        else:
            map_img.append(0)
    map_img = np.array(map_img).reshape(shape)
    if agent is not None:
        map_img[agent[0], agent[1]] = allow[agent_dir[agent[2]]]
    return map_img


def create_input(data):
    target = []
    for i in range(len(data[0])):
        target.append([])
    for arg in data:
        _ = [np.array([i]) if type(i) is int else np.array(i)
             for i in arg]
        for i, each in enumerate(_):
            target[i].append(each)
    target = [np.array(i) for i in target]
    return target


def get_size(obj):
    marked = {id(obj)}
    obj_q = [obj]
    size = 0

    while obj_q:
        size += sum(sys.getsizeof(i) for i in obj_q)
        all_refr = ((id(o), o) for o in gc.get_referents(*obj_q))
        new_refr = {o_id: o for o_id, o in all_refr if o_id not in marked and not isinstance(o, type)}
        obj_q = new_refr.values()
        marked.update(new_refr.keys())

    return size


def init_ray(ray_setting=None):
    if ray_setting is not None:
        with open(ray_setting, 'r') as file:
            settings = yaml.safe_load(file)
        ray.init(**settings)
    else:
        ray.init("auto")


def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)


def convert_np_arrays(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_np_arrays(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_np_arrays(item) for item in obj]
    else:
        return obj


def flatten_dict(d):
    flat_dict = {}
    for key, value in d.items():
        if isinstance(value, dict):
            for subkey, subvalue in value.items():
                flat_dict[subkey] = subvalue
        else:
            flat_dict[key] = value
    return flat_dict


# This function is copied from:
# https://github.com/DLR-RM/stable-baselines3/
def get_obs_shape(
        observation_space: spaces.Space,
) -> Union[Tuple[int, ...], Dict[str, Tuple[int, ...]]]:
    """
    Get the shape of the observation (useful for the buffers).

    :param observation_space:
    :return:
    """
    if isinstance(observation_space, spaces.Box):
        return observation_space.shape
    elif isinstance(observation_space, spaces.Discrete):
        # Observation is an int
        return 1,
    elif isinstance(observation_space, spaces.MultiDiscrete):
        # Number of discrete features
        return int(len(observation_space.nvec)),
    elif isinstance(observation_space, spaces.MultiBinary):
        # Number of binary features
        return int(observation_space.n),
    elif isinstance(observation_space, spaces.Dict):
        return {key: get_obs_shape(subspace) for (key, subspace) in observation_space.spaces.items()}


# This function is copied from:
# https://github.com/DLR-RM/stable-baselines3/
def get_action_dim(action_space: spaces.Space) -> int:
    """
    Get the dimension of the action space.

    :param action_space:
    :return:
    """
    if isinstance(action_space, spaces.Box):
        return int(np.prod(action_space.shape))
    elif isinstance(action_space, spaces.Discrete):
        # Action is an int
        return 1
    elif isinstance(action_space, spaces.MultiDiscrete):
        # Number of discrete actions
        return int(len(action_space.nvec))
    elif isinstance(action_space, spaces.MultiBinary):
        # Number of binary actions
        return int(action_space.n)
    else:
        raise NotImplementedError(f"{action_space} action space is not supported")
