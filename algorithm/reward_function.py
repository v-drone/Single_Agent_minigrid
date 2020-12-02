import numpy as np


def reward_function(old, new, basic_reward, step_count):
    """
    calculate reward
    :param old: state old
    :param new: state now
    :param basic_reward: basic reward of finish come from env
    :param step_count: number of steps
    :return: float
    reward
    """
    # parameters
    c = 0.5
    a = 1
    v = 0.005
    distance_change = old["relative_position"] - new["relative_position"]
    # basic reward
    distance_reward = np.sqrt(distance_change[0] ** 2 + distance_change[1] ** 2) / c
    if sum(distance_change) < 0:
        distance_reward = - distance_reward
    basic_reward = basic_reward * a
    step_discount = step_count * v
    if distance_reward > 0:
        return [distance_reward, basic_reward]
    else:
        return [- step_discount, basic_reward]
