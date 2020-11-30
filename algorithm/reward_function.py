import numpy as np


def reward_function(old, new, basic_reward):
    """
    calculate reward
    :param old: state old
    :param new: state now
    :param basic_reward: number of steps
    :return: float
    reward
    """
    # parameters
    c = 2
    a = 10
    distance_change = old["relative_position"] - new["relative_position"]
    # basic reward
    distance_reward = np.sqrt(distance_change[0] ** 2 + distance_change[1] ** 2) / c
    if sum(distance_change) < 0:
        distance_reward = - distance_reward
    basic_reward = basic_reward * a
    return [distance_reward, basic_reward]
