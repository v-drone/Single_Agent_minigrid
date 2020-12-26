import numpy as np


def reward_function(old, new, basic_reward, step_count, same_position):
    """
    calculate reward
    :param old: state old
    :param new: state now
    :param basic_reward: basic reward of finish come from env
    :param step_count: number of steps
    :param same_position: number of steps in same position
    :return: float
    reward
    """
    # parameters
    c = 0
    a = 1
    v1 = 0.0001
    v2 = 0.002
    # distance change
    distance_change = old["relative_position"] - new["relative_position"]
    # basic reward
    distance_reward = sum(distance_change) * c
    basic_reward = basic_reward * a
    same_position_discount = - v1 * np.power(2, same_position)
    step_discount = - v2 * step_count
    return sum([distance_reward, basic_reward, same_position_discount, step_discount])
