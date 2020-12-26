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
    v1 = 0.001
    v2 = 0.0002
    # distance change
    distance_change = old["relative_position"] - new["relative_position"]
    # basic reward
    distance_reward = sum(distance_change)
    basic_reward = basic_reward
    same_position_discount = - np.tanh(same_position / 5)
    step_discount = - step_count
    return sum([c * distance_reward, a * basic_reward, v1 * same_position_discount, v2 * step_discount])
