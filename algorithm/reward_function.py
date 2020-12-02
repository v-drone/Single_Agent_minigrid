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
    c = 1
    a = 50
    v = 0.001
    distance_change = old["relative_position"] - new["relative_position"]
    # basic reward
    distance_reward = 0
    basic_reward = basic_reward * a
    step_discount = - v * same_position - v * step_count
    if distance_reward == 0:
        return [distance_reward, basic_reward]
    else:
        return [step_discount, basic_reward]
