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
    b = 5
    d = 0.5
    v1 = 0.001
    v2 = 0.002
    # the reward from environment
    basic_reward = basic_reward * b
    # distance to goal changed
    distance_change = old["relative_position"] - new["relative_position"]
    distance_change = sum(distance_change) * d
    # stay over
    step_discount = - v1 * same_position - v2 * step_count
    return [step_discount + distance_change, basic_reward]

