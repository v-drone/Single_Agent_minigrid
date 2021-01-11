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
    v1 = 0
    v2 = 0
    # basic reward
    # the reward from environment
    basic_reward = basic_reward * a
    # distance to goal changed
    distance_change = old["relative_position"] - new["relative_position"]
    distance_change = sum(distance_change) * c
    # stay over
    step_discount = - v1 * same_position - v2 * step_count
    return [step_discount + distance_change, basic_reward]

