def reward_function(old, new, basic_reward, step_count, same_position, done):
    """
    calculate reward
    :param old: state old
    :param new: state now
    :param basic_reward: basic reward of finish come from env
    :param step_count: number of steps
    :param same_position: number of steps in same position
    :param done: the game is finish or not
    :return: float
    reward
    """
    # parameters
    b = 10
    v1 = 0.0001
    v2 = 0.0001
    # the reward from environment
    basic_reward = basic_reward * b
    # stay over
    step_discount = - v1 * same_position - v2 * step_count
    return [step_discount, basic_reward]
