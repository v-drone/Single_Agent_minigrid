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
    b = 1
    v1 = 0.005
    v2 = 0.01
    # the reward from environment
    basic_reward = basic_reward * b
    # stay over
    same_position = - v1 * same_position
    # step used discount
    if done:
        step_discount = - v2 * step_count
    else:
        step_discount = 0
    return [same_position + step_discount, basic_reward]
