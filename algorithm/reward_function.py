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
    v2 = 0
    # the reward from environment
    if done:
        fuzzy_distance = (new["fuzzy_distance"][0] + 1) * (new["fuzzy_distance"][1] + 1)
        buffer = (fuzzy_distance - step_count) / 10
        if basic_reward == 1:
            basic_reward *= 1 + buffer
            print("+", buffer, step_count, new["fuzzy_distance"])
        else:
            basic_reward = 0
    else:
        basic_reward = 0
    # stay over
    same_position = - v1 * same_position
    # step used discount
    return [same_position, basic_reward]
