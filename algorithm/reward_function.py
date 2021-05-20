def short_term_reward_function(old, new, same_position):
    # parameters
    v1 = - 0.005
    # the reward from environment
    # stay over
    same_position = v1 * same_position
    return sum([same_position])


def long_term_reward_function(find_keys, travel_area):
    return find_keys + travel_area
