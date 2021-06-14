import numpy as np


def short_term_reward_function(old, new, same_position):
    # stay over
    same_position = - 0.005 * same_position
    # small pos for search
    search = np.array(new["reward"][1]) - np.array(old["reward"][1])
    return sum([same_position, search])


def long_term_reward_function(find_keys, travel_area):
    return find_keys + travel_area
