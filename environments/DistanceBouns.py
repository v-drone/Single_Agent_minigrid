import gymnasium as gym
import numpy as np


class CloserWrapper(gym.RewardWrapper):
    def __init__(
            self,
            env,
            bonus=0.01
    ):
        # Rendering attributes for observations
        super().__init__(env)
        self.bonus = bonus
        self.prev_distance = 60

    def reset(self, **kwargs):
        outputs = self.env.reset(**kwargs)
        self.prev_distance = sum(abs(np.array(self.agent_pos) - np.array(self.goal)))
        return outputs

    def reward(self, reward):
        current_distance = sum(abs(np.array(self.agent_pos) - np.array(self.goal)))
        # Calculate the change in distance to the closest blue tile
        if current_distance < self.prev_distance:
            reward += self.bonus
        self.prev_distance = current_distance
        return reward
