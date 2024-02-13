import gymnasium as gym


class HitWrapper(gym.RewardWrapper):
    def __init__(
            self,
            env,
            bonus=0.01
    ):
        # Rendering attributes for observations
        super().__init__(env)
        self.bonus = bonus

    def reward(self, reward):
        # Calculate the change in distance to the closest blue tile
        if self.current_distance < self.prev_distance:
            reward += self.bonus
        return reward
