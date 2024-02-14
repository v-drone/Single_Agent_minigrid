import gymnasium as gym


class HitTrodWrapper(gym.RewardWrapper):
    def __init__(
            self,
            env,
            bonus=0.02
    ):
        # Rendering attributes for observations
        super().__init__(env)
        self.bonus = bonus
        self.counter = 0

    def reset(self, **kwargs):
        outputs = self.env.reset(**kwargs)
        self.counter = 0
        return outputs

    def reward(self, reward):
        # Calculate the change in distance to the closest blue tile
        if len(self.visited_trods) > self.counter:
            reward += self.bonus
        self.counter = len(self.visited_trods)
        return reward
