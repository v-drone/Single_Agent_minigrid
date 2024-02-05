import gymnasium as gym


class SmallNegativeWrapper(gym.RewardWrapper):
    def __init__(
            self,
            env,
            dis=0.001,
    ):
        # Rendering attributes for observations
        super().__init__(env)
        self.dis = dis

    def reward(self, reward):
        return float(reward) - self.dis
