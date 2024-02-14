import gymnasium as gym
import numpy as np
import torch


class SimpleRIDEWrapper(gym.RewardWrapper):
    def __init__(self,
                 env,
                 model_path,
                 device="cpu",
                 impact_coefficient=0.005
                 ):
        super(SimpleRIDEWrapper, self).__init__(env)
        self.device = device
        self.prev_state = None
        self.impact_coefficient = impact_coefficient
        self.model_path = model_path
        self.num_step = 0
        self.model = None
        self.load_model()
        self.reset_counter = 0

    def load_model(self):
        if self.device == "cuda":
            self.model = torch.load(self.model_path, map_location="cuda")
        else:
            self.model = torch.load(self.model_path, map_location="cpu")

    def reset(self, **kwargs):
        outputs = self.env.reset(**kwargs)
        self.prev_state = outputs[0]
        self.num_step = 0
        self.reset_counter += 1
        if self.reset_counter >= 100:
            self.load_model()
            self.reset_counter = 0
        return outputs

    def step(self, action):
        new_state, reward, terminated, truncated, info = self.env.step(action)
        self.num_step += 1
        intrinsic_reward = self.calculate_intrinsic_reward(new_state)
        # Modify the reward with the intrinsic reward
        total_reward = float(reward) + intrinsic_reward * self.impact_coefficient
        self.prev_state = new_state
        return new_state, total_reward, terminated, truncated, info

    def calculate_intrinsic_reward(self, state):
        if self.device == "cuda":
            prev = torch.Tensor(np.expand_dims(self.prev_state, axis=0)).to("cuda")
            new = torch.Tensor(np.expand_dims(state, axis=0)).to("cuda")
            prev_emb = self.model(prev)[0].to("cpu")
            new_emb = self.model(new)[0].to("cpu")
        else:
            prev = torch.Tensor(np.expand_dims(self.prev_state, axis=0))
            new = torch.Tensor(np.expand_dims(state, axis=0))
            prev_emb = self.model(prev)[0]
            new_emb = self.model(new)[0]

        # cal L2 norm divided by N_ep(s_t+1)
        reward = float(torch.linalg.norm(new_emb - prev_emb).detach().numpy())
        reward = reward / (np.sqrt(self.num_step))
        return reward
