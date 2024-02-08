import torch
import logging
import torch.nn as nn
import gymnasium as gym
from ray.rllib.algorithms.dqn.dqn_torch_model import DQNTorchModel
from ray.rllib.utils.typing import ModelConfigDict
from typing import Sequence

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BasicCNN(DQNTorchModel):
    def __init__(
            self,
            obs_space: gym.spaces.Space,
            action_space: gym.spaces.Discrete,
            num_outputs: int,
            model_config: ModelConfigDict,
            name: str,
            *,
            q_hiddens: Sequence[int] = (256,),
            dueling: bool = False,
            dueling_activation: str = "relu",
            num_atoms: int = 1,
            use_noisy: bool = False,
            v_min: float = -10.0,
            v_max: float = 10.0,
            sigma0: float = 0.5,
            add_layer_norm: bool = False,
            img_size=0,
            **kwargs
    ):
        super().__init__(obs_space=obs_space, action_space=action_space,
                         num_outputs=num_outputs, model_config=model_config,
                         name=name, q_hiddens=q_hiddens,
                         dueling=dueling, dueling_activation=dueling_activation,
                         num_atoms=num_atoms,
                         use_noisy=use_noisy,
                         v_min=v_min, v_max=v_max, sigma0=sigma0,
                         add_layer_norm=add_layer_norm)
        self.img_size = img_size
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),  # Output: 50x50x32
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # Output: 25x25x64
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # Output: 13x13x128
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  # Output: 13x13x256
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(1)
        )
        self._features = None

    def import_from_h5(self, h5_file: str) -> None:
        pass

    def process_conv(self, obs):
        batch_size, f = obs.shape
        bat = obs[:, -1]
        img = obs[:, 0:-1]
        # permute b/c data comes in as [B, dim, dim, channels]:
        img = img.reshape([batch_size, self.img_size, self.img_size, 3])
        return img, bat, batch_size

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"].float()
        img, bat, batch_size = self.process_conv(obs)
        img = img.permute(0, 3, 1, 2)
        img = self.conv_layers(img)
        img = img.view(batch_size, -1)
        self._features = torch.concat([img, bat.unsqueeze(-1)], dim=-1)
        return self._features.flatten(1), state

    def value_function(self):
        pass


class WrappedModel(nn.Module):
    def __init__(self, original_model):
        super(WrappedModel, self).__init__()
        self.original_model = original_model

    def forward(self, obs):
        img, bat, batch_size = self.original_model.process_conv(obs)
        img = img.permute(0, 3, 1, 2)
        img = self.original_model.conv_layers(img)
        img = img.view(batch_size, -1)
        features = torch.concat([img, bat.unsqueeze(-1)], dim=-1)
        action_scores = features.flatten(start_dim=1)  # Ensure no in-place modification
        advantage = self.original_model.advantage_module(action_scores)
        value = self.original_model.value_module(features)
        logit = torch.unsqueeze(torch.ones_like(action_scores), -1)  # No in-place modification here
        return advantage, value, logit
