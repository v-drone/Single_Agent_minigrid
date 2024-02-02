import torch
import gymnasium as gym
import torch.nn as nn
from typing import Sequence
from ray.rllib.algorithms.dqn.dqn_torch_model import DQNTorchModel
from ray.rllib.utils.typing import ModelConfigDict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CNN(DQNTorchModel):
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
            add_layer_norm: bool = False
    ):
        super().__init__(obs_space=obs_space, action_space=action_space,
                         num_outputs=num_outputs, model_config=model_config,
                         name=name, q_hiddens=q_hiddens,
                         dueling=dueling, dueling_activation=dueling_activation,
                         num_atoms=num_atoms,
                         use_noisy=use_noisy,
                         v_min=v_min, v_max=v_max, sigma0=sigma0,
                         add_layer_norm=add_layer_norm)

        self.conv_layers = nn.Sequential(
            nn.Conv2d(obs_space.shape[-1], 32, kernel_size=3, stride=2, padding=1),  # Output: 50x50x32
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # Output: 25x25x64
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # Output: 13x13x128
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),  # Output: 13x13x256
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(start_dim=1)
        )

        with torch.no_grad():
            dummy_input = torch.zeros(1, *obs_space.shape[1:]).permute(0, 3, 1, 2)
            self.conv_out_size = self.conv_layers(dummy_input).shape[-1]
        # Add a GRU layer for sentence data
        custom_model_config = model_config["custom_model_config"]
        self.lstm = torch.nn.LSTM(input_size=self.conv_out_size,
                                  hidden_size=custom_model_config["hidden_size"],
                                  num_layers=1,
                                  batch_first=True)
        self._features = None

    def import_from_h5(self, h5_file: str) -> None:
        pass

    def forward(self, input_dict, state, seq_lens):
        self._features = input_dict["obs"].float()
        batch_size, seq_len, c, h, w = self._features.shape
        self._features = self._features.permute(0, 1, 4, 2, 3)
        self._features = self._features.flatten(0, 1)
        self._features = self.conv_layers(self._features)
        self._features = self._features.view(batch_size, seq_len, -1)
        self._features, state = self.lstm(self._features, None)
        self._features = self._features[:, -1, :]
        return self._features.flatten(1), state

    def value_function(self):
        pass
