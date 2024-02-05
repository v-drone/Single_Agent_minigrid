import torch
import logging
import gymnasium as gym
import torch.nn as nn
from typing import Sequence
from ray.rllib.algorithms.dqn.dqn_torch_model import DQNTorchModel
from ray.rllib.utils.typing import ModelConfigDict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CnnLSTM(DQNTorchModel):
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
            hidden_size=256,
            sen_len=30,
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
        self.sen_len = sen_len
        self.hidden_size = hidden_size
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),  # Output: 50x50x32
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # Output: 25x25x64
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # Output: 13x13x128
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  # Output: 13x13x256
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(1)
        )
        with torch.no_grad():
            dummy_input = torch.zeros(1, *[3, img_size, img_size])
            self.conv_out_size = self.conv_layers(dummy_input).shape[-1]
        # Add a GRU layer for sentence data
        self.lstm = torch.nn.LSTM(input_size=self.conv_out_size + 1,
                                  hidden_size=self.hidden_size,
                                  num_layers=2,
                                  batch_first=True)

    def process_conv(self, obs):
        batch_size, seq_len, f = obs.shape
        bat = obs[:, :, -1]
        img = obs[:, :, 0:-1]
        img = img.reshape([batch_size, seq_len, 3, self.img_size, self.img_size])
        return img, bat, batch_size, seq_len

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"].float()
        img, bat, batch_size, seq_len = self.process_conv(obs)
        img = img.flatten(0, 1)
        img = self.conv_layers(img)
        img = img.view(batch_size, seq_len, -1)
        lstm_input = torch.concat([img, bat.unsqueeze(-1)], dim=-1)
        lstm_output, state = self.lstm(lstm_input, None)
        return lstm_output.flatten(1), [state]

    def value_function(self):
        pass
