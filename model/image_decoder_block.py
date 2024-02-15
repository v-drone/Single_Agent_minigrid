import torch
import torch.nn as nn
import gymnasium as gym
from model.image_decoder import BasicCNN
from ray.rllib.models.torch.misc import SlimConv2d
from ray.rllib.utils.typing import ModelConfigDict
from typing import Sequence
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.tunnel = nn.Sequential()
        self.tunnel.append(SlimConv2d(in_channels, out_channels, kernel=3, stride=1, padding=1))
        self.tunnel.append(SlimConv2d(out_channels, out_channels, kernel=3, stride=1, padding=1))
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut.append(SlimConv2d(in_channels, out_channels, kernel=3, stride=1, padding=1))

    def forward(self, x):
        identity = x
        conv = self.tunnel(x)
        shortcut = self.shortcut(identity)
        output = torch.add(conv, shortcut)
        return output


class BlockCNN(BasicCNN):
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
                         num_atoms=num_atoms, use_noisy=use_noisy,
                         v_min=v_min, v_max=v_max, sigma0=sigma0,
                         add_layer_norm=add_layer_norm,
                         img_size=img_size,
                         **kwargs
                         )
        self.conv_layers = nn.Sequential(
            ResidualBlock(3, 32),
            nn.AvgPool2d(2),                                # Output: 50x50x32
            ResidualBlock(32, 64),
            nn.AvgPool2d(2),                                # Output: 25x25x64
            ResidualBlock(64, 128),
            nn.AvgPool2d(2),                                # Output: 12x12x128
            ResidualBlock(128, 256),
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(1)
        )
