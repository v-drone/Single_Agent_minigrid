import torch
import gymnasium as gym
from torch import nn
from model.image_decoder import BasicCNN
from ray.rllib.utils.typing import ModelConfigDict
from typing import Sequence
from torchvision.models.mobilenetv3 import InvertedResidualConfig, partial, MobileNetV3
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _mobilenet_v3_conf(width_mult: float = 1.0):
    adjust_channels = partial(InvertedResidualConfig.adjust_channels, width_mult=width_mult)
    bneck_conf = partial(InvertedResidualConfig, width_mult=width_mult)
    inverted_residual_setting = [
        bneck_conf(3, 3, 16, 32, False, "RE", 2, 1),  # Output: 50x50x16
        bneck_conf(32, 3, 64, 64, False, "RE", 2, 1),  # Output: 25x25x24
        bneck_conf(64, 3, 72, 128, False, "RE", 2, 1),  # Output: 13x13x24
        bneck_conf(128, 3, 120, 256, True, "HS", 1, 1),  # Output: 13x13x256, with SE and HS
    ]
    last_channel = adjust_channels(1024)  # C5
    return inverted_residual_setting, last_channel


class MobileNet(BasicCNN):

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
    ):
        super().__init__(obs_space=obs_space, action_space=action_space,
                         num_outputs=num_outputs, model_config=model_config,
                         name=name, q_hiddens=q_hiddens,
                         dueling=dueling, dueling_activation=dueling_activation,
                         num_atoms=num_atoms,
                         use_noisy=use_noisy,
                         v_min=v_min, v_max=v_max, sigma0=sigma0,
                         add_layer_norm=add_layer_norm)

        inverted_residual_setting, last_channel = _mobilenet_v3_conf()

        self.conv_layers = nn.Sequential()
        self.conv_layers.append(nn.Conv2d(3, 8, kernel_size=1, stride=1, padding=0))
        self.conv_layers.append(MobileNetV3(inverted_residual_setting, last_channel, num_classes=3).features[1: -1])
        self.conv_layers.append(nn.AdaptiveAvgPool2d(1))
        self.conv_layers.append(nn.Flatten(start_dim=1))

        with torch.no_grad():
            dummy_input = torch.zeros(1, *obs_space.shape).permute(0, 3, 1, 2)
            self.conv_out_size = self.conv_layers(dummy_input).flatten(1).shape[-1]
            feature_in = self.conv_out_size
