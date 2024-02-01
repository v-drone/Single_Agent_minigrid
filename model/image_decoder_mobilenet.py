import torch
from torch import nn
from gymnasium.spaces import Discrete
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.misc import SlimFC
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


class MobileNet(TorchModelV2, nn.Module):

    def __init__(self, obs_space, action_space: Discrete, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
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

        self.fc_layers = nn.Sequential(
            SlimFC(feature_in, 128),
            SlimFC(128, 128),
            SlimFC(128, action_space.n)
        )
        self._features = None

    def import_from_h5(self, h5_file: str) -> None:
        pass

    def forward(self, input_dict, state, seq_lens):
        self._features = input_dict["obs"].float()
        # permute b/c data comes in as [B, dim, dim, channels]:
        self._features = self._features.permute(0, 3, 1, 2)
        self._features = self.conv_layers(self._features)
        self._features = self.fc_layers(self._features.flatten(1))
        return self._features, state

    def value_function(self):
        pass
