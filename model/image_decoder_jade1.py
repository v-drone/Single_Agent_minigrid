import torch
import torch.nn as nn
from gymnasium.spaces.discrete import Discrete
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.misc import SlimFC
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out += residual
        return out


class CustomBlockCNN(TorchModelV2, nn.Module):

    def __init__(self, obs_space, action_space: Discrete, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        self.conv_layers = nn.Sequential(
            ResidualBlock(obs_space.shape[-1], 16, stride=3),
            ResidualBlock(16, 32, stride=3),
            ResidualBlock(32, 64, stride=2),
            ResidualBlock(64, 256, stride=2),
        )

        with torch.no_grad():
            dummy_input = torch.zeros(1, *obs_space.shape).permute(0, 3, 1, 2)
            conv_out_size = self.conv_layers(dummy_input).flatten(1).shape[-1]

        self.fc_layers = nn.Sequential(
            SlimFC(conv_out_size, 128, activation_fn='relu'),
            SlimFC(128, action_space.n)
        )
        self._features = None

    def import_from_h5(self, h5_file: str) -> None:
        pass

    def forward(self, input_dict, state, seq_lens):
        # logging.info(input_dict)
        # logging.info(input_dict["obs"].shape)
        self._features = input_dict["obs"].float()
        # permute b/c data comes in as [B, dim, dim, channels]:
        self._features = self._features.permute(0, 3, 1, 2)
        self._features = self.conv_layers(self._features)
        self._features = self.fc_layers(self._features.flatten(1))
        return self._features, state

    def value_function(self):
        pass
