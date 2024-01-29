import torch
from torch import nn
from gymnasium.spaces import Discrete
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.misc import SlimFC
from torchvision.models.mobilenetv3 import mobilenet_v3_small
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MobileNet(TorchModelV2, nn.Module):

    def __init__(self, obs_space, action_space: Discrete, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        self.conv_layers = nn.Sequential()
        self.conv_layers.append(mobilenet_v3_small(num_classes=action_space.n).features)
        self.conv_layers.append(nn.AdaptiveAvgPool2d(1))
        self.conv_layers.append(nn.Flatten(start_dim=1))

        with torch.no_grad():
            dummy_input = torch.zeros(1, *obs_space.shape).permute(0, 3, 1, 2)
            self.conv_out_size = self.conv_layers(dummy_input).flatten(1).shape[-1]
            feature_in = self.conv_out_size

        self.fc_layers = nn.Sequential(
            SlimFC(feature_in, 128),
            SlimFC(128, 64),
            SlimFC(64, 64),
            SlimFC(64, action_space.n)
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
