import torch

import torch.nn as nn
from gymnasium.spaces.discrete import Discrete
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.misc import SlimFC, SlimConv2d
import logging
from ray.rllib.algorithms.dqn import DQNTorchPolicy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CustomCNN(TorchModelV2, nn.Module):

    def __init__(self, obs_space, action_space: Discrete, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        self.conv_layers = nn.Sequential(
            SlimConv2d(obs_space.shape[-1], 32, kernel=8, stride=4, padding=2),
            SlimConv2d(32, 64, kernel=4, stride=2, padding=1),
            SlimConv2d(64, 64, kernel=3, stride=2, padding=1),
            SlimConv2d(64, 128, kernel=3, stride=2, padding=1),
            SlimConv2d(128, 512, kernel=2, stride=2, padding=0),
        )

        with torch.no_grad():
            dummy_input = torch.zeros(1, *obs_space.shape).permute(0, 3, 1, 2)
            conv_out_size = self.conv_layers(dummy_input).flatten(1).shape[-1]

        # self.fc_layers = nn.Sequential(
        #     SlimFC(conv_out_size, 256),
        #     SlimFC(256, 3)
        # )
        self._features = None

    def import_from_h5(self, h5_file: str) -> None:
        pass

    def forward(self, input_dict, state, seq_lens):
        self._features = input_dict["obs"].float()
        # permute b/c data comes in as [B, dim, dim, channels]:
        self._features = self._features.permute(0, 3, 1, 2)
        self._features = self.conv_layers(self._features).flatten(1)
        # self._features = self.fc_layers(self._features)
        return self._features, state

    def value_function(self):
        pass
