import torch
from torch import nn
from gymnasium.spaces import Discrete
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from torchvision.models.mobilenetv3 import mobilenet_v3_small
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MobileNet(TorchModelV2, nn.Module):

    def __init__(self, obs_space, action_space: Discrete, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        self.conv_layers = mobilenet_v3_small(num_classes=action_space.n)
        self._features = None

    def import_from_h5(self, h5_file: str) -> None:
        pass

    def forward(self, input_dict, state, seq_lens):
        self._features = input_dict["obs"].float()
        # permute b/c data comes in as [B, dim, dim, channels]:
        self._features = self._features.permute(0, 3, 1, 2)
        self._features = self.conv_layers(self._features)
        return self._features, state

    def value_function(self):
        pass
