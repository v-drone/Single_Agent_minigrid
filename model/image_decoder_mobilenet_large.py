import torch.nn as nn
from gymnasium.spaces.discrete import Discrete
from gymnasium.spaces import Space
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from torchvision.models import mobilenetv3

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MobileNet(TorchModelV2, nn.Module):

    def __init__(self, obs_space, action_space: Space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        self.basic_net = mobilenetv3.mobilenet_v3_large(num_classes=num_outputs)
        self.basic_net.features[0][0] = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self._features = None

    def import_from_h5(self, h5_file: str) -> None:
        pass

    def forward(self, input_dict, state, seq_lens):
        # logging.info(input_dict)
        # logging.info(input_dict["obs"].shape)
        self._features = input_dict["obs"].float()
        # permute b/c data comes in as [B, dim, dim, channels]:
        self._features = self._features.permute(0, 3, 1, 2)
        self._features = self.basic_net(self._features)
        return self._features, state

    def value_function(self):
        pass
