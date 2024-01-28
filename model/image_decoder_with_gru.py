import torch
import torch.nn as nn
from gymnasium.spaces.discrete import Discrete
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.misc import SlimFC
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CNN(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space: Discrete, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        self.conv_layers = nn.Sequential(
            nn.Conv2d(obs_space.shape[-1], 32, kernel_size=3, stride=2, padding=1),  # Output: 50x50x32
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # Output: 25x25x64
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # Output: 13x13x128
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  # Output: 13x13x256
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(start_dim=1)
        )

        with torch.no_grad():
            dummy_input = torch.zeros(1, *obs_space.shape[1:]).permute(0, 3, 1, 2)
            self.conv_out_size = self.conv_layers(dummy_input).shape[-1]
        # Add a GRU layer for sentence data
        custom_model_config = model_config["custom_model_config"]
        self.gru = torch.nn.GRU(input_size=self.conv_out_size,
                                hidden_size=model_config["custom_model_config"]["hidden_size"],
                                num_layers=1,
                                batch_first=True)
        feature_in = custom_model_config["hidden_size"] * custom_model_config["sen_len"]

        self.fc_layers = nn.Sequential(
            SlimFC(feature_in, 256),
            SlimFC(256, 128),
            SlimFC(128, 128),
            SlimFC(128, action_space.n)
        )
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
        self._features, state = self.gru(self._features, None)
        self._features = self._features.flatten(1)

        return self.fc_layers(self._features), [state]

    def value_function(self):
        pass
