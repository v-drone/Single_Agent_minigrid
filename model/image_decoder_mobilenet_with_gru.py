import torch
from gymnasium.spaces import Space
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.misc import SlimFC
from torchvision.models import mobilenetv3
from torchvision.models.mobilenetv3 import InvertedResidual, InvertedResidualConfig, partial, MobileNetV3
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _mobilenet_v3_conf(width_mult: float = 1.0):
    adjust_channels = partial(InvertedResidualConfig.adjust_channels, width_mult=width_mult)
    bneck_conf = partial(InvertedResidualConfig, width_mult=width_mult)
    inverted_residual_setting = [
        bneck_conf(3, 3, 16, 16, False, "RE", 2, 1),  # Output: 50x50x16
        bneck_conf(16, 3, 64, 24, False, "RE", 2, 1),  # Output: 25x25x24
        bneck_conf(24, 3, 72, 24, False, "RE", 2, 1),  # Output: 13x13x24
        bneck_conf(24, 3, 96, 40, True, "HS", 2, 1),  # Output: 7x7x40, with SE and HS
        bneck_conf(40, 3, 240, 40, True, "HS", 1, 1),  # Output: 7x7x40, with SE and HS
        bneck_conf(40, 3, 120, 48, True, "HS", 1, 1),
        bneck_conf(48, 3, 144, 48, True, "HS", 1, 1),
        bneck_conf(48, 1, 288, 96, False, "HS", 1, 1)
    ]
    last_channel = adjust_channels(1024)  # C5
    return inverted_residual_setting, last_channel


class MobileNet(TorchModelV2, torch.nn.Module):

    def __init__(self, obs_space, action_space: Space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        torch.nn.Module.__init__(self)
        inverted_residual_setting, last_channel = _mobilenet_v3_conf()

        self.features = torch.nn.Sequential()
        self.features.append(torch.nn.Conv2d(3, 8, kernel_size=1, stride=1, padding=0))
        self.features.append(MobileNetV3(inverted_residual_setting, last_channel, num_classes=3).features[1: -1])
        self.features.append(torch.nn.AdaptiveAvgPool2d(1))
        self.features.append(torch.nn.Flatten(start_dim=1))

        with torch.no_grad():
            dummy_input = torch.zeros(1, *obs_space.shape[1:]).permute(0, 3, 1, 2)
            self.conv_out_size = self.features(dummy_input).shape[-1]
            feature_in = self.conv_out_size
        # Add a GRU layer for sentence data
        custom_model_config = model_config["custom_model_config"]
        self.gru = torch.nn.GRU(input_size=self.conv_out_size,
                                hidden_size=model_config["custom_model_config"]["hidden_size"],
                                num_layers=1,
                                batch_first=True)
        feature_in = custom_model_config["hidden_size"] * custom_model_config["sen_len"]

        self.fc_layers = torch.nn.Sequential(
            SlimFC(feature_in,
                   256, activation_fn='relu'),
            SlimFC(256, num_outputs)
        )

        self._features = None

    def import_from_h5(self, h5_file: str) -> None:
        pass

    def forward(self, input_dict, state, seq_lens):
        self._features = input_dict["obs"].float()
        batch_size, seq_len, c, h, w = self._features.shape
        self._features = self._features.permute(0, 1, 4, 2, 3)
        self._features = self._features.flatten(0, 1)

        self._features = self.features(self._features)

        # self._features = self._features.view(batch_size, seq_len, -1)
        # self._features, state = self.gru(self._features, None)
        # self._features = self._features.flatten(1)

        return self.fc_layers(self._features), [state]

    def value_function(self):
        pass
