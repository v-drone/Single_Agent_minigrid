import torch
import logging
import torch.nn as nn
import gymnasium as gym
from ray.rllib.algorithms.dqn.dqn_torch_model import DQNTorchModel
from ray.rllib.utils.typing import ModelConfigDict
from typing import Sequence


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def sigmoid(x, alpha=50, beta=0.1):
    return 1 / (1 + torch.exp(-beta * (x - alpha)))


class SimpleAttention(nn.Module):
    def __init__(self, num_channels, reduction_ratio=16):
        super(SimpleAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(num_channels, num_channels // reduction_ratio)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(num_channels // reduction_ratio, num_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.avg_pool(x).view(b, c)
        y = self.relu(self.fc1(y))
        y = self.sigmoid(self.fc2(y))
        y = y.view(b, c, 1, 1)
        _ = x * y.expand_as(x)
        return x * y.expand_as(x)


class BatteryAttention(nn.Module):
    def __init__(self, feature_dim, battery_dim=1):
        super(BatteryAttention, self).__init__()
        self.attention_fc = nn.Linear(battery_dim, feature_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, features, battery):
        # features [batch_size, feature_dim]
        # battery [batch_size, 1]
        attention_weights = self.sigmoid(self.attention_fc(battery))
        weighted_features = features * attention_weights
        return weighted_features


class AttentionCNN(DQNTorchModel):
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
            map_size=0,
            view_size=0,
            code_size=0,
            battery=100,
            **kwargs
    ):
        super().__init__(obs_space=obs_space, action_space=action_space,
                         num_outputs=num_outputs, model_config=model_config,
                         name=name, q_hiddens=q_hiddens,
                         dueling=dueling, dueling_activation=dueling_activation,
                         num_atoms=num_atoms,
                         use_noisy=use_noisy,
                         v_min=v_min, v_max=v_max, sigma0=sigma0,
                         add_layer_norm=add_layer_norm)
        self.map_size = map_size
        self.view_size = view_size
        self.code_size = code_size
        self.battery = battery
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),  # Output: 50x50x32
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # Output: 25x25x64
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # Output: 13x13x128
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # Output: 7x7x256
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),  # Output: 7x7x512
            nn.LeakyReLU(negative_slope=0.01),
            SimpleAttention(512),
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(1),
        )
        self.view_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),  # Output: 32x32x32
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # Output: 16x16x64
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # Output: 8x8x128
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # Output: 4x4x256
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),  # Output: 4x4x512
            nn.LeakyReLU(negative_slope=0.01),
            SimpleAttention(512),
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(1),
        )

        self.code_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),  # Output: 32x32x16
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # Output: 16x16x32
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # Output: 8x8x64
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # Output: 4x4x128
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  # Output: 4x4x256
            nn.LeakyReLU(negative_slope=0.01),
            SimpleAttention(256),
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(1),
        )

        self.battery_attention = BatteryAttention(feature_dim=512 + 512 + 256)
        self._features = None

    def import_from_h5(self, h5_file: str) -> None:
        pass

    def process_conv(self, obs):
        batch_size, f = obs.shape
        bat = obs[:, -1]
        epsilon = 1e-6
        bat_prime = bat + epsilon
        bat_normalized = 1 - sigmoid(bat_prime, int(self.battery / 2), 0.1)

        if bat_normalized.device != obs.device:
            bat_normalized = bat_normalized.to(obs.device)

        img = obs[:, 0:self.map_size * self.map_size * 3]
        img = img.reshape([batch_size, self.map_size, self.map_size, 3])
        location = self.map_size * self.map_size * 3
        view = obs[:, location: location + self.view_size * self.view_size * 3]
        view = view.reshape([batch_size, self.view_size, self.view_size, 3])
        location += self.view_size * self.view_size * 3
        code = obs[:, location: -1]
        code = code.reshape([batch_size, self.code_size, self.code_size, 1])
        return img, view, code, bat_normalized, batch_size

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"].float()
        img, view, code, bat, batch_size = self.process_conv(obs)

        # map
        img = img.permute(0, 3, 1, 2)
        img = self.conv_layers(img)
        img = img.view(batch_size, -1)

        # view
        view = view.permute(0, 3, 1, 2)
        view = self.view_layers(view)
        view = view.view(batch_size, -1)

        # code
        code = code.permute(0, 3, 1, 2)
        code = self.code_layers(code)
        code = code.view(batch_size, -1)

        self._features = torch.concat([img, view, code], dim=-1)

        self._features = self.battery_attention(self._features, bat.unsqueeze(-1))

        return self._features, state

    def value_function(self):
        pass


class WrappedModel(nn.Module):
    def __init__(self, original_model):
        super(WrappedModel, self).__init__()
        self.original_model = original_model

    def forward(self, obs):
        map_img, view_img, code, bat, batch_size = self.original_model.process_conv(obs)

        # map_img
        map_img = map_img.permute(0, 3, 1, 2)
        map_img = self.original_model.conv_layers(map_img)
        map_img = map_img.view(batch_size, -1)

        # view_img
        view_img = view_img.permute(0, 3, 1, 2)
        view_img = self.original_model.view_layers(view_img)
        view_img = view_img.view(batch_size, -1)

        # code_img
        code = code.permute(0, 3, 1, 2)
        code = self.original_model.code_layers(code)
        code = code.view(batch_size, -1)

        features = torch.concat([map_img, view_img, code], dim=-1)
        features = self.original_model.battery_attention(features, bat.unsqueeze(-1))
        action_scores = features.flatten(start_dim=1)  # Ensure no in-place modification
        advantage = self.original_model.advantage_module(action_scores)
        logit = torch.unsqueeze(torch.ones_like(action_scores), -1)  # No in-place modification here
        if self.original_model.dueling:
            value = self.original_model.value_module(features)
            return advantage, value, logit
        else:
            return advantage, logit, logit


class WrappedEmbedding(nn.Module):
    def __init__(self, original_model):
        super(WrappedEmbedding, self).__init__()
        self.original_model = original_model
        self.map_size = original_model.map_size
        self.code_size = original_model.code_size
        self.view_size = original_model.view_size

    def forward(self, obs):
        map_img, view_img, code, bat, batch_size = self.original_model.process_conv(obs)

        # map_img
        map_img = map_img.permute(0, 3, 1, 2)
        map_img = self.original_model.conv_layers(map_img)
        map_img = map_img.view(batch_size, -1)

        # view_img
        view_img = view_img.permute(0, 3, 1, 2)
        view_img = self.original_model.view_layers(view_img)
        view_img = view_img.view(batch_size, -1)

        # code_img
        code = code.permute(0, 3, 1, 2)
        code = self.original_model.code_layers(code)
        code = code.view(batch_size, -1)

        features = torch.concat([map_img, view_img, code], dim=-1)
        features = self.original_model.battery_attention(features, bat.unsqueeze(-1))
        return features
