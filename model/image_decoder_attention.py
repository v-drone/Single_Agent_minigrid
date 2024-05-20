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


class ValueAttention(nn.Module):
    def __init__(self, feature_dim, value_dim=1, hidden_dim=128):
        super(ValueAttention, self).__init__()
        self.feature_transform = nn.Linear(feature_dim, hidden_dim)
        self.value_transform = nn.Linear(value_dim, hidden_dim)
        self.final_transform = nn.Linear(hidden_dim, feature_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, features, value):
        transformed_features = self.feature_transform(features)
        value_info = self.value_transform(value)
        combined = transformed_features + value_info
        attention_weights = self.sigmoid(self.final_transform(combined))
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
        self.battery = battery
        self.conv_layers = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, stride=2, padding=1),  # Output: 45x45x32
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # Output: 23x23x64
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # Output: 11x11x128
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # Output: 6x6x256
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),  # Output: 6x6x512
            nn.LeakyReLU(negative_slope=0.01),
            SimpleAttention(512),
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(1),
        )
        self.view_layers = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, stride=2, padding=1),  # Output: 50x50x32
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # Output: 25x25x64
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # Output: 13x13x128
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # Output: 7x7x256
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),  # Output: 7x7x512
            nn.LeakyReLU(negative_slope=0.01),
            SimpleAttention(512),
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(1),
        )
        self.front_attention = ValueAttention(feature_dim=256)
        self.map_attention = ValueAttention(feature_dim=256)

    def import_from_h5(self, h5_file: str) -> None:
        pass

    def process_conv(self, obs):
        batch_size, f = obs.shape
        bat = obs[:, -1]
        speed = obs[: -2]
        yaw = obs[: -3]
        epsilon = 1e-65
        bat_prime = bat + epsilon
        bat_normalized = 1 - sigmoid(bat_prime, int(self.battery / 2), 0.1)

        if bat_normalized.device != obs.device:
            bat_normalized = bat_normalized.to(obs.device)
        view = obs[:, 0: self.view_size * self.view_size * 3]
        view = view.reshape([batch_size, self.view_size, self.view_size, 3])
        location = self.view_size * self.view_size * 3

        img = obs[:, location: location + self.map_size * self.map_size * 4]
        img = img.reshape([batch_size, self.map_size, self.map_size, 4])
        location += self.map_size * self.map_size * 4
        return img, view, bat_normalized, speed, yaw, batch_size

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"].float()
        img, view, bat, speed, yaw, batch_size = self.process_conv(obs)

        # map
        img = img.permute(0, 3, 1, 2)
        img = self.conv_layers(img)
        img = img.view(batch_size, -1)

        # view
        view = view.permute(0, 3, 1, 2)
        view = self.view_layers(view)
        view = view.view(batch_size, -1)

        img = self.map_attention(img, torch.concat([yaw.unsqueeze(-1), bat.unsqueeze(-1)]))

        view = self.front_attention(view, torch.concat([yaw.unsqueeze(-1), speed.unsqueeze(-1)]))
        logging.info(torch.concat([img, view], dim=-1).shape)
        return torch.concat([img, view], dim=-1), state

    def value_function(self):
        pass


class WrappedModel(nn.Module):
    def __init__(self, original_model):
        super(WrappedModel, self).__init__()
        self.original_model = original_model

    def forward(self, obs):
        map_img, view_img, bat, speed, yaw, batch_size = self.original_model.process_conv(obs)

        # map_img
        map_img = map_img.permute(0, 3, 1, 2)
        map_img = self.original_model.conv_layers(map_img)
        map_img = map_img.view(batch_size, -1)

        # view_img
        view_img = view_img.permute(0, 3, 1, 2)
        view_img = self.original_model.view_layers(view_img)
        view_img = view_img.view(batch_size, -1)

        map_img = self.original_model.map_attention(map_img, torch.concat([yaw.unsqueeze(-1), bat.unsqueeze(-1)]))

        view_img = self.original_model.front_attention(view_img, torch.concat([yaw.unsqueeze(-1), speed.unsqueeze(-1)]))

        features = torch.concat([map_img, view_img], dim=-1)
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
        map_img, view_img, bat, speed, yaw, batch_size = self.original_model.process_conv(obs)
        # map_img
        map_img = map_img.permute(0, 3, 1, 2)
        map_img = self.original_model.conv_layers(map_img)
        map_img = map_img.view(batch_size, -1)

        # view_img
        view_img = view_img.permute(0, 3, 1, 2)
        view_img = self.original_model.view_layers(view_img)
        view_img = view_img.view(batch_size, -1)

        map_img = self.original_model.map_attention(map_img, torch.concat([yaw.unsqueeze(-1), bat.unsqueeze(-1)]))
        view_img = self.original_model.front_attention(view_img, torch.concat([yaw.unsqueeze(-1), speed.unsqueeze(-1)]))
        return torch.concat([map_img, view_img], dim=-1)
