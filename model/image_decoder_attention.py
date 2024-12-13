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
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(num_channels, num_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(num_channels // reduction_ratio, num_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.global_avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        b, c, h, w = x.size()
        proj_query = self.query_conv(x).view(b, -1, w * h).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(b, -1, w * h)
        energy = torch.bmm(proj_query, proj_key)
        attention = torch.softmax(energy, dim=-1)
        proj_value = self.value_conv(x).view(b, -1, w * h)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(b, c, h, w)
        out = self.gamma * out + x
        return out


class AdditionalInfoProcessor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(AdditionalInfoProcessor, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, output_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.fc(x)


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
            **kwargs
    ):
        super().__init__(
            obs_space=obs_space,
            action_space=action_space,
            num_outputs=num_outputs,
            model_config=model_config,
            name=name,
            q_hiddens=q_hiddens,
            dueling=dueling,
            dueling_activation=dueling_activation,
            num_atoms=num_atoms,
            use_noisy=use_noisy,
            v_min=v_min,
            v_max=v_max,
            sigma0=sigma0,
            add_layer_norm=add_layer_norm,
        )
        self.view_size = 100
        self.map_size = 100
        self.map_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            SimpleAttention(32),
            nn.MaxPool2d(2),  # 50x50
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            SimpleAttention(64),
            nn.MaxPool2d(2),  # 25x25
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            SimpleAttention(128),
            nn.AdaptiveAvgPool2d((1, 1)),  # 1x1
            nn.Flatten()
        )
        self.view_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),  # Output: 50x50x32
            nn.LeakyReLU(negative_slope=0.01),
            SimpleAttention(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # Output: 25x25x64
            nn.LeakyReLU(negative_slope=0.01),
            SimpleAttention(64),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # Output: 13x13x128
            nn.LeakyReLU(negative_slope=0.01),
            SimpleAttention(128),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  # Output: 7x7x256
            nn.LeakyReLU(negative_slope=0.01),
            SimpleAttention(256),
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(1),
        )

        self.additional_info = 32
        self.additional_info_processor = AdditionalInfoProcessor(input_dim=2, output_dim=self.additional_info)

    def import_from_h5(self, h5_file: str) -> None:
        pass

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"].float()
        batch_size = obs.size(0)

        view_image_size = 3 * 100 * 100
        map_image_size = 4 * 100 * 100

        view_image = obs[:, :view_image_size]
        view_image = view_image.view(batch_size, 3, 100, 100)

        map_image = obs[:, view_image_size:view_image_size + map_image_size]
        map_image = map_image.view(batch_size, 4, 100, 100)

        additional_params = obs[:, -2:]

        view_features = self.view_layers(view_image)

        map_features = self.map_layers(map_image)

        param_features = self.additional_info_processor(additional_params)

        combined_features = torch.cat([view_features, map_features, param_features], dim=1)

        return combined_features, state

    def value_function(self):
        pass


class WrappedModel(nn.Module):
    def __init__(self, original_model):
        super(WrappedModel, self).__init__()
        self.original_model = original_model

    def forward(self, obs):
        batch_size = obs.shape[0]
        view_image_size = 3 * self.original_model.view_size * self.original_model.view_size
        map_image_size = 4 * self.original_model.map_size * self.original_model.map_size

        view_image = obs[:, :view_image_size].view(
            batch_size, 3, self.original_model.view_size, self.original_model.view_size)
        map_image = obs[:, view_image_size:view_image_size + map_image_size].view(
            batch_size, 4, self.original_model.map_size, self.original_model.map_size)

        yaw = obs[:, -3]
        speed = obs[:, -2]
        bat = obs[:, -1]
        bat = self.original_model.normalize_bat(bat)

        # feature
        map_features = self.original_model.map_layers(map_image)
        view_features = self.original_model.view_layers(view_image)

        img = self.original_model.map_attention(map_features, torch.stack([yaw, speed, bat], dim=1))
        view_ = self.original_model.front_attention(view_features, torch.stack([yaw, speed], dim=1))

        features = torch.cat([img, view_], dim=-1)

        # dueling: advantage_module, value_module
        advantage = self.original_model.advantage_module(features)
        if self.original_model.dueling:
            value = self.original_model.value_module(features)
        else:
            value = torch.zeros_like(advantage.mean(dim=-1, keepdim=True))

        logit = torch.ones_like(advantage).unsqueeze(-1)
        return advantage, value, logit


class WrappedEmbedding(nn.Module):
    def __init__(self, original_model):
        super(WrappedEmbedding, self).__init__()
        self.original_model = original_model
        self.map_size = original_model.map_size
        self.view_size = original_model.view_size

    def forward(self, obs):
        batch_size = obs.shape[0]

        view_image_size = 3 * self.view_size * self.view_size
        map_image_size = 4 * self.map_size * self.map_size

        view_image = obs[:, :view_image_size].view(
            batch_size, 3, self.view_size, self.view_size)
        map_img = obs[:, view_image_size:view_image_size + map_image_size].view(
            batch_size, 4, self.map_size, self.map_size)

        yaw = obs[:, -3]
        speed = obs[:, -2]
        bat = obs[:, -1]
        bat = self.original_model.normalize_bat(bat)

        img = self.original_model.map_layers(map_img)
        view_feat = self.original_model.view_layers(view_image)

        img = self.original_model.map_attention(img, torch.stack([yaw, speed, bat], dim=1))
        view_feat = self.original_model.front_attention(view_feat, torch.stack([yaw, speed], dim=1))

        return torch.cat([img, view_feat], dim=-1)

