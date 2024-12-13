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


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_planes, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x


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
            nn.Conv2d(4, 32, kernel_size=3, stride=1, padding=1),  # (32,100,100)
            nn.LeakyReLU(negative_slope=0.01),
            CBAM(32),
            nn.MaxPool2d(2),  # (32,50,50)
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.01),
            CBAM(64),
            nn.MaxPool2d(2),  # (64,25,25)
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.01),
            CBAM(128),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.01),
            CBAM(256),
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(1),
        )

        self.view_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),  # (32,100,100)
            nn.LeakyReLU(negative_slope=0.01),
            CBAM(32),
            nn.MaxPool2d(2),  # (32,50,50)
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.01),
            CBAM(64),
            nn.MaxPool2d(2),  # (64,25,25)
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.01),
            CBAM(128),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.01),
            CBAM(256),
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

        view_image = obs[:, :view_image_size].view(batch_size, 3, 100, 100)
        map_image = obs[:, view_image_size:view_image_size + map_image_size].view(batch_size, 4, 100, 100)

        additional_params = obs[:, -2:]

        view_features = self.view_layers(view_image)
        map_features = self.map_layers(map_image)
        param_features = self.additional_info_processor(additional_params)

        combined_features = torch.cat([view_features, map_features, param_features], dim=1)

        return combined_features, state


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
