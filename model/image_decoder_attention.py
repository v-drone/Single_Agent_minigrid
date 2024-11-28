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


class SEBlock(nn.Module):
    def __init__(self, num_channels, reduction_ratio=16):
        super(SEBlock, self).__init__()
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
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
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

        self.view_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            SEBlock(32),
            nn.MaxPool2d(2),  # 50x50
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            SEBlock(64),
            nn.MaxPool2d(2),  # 25x25
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            SEBlock(128),
            nn.AdaptiveAvgPool2d((1, 1)),  # 1x1
            nn.Flatten()
        )

        self.map_layers = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            SEBlock(32),
            nn.MaxPool2d(2),  # 50x50
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            SEBlock(64),
            nn.MaxPool2d(2),  # 25x25
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            SEBlock(128),
            nn.AdaptiveAvgPool2d((1, 1)),  # 1x1
            nn.Flatten()
        )
        self.additional_info = 32
        self.additional_info_processor = AdditionalInfoProcessor(input_dim=2, output_dim=self.additional_info)

        self.fc = nn.Sequential(
            nn.Linear(128 + 128 + self.additional_info, num_outputs),
        )

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

        q_values = self.fc(combined_features)
        return q_values, state

    def value_function(self):
        pass


class WrappedModel(nn.Module):
    def __init__(self, original_model):
        super(WrappedModel, self).__init__()
        self.original_model = original_model

    def forward(self, obs):
        img, view, bat, speed, yaw, batch_size = self.original_model.process_conv(obs)

        # img
        img = img.permute(0, 3, 1, 2)
        img = self.original_model.map_layers(img)
        img = img.view(batch_size, -1)

        # view
        view = view.permute(0, 3, 1, 2)
        view = self.original_model.view_layers(view)
        view = view.view(batch_size, -1)

        img = self.original_model.map_attention(img, torch.stack([bat], dim=1))
        view = self.original_model.front_attention(view, torch.stack([yaw, speed], dim=1))

        features = torch.concat([img, view], dim=-1)
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
        img, view, bat, speed, yaw, batch_size = self.original_model.process_conv(obs)
        # img
        img = img.permute(0, 3, 1, 2)
        img = self.original_model.map_layers(img)
        img = img.view(batch_size, -1)

        # view
        view = view.permute(0, 3, 1, 2)
        view = self.original_model.view_layers(view)
        view = view.view(batch_size, -1)

        img = self.original_model.map_attention(img, torch.concat([yaw.unsqueeze(-1), bat.unsqueeze(-1)]))
        view = self.original_model.front_attention(view, torch.concat([yaw.unsqueeze(-1), speed.unsqueeze(-1)]))
        return torch.concat([img, view], dim=-1)
