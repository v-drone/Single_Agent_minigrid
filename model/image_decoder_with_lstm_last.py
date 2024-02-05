import torch
import gymnasium as gym
from typing import Sequence
from model.image_decoder_with_lstm import CnnLSTM
from ray.rllib.utils.typing import ModelConfigDict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LastCNN(CnnLSTM):
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
            img_size=0,
            hidden_size=256,
            sen_len=30,
            **kwargs
    ):
        super().__init__(obs_space=obs_space, action_space=action_space,
                         num_outputs=num_outputs, model_config=model_config,
                         name=name, q_hiddens=q_hiddens,
                         dueling=dueling, dueling_activation=dueling_activation,
                         num_atoms=num_atoms,
                         use_noisy=use_noisy,
                         v_min=v_min, v_max=v_max, sigma0=sigma0,
                         add_layer_norm=add_layer_norm,
                         img_size=img_size,
                         hidden_size=hidden_size,
                         sen_len=sen_len,
                         **kwargs
                         )

    def import_from_h5(self, h5_file: str) -> None:
        pass

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"].float()
        img, bat, batch_size, seq_len = self.process_conv(obs)
        img = img.flatten(0, 1)
        img = self.conv_layers(img)
        img = img.view(batch_size, seq_len, -1)
        lstm_input = torch.concatenate([img, bat.unsqueeze(-1)], dim=-1)
        lstm_output, state = self.lstm(lstm_input, None)
        lstm_output = lstm_output[:, -1, :]
        return lstm_output.flatten(1), [state]

    def value_function(self):
        pass
