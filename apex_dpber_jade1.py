import ray
import torch
import pickle
import argparse
import numpy as np
from os import path
from dynaconf import Dynaconf
from utils import check_path
from apex_dpber_jade import set_hyper_parameters, train_loop
from ray.rllib.models import ModelCatalog
from algorithms.apex_ddqn import ApexDDQNWithDPBER
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.utils.typing import PolicyID
from ray.rllib.policy import Policy

# Init Ray
ray.init(
    num_cpus=20, num_gpus=1,
    include_dashboard=True,
    _system_config={"maximum_gcs_destroyed_actor_cached_count": 200},
)

parser = argparse.ArgumentParser()
parser.add_argument("-R", "--run_name", dest="run_name", type=int)
parser.add_argument("-S", "--setting", dest="setting_path", type=str)
parser.add_argument("-L", "--log_path", dest="log_path", type=str)
parser.add_argument("-C", "--checkpoint_path", dest="checkpoint_path", type=str)
parser.add_argument("-E", "--env", dest="env_path", type=str)

setting = parser.parse_args().setting_path
setting = Dynaconf(envvar_prefix="DYNACONF", settings_files=setting)

# Config path
env_name = parser.parse_args().env_path
run_name = str(parser.parse_args().run_name)
log_path = str(parser.parse_args().log_path)
checkpoint_path = str(parser.parse_args().checkpoint_path)

# Load hyper_parameters
with open('./model_checkpoints/only_blue/basic_cnn.pkl', 'rb') as f:
    policy_weights = pickle.load(f)["weights"]
    policy_weights = {k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v
                      for k, v in policy_weights.items()}


class RestoreReCallbacks(DefaultCallbacks):
    def __init__(self):
        super().__init__()
        self.policy_weights = policy_weights

    def on_create_policy(self, *, policy_id: PolicyID, policy: Policy) -> None:
        policy.model.load_state_dict(self.policy_weights)


hyper_parameters, env_example = set_hyper_parameters(setting, checkpoint_path, env_name)
hyper_parameters["callbacks"] = RestoreReCallbacks

# Set env
# RIDE
# ride_model = "./emb.pt"
# hyper_parameters["env_config"]["ride_model"] = ride_model
# hyper_parameters["env_config"]["device"] = "cpu"

# HIT
hyper_parameters["env_config"]["hit"] = 0.1
hyper_parameters["env_config"]["closer"] = False


hyper_parameters["hiddens"] = [256, 256, 128]
model_name = "BasicCNN"

# Load Model
from model.image_decoder import BasicCNN

ModelCatalog.register_custom_model(model_name, BasicCNN)
hyper_parameters["model"] = {
    "custom_model": model_name,
    "no_final_linear": True,
    "fcnet_hiddens": hyper_parameters["hiddens"] + [257],
    "custom_model_config": {
        "img_size": 100,
        "dueling_activation": "relu",
    }
}

run_name = "%s %s dpber %s" % (env_name, model_name, run_name)
# Check path available
check_path(log_path)
log_path = path.join(log_path, run_name)
check_path(log_path)
log_path = path.join(log_path, "data")
check_path(log_path)
check_path(checkpoint_path)
checkpoint_path = path.join(checkpoint_path, run_name)
check_path(checkpoint_path)

# Run algorithms
print(hyper_parameters["env_config"])
trainer = ApexDDQNWithDPBER(config=hyper_parameters, env="example")
train_loop(trainer, env_example, run_name, setting, checkpoint_path, log_path)
