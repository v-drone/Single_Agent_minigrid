import os
import ray
import pickle
import argparse
from os import path
from dynaconf import Dynaconf
from apex_dpber_jade import set_hyper_parameters, train_loop
from model.image_decoder import BasicCNN
from ray.rllib.models import ModelCatalog
from algorithms.apex_ddqn import ApexDDQNWithDPBER
from utils import check_path

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
log_path = parser.parse_args().log_path
checkpoint_path = parser.parse_args().checkpoint_path
run_name = env_name + " dpber " + run_name
check_path(log_path)
log_path = str(path.join(log_path, run_name))
check_path(log_path)
check_path(checkpoint_path)
checkpoint_path = path.join(checkpoint_path, run_name)
check_path(checkpoint_path)

# Load Model
ModelCatalog.register_custom_model("BasicCNN", BasicCNN)

# Load hyper_parameters
hyper_parameters, env_example = set_hyper_parameters(setting, checkpoint_path, env_name)
hyper_parameters["replay_buffer_config"]["capacity"] = 2000000
hyper_parameters["model"] = {
    "custom_model": "BasicCNN",
    "no_final_linear": True,
    "fcnet_hiddens": hyper_parameters["hiddens"] + [257],
    "custom_model_config": {
        "img_size": 100,
    }
}

# Run algorithms
trainer = ApexDDQNWithDPBER(config=hyper_parameters, env="example")
train_loop(trainer, run_name, setting, checkpoint_path, log_path)
