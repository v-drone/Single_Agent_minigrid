import ray
import argparse
from os import path
from dynaconf import Dynaconf
from apex_dpber_jade import set_hyper_parameters, train_loop
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
log_path = str(parser.parse_args().log_path)
checkpoint_path = str(parser.parse_args().checkpoint_path)

# Load Model
from model.image_decoder_block import BlockCNN

hyper_parameters, env_example = set_hyper_parameters(setting, checkpoint_path, env_name)
hyper_parameters["hiddens"] = [256, 256, 128]
model_name = "BlockCNN"
ModelCatalog.register_custom_model(model_name, BlockCNN)
hyper_parameters["model"] = {
    "custom_model": model_name,
    "no_final_linear": True,
    "fcnet_hiddens": hyper_parameters["hiddens"] + [256 + 1],
    "custom_model_config": {
        "img_size": 100,
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
trainer = ApexDDQNWithDPBER(config=hyper_parameters, env="example")
train_loop(trainer, env_example, run_name, setting, checkpoint_path, log_path)
