import os
import ray
import argparse
import json
import pickle
import tqdm
from os import path
from utils import check_path, convert_np_arrays
from gymnasium.wrappers import TimeLimit
from dynaconf import Dynaconf
from ray.rllib.models import ModelCatalog
from environments.MutilRoadEnv import RouteEnv
from ray.tune.registry import register_env
from ray.tune.logger import JsonLogger
from algorithms.apex_ddqn import ApexDDQNWithDPBER
from replay_buffer.mpber import MultiAgentPrioritizedBlockReplayBuffer
from environments.ObsWrapper import FullRGBImgPartialObsWrapper
from model.image_decoder_mobilenet_large import MobileNet
from minigrid.wrappers import ImgObsWrapper

os.environ['RAY_LOG_LEVEL'] = "DEBUG"
parser = argparse.ArgumentParser()
parser.add_argument("-R", "--run_name", dest="run_name", type=int)
parser.add_argument("-L", "--log_path", dest="log_path", type=str)
parser.add_argument("-C", "--checkpoint_path", dest="checkpoint_path", type=str)

# Config path
env_name = "RandomRoad"
run_name = "MultiRoad DPBER CNN %s" % str(parser.parse_args().run_name)
log_path = parser.parse_args().log_path
checkpoint_path = parser.parse_args().checkpoint_path

setting = "/jmain02/home/J2AD006/jxb06/cxz46-jxb06/Single_Agent_minigrid/apex.yml"
setting = Dynaconf(envvar_prefix="DYNACONF", settings_files=setting)

# Init Ray
ray.init(
    num_cpus=20, num_gpus=1,
    include_dashboard=True,
    _system_config={"maximum_gcs_destroyed_actor_cached_count": 200},
)

# Set hyper parameters
hyper_parameters = setting.hyper_parameters.to_dict()
hyper_parameters["logger_config"] = {"type": JsonLogger, "logdir": checkpoint_path}
print("log path: %s \ncheck_path: %s" % (log_path, checkpoint_path))

hyper_parameters["env_config"] = {
    "size": 12,
    "roads": (1, 4),
    "max_steps": 300,
    "battery": 100,
    "render_mode": "rgb_array",
    "agent_pov": False
}


# Build env
def env_creator(env_config):
    _env = RouteEnv(**env_config)
    _env = FullRGBImgPartialObsWrapper(_env, tile_size=8, img_size=100)
    _env = TimeLimit(_env, max_episode_steps=env_config["max_steps"])
    return ImgObsWrapper(_env)


register_env(env_name, env_creator)

env = env_creator(hyper_parameters["env_config"])
obs, _ = env.reset()
step = env.step(1)
print(env.action_space, env.observation_space)

print(MobileNet(obs_space=env.observation_space,
                action_space=env.action_space,
                num_outputs=env.action_space.n,
                model_config={},
                name="Test"))

ModelCatalog.register_custom_model("MobileNet", MobileNet)

hyper_parameters["model"] = {
    "custom_model": "CustomBCNN",
    "no_final_linear": True,
    "fcnet_hiddens": hyper_parameters["hiddens"],
    "custom_model_config": {},
}

# Check path available
check_path(log_path)
log_path = path.join(log_path, run_name)
check_path(log_path)
check_path(checkpoint_path)
checkpoint_path = path.join(checkpoint_path, run_name)
check_path(checkpoint_path)
print(checkpoint_path, print(log_path))

# Set BER
sub_buffer_size = hyper_parameters["rollout_fragment_length"]
replay_buffer_config = {
    **hyper_parameters["replay_buffer_config"],
    "type": MultiAgentPrioritizedBlockReplayBuffer,
    "capacity": int(hyper_parameters["replay_buffer_config"]["capacity"]),
    "obs_space": env.observation_space,
    "action_space": env.action_space,
    "sub_buffer_size": sub_buffer_size,
    "worker_side_prioritization": False,
    "replay_buffer_shards_colocated_with_driver": True,
    "rollout_fragment_length": hyper_parameters["rollout_fragment_length"]
}
hyper_parameters["replay_buffer_config"] = replay_buffer_config
hyper_parameters["train_batch_size"] = int(hyper_parameters["train_batch_size"] / sub_buffer_size)

# Set trainer
trainer = ApexDDQNWithDPBER(config=hyper_parameters, env="example")

with open(os.path.join(checkpoint_path, "%s_config.pyl" % run_name), "wb") as f:
    pickle.dump(trainer.config.to_dict(), f)

checkpoint_path = path.join(checkpoint_path, "results")
check_path(checkpoint_path)

# Run algorithms
for i in tqdm.tqdm(range(1, setting.log.max_run)):
    try:
        result = trainer.train()
        time_used = result["time_total_s"]
        if i % 10 == 0:
            trainer.save_checkpoint(checkpoint_path)
        with open(path.join(log_path, str(i) + ".json"), "w") as f:
            result["config"] = None
            json.dump(convert_np_arrays(result), f)
        if time_used >= setting.log.max_time:
            break
    except:
        pass
