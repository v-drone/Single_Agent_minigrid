import os
import ray
import datetime
import argparse
import json
import pickle
import tqdm
from os import path
from utils import check_path, convert_np_arrays, flatten_dict
from gymnasium.wrappers import TimeLimit
from dynaconf import Dynaconf
from ray.rllib.models import ModelCatalog
from model.image_decoder import CNN
from ray.tune.registry import register_env
from ray.tune.logger import JsonLogger
from algorithms.apex_ddqn import ApexDDQNWithDPBER
from replay_buffer.mpber import MultiAgentPrioritizedBlockReplayBuffer
from gymnasium.experimental.wrappers import ResizeObservationV0
from minigrid.wrappers import FullyObsWrapper, RGBImgObsWrapper, ImgObsWrapper
import gymnasium as gym

import mlflow
from mlflow.exceptions import MlflowException
from func_timeout import FunctionTimedOut
from botocore.exceptions import ConnectionClosedError

os.environ['RAY_LOG_LEVEL'] = "DEBUG"
parser = argparse.ArgumentParser()
parser.add_argument("-R", "--run_name", dest="run_name", type=int)
parser.add_argument("-L", "--log_path", dest="log_path", type=str)
parser.add_argument("-C", "--checkpoint_path", dest="checkpoint_path", type=str)

# Config path
env_name = "LavaCrossingS9N3"
run_name = "%s DPBER MobileNet small %s" % (env_name, datetime.datetime.strftime(datetime.datetime.now(), "%Y%m%d"))
log_path = parser.parse_args().log_path
checkpoint_path = parser.parse_args().checkpoint_path

# Check path available
check_path(log_path)
log_path = path.join(log_path, run_name)
check_path(log_path)
check_path(checkpoint_path)
checkpoint_path = path.join(checkpoint_path, run_name)
check_path(checkpoint_path)
print(checkpoint_path, print(log_path))

setting = "/jmain02/home/J2AD006/jxb06/cxz46-jxb06/Single_Agent_minigrid/apex.yml"
setting = Dynaconf(envvar_prefix="DYNACONF", settings_files=setting)

# Init Ray
ray.init(
    num_cpus=16, num_gpus=1,
    object_store_memory=100 * 1024 * 1024 * 1024,
    include_dashboard=True,
    _system_config={"maximum_gcs_destroyed_actor_cached_count": 200},
)

# Set hyper parameters
hyper_parameters = setting.hyper_parameters.to_dict()
hyper_parameters["logger_config"] = {"type": JsonLogger, "logdir": checkpoint_path}
hyper_parameters["env_config"] = {
    "size": 12,
    "roads": (1, 4),
    "max_steps": 300,
    "battery": 100,
    "img_size": 80,
    "tile_size": 8,
    "num_stack": 1,
    "render_mode": "rgb_array",
    "agent_pov": False
}


# Build env
def env_creator(env_config):
    env = gym.make("MiniGrid-LavaCrossingS9N3-v0", render_mode="rgb_array")
    env = FullyObsWrapper(env)
    env = RGBImgObsWrapper(env, tile_size=env_config["tile_size"])
    env = ImgObsWrapper(env)
    env = ResizeObservationV0(env, (env_config["img_size"], env_config["img_size"]))
    env = TimeLimit(env, max_episode_steps=env_config["max_steps"])
    return env


register_env(env_name, env_creator)

env_example = env_creator(hyper_parameters["env_config"])
obs, _ = env_example.reset()
step = env_example.step(1)
print(env_example.action_space, env_example.observation_space)

ModelCatalog.register_custom_model("CNN", CNN)

hyper_parameters["model"] = {
    "custom_model": "CNN",
    "no_final_linear": True,
    "fcnet_hiddens": hyper_parameters["hiddens"],
    "custom_model_config": {},
}

# Set BER
sub_buffer_size = hyper_parameters["rollout_fragment_length"]
replay_buffer_config = {
    **hyper_parameters["replay_buffer_config"],
    "type": MultiAgentPrioritizedBlockReplayBuffer,
    "capacity": int(hyper_parameters["replay_buffer_config"]["capacity"]),
    "obs_space": env_example.observation_space,
    "action_space": env_example.action_space,
    "sub_buffer_size": sub_buffer_size,
    "worker_side_prioritization": False,
    "replay_buffer_shards_colocated_with_driver": True,
    "rollout_fragment_length": hyper_parameters["rollout_fragment_length"]
}
hyper_parameters["replay_buffer_config"] = replay_buffer_config
hyper_parameters["train_batch_size"] = int(hyper_parameters["train_batch_size"] / sub_buffer_size)
trainer = ApexDDQNWithDPBER(config=hyper_parameters, env=env_name)

with open(os.path.join(checkpoint_path, "%s_config.pyl" % run_name), "wb") as f:
    pickle.dump(trainer.config.to_dict(), f)

checkpoint_path = path.join(checkpoint_path, "results")
check_path(checkpoint_path)

# Run algorithms
keys_to_extract_sam = {"episode_reward_max", "episode_reward_min", "episode_reward_mean"}
keys_to_extract_sta = {"num_agent_steps_sampled", "num_agent_steps_trained", "episode_reward_mean"}
keys_to_extract_buf = {"add_batch_time_ms", "replay_time_ms", "update_priorities_time_ms"}

# Run algorithms
for i in tqdm.tqdm(range(1, 2000)):
    result = trainer.train()
    try:
        result = trainer.train()
        time_used = result["time_total_s"]
    except:
        continue
    try:
        sampler = result.get("sampler_results", {}).copy()
        eva = result.get("evaluation", {}).copy()
        info = result.get("info", {}).copy()
        sam = {key: sampler[key] for key in keys_to_extract_sam if key in sampler}
        sta = {key: info[key] for key in keys_to_extract_sta if key in info}
        buf = flatten_dict(info.get("replay_shard_0", {}))
        lea = info.get("learner", {}).get("time_usage", {})
        if eva:
            eva = {"eval_" + key: sampler[key] for key in keys_to_extract_sam if key in eva}
        mlflow.log_metrics({**sam, **sta, **buf, **lea, **eva}, step=result["episodes_total"])
        if i % (setting.log.log * 10) == 0:
            trainer.save_checkpoint(checkpoint_path)
            mlflow.log_artifacts(log_path)
            mlflow.log_artifacts(checkpoint_path)
    except FunctionTimedOut:
        tqdm.tqdm.write("logging failed")
    except MlflowException:
        tqdm.tqdm.write("logging failed")
    except ConnectionClosedError:
        tqdm.tqdm.write("logging failed")
    with open(path.join(log_path, str(i) + ".json"), "w") as f:
        result["config"] = None
        json.dump(convert_np_arrays(result), f)
    if time_used >= setting.log.max_time:
        break
