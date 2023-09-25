from config import *
import os
import mlflow
import ray
from dynaconf import Dynaconf
from environments.MutilRoadEnv import RouteEnv

from algorithms.apex_ddqn import ApexDDQNWithDPBER
from replay_buffer.mpber import MultiAgentPrioritizedBlockReplayBuffer

from ray.tune.logger import UnifiedLogger
from utils import check_path, convert_np_arrays
import gymnasium as gym

# Build env
gym.register(
    id='MiniGrid-RandomPath-v0',
    entry_point='environments.MutilRoadEnv:RouteEnv'
)

env = gym.make("MiniGrid-RandomPath-v0", render_mode="human", size=20, roads=(5, 7), max_steps=1000, battery=100)
env.reset()

observation_space = env.observation_space
action_space = env.action_space

# Init Ray
ray.init(
    num_cpus=10, num_gpus=1,
    include_dashboard=False,
    _system_config={"maximum_gcs_destroyed_actor_cached_count": 200},
)

# Config path
log_path = "./logs/"
checkpoint_path = "./checkpoints"
sub_buffer_size = 16
setting = Dynaconf(envvar_prefix="DYNACONF", settings_files="./drone.yml")

# Set hyper parameters
hyper_parameters = setting.hyper_parameters.to_dict()
hyper_parameters["logger_config"] = {"type": UnifiedLogger, "logdir": checkpoint_path}
print("log path: %s \n check_path: %s" % (log_path, checkpoint_path))

# Set MLflow & log parameters
mlflow.set_tracking_uri("http://127.0.0.1:9999")
mlflow.set_experiment(experiment_name="mutil_road")
mlflow_client = mlflow.tracking.MlflowClient()
run_name = "image_decoder_DPBER"
mlflow_run = mlflow.start_run(run_name=run_name, tags={"mlflow.user": "Jo.ZHOU"})

mlflow.log_params({
    **hyper_parameters.replay_buffer_config.to_dict(),
    "type": "MultiAgentPrioritizedBlockReplayBuffer",
    "sub_buffer_size": sub_buffer_size,
})
mlflow.log_params(
    {key: hyper_parameters[key] for key in hyper_parameters.keys() if key not in ["replay_buffer_config"]})

# Set ER
replay_buffer_config = {
    **hyper_parameters.replay_buffer_config.to_dict(),
    "type": MultiAgentPrioritizedBlockReplayBuffer,
    "capacity": int(hyper_parameters.replay_buffer_config.capacity),
    "obs_space": observation_space,
    "action_space": action_space,
    "sub_buffer_size": sub_buffer_size,
    "worker_side_prioritization": False,
    "replay_buffer_shards_colocated_with_driver": True,
    "rollout_fragment_length": sub_buffer_size
}
hyper_parameters["replay_buffer_config"] = replay_buffer_config
hyper_parameters["train_batch_size"] = int(hyper_parameters["train_batch_size"] / sub_buffer_size)
algorithm = ApexDDQNWithDPBER(config=hyper_parameters, )
