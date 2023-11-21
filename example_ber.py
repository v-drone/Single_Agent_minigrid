import ray
from dynaconf import Dynaconf
from ray.tune.registry import register_env
from ray.rllib.algorithms.dqn import DQN
from environments.ObsWrapper import FullRGBImgPartialObsWrapper
from model.image_decoder import CustomCNN
from minigrid.wrappers import ImgObsWrapper
from ray.tune.logger import UnifiedLogger
import gymnasium as gym


# Init Ray
ray.init(
    num_cpus=10, num_gpus=1,
    include_dashboard=True,
    _system_config={"maximum_gcs_destroyed_actor_cached_count": 200},
)

# Build env
gym.register(
    id='MiniGrid-RandomPath-v0',
    entry_point='environments.MutilRoadEnv:RouteEnv'
)

env = gym.make("MiniGrid-RandomPath-v0", render_mode="human", size=20, roads=(5, 7), max_steps=1000, battery=100)
env.reset()

observation_space = env.observation_space
action_space = env.action_space

# Config path
log_path = "./logs/"
checkpoint_path = "./checkpoints"
sub_buffer_size = 16
setting = Dynaconf(envvar_prefix="DYNACONF", settings_files="./drone.yml")

# Set hyper parameters
hyper_parameters = setting.hyper_parameters.to_dict()
hyper_parameters["logger_config"] = {"type": UnifiedLogger, "logdir": checkpoint_path}
print("log path: %s \ncheck_path: %s" % (log_path, checkpoint_path))

hyper_parameters["train_batch_size"] = int(hyper_parameters["train_batch_size"] / sub_buffer_size)
hyper_parameters["env_config"] = {
    "size": 20,
    "roads": (5, 7),
    "max_steps": 1000,
    "battery": 100,
    "render_mode": "rgb_array",
    "agent_pov": False
}
hyper_parameters["model"] = {
    "custom_model": CustomCNN,
    "custom_model_config": {},
}


# Build env
def env_creator(env_config):
    gym.register(
        id='MiniGrid-RandomPath',
        entry_point='environments.MutilRoadEnv:RouteEnv'
    )
    env_gen = gym.make("MiniGrid-RandomPath", **env_config)
    env_gen = FullRGBImgPartialObsWrapper(env_gen, tile_size=10)  # Get pixel observations
    return ImgObsWrapper(env_gen)


register_env("RandomPath", env_creator)

env = env_creator(hyper_parameters["env_config"])
env.reset()
print(env.action_space, env.observation_space)

algorithm = DQN(config=hyper_parameters, env="RandomPath")
