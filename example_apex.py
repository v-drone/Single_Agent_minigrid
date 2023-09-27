import mlflow
import ray
from dynaconf import Dynaconf
from ray.rllib.algorithms.apex_dqn import ApexDQN
from ray.tune.logger import UnifiedLogger
from environments.ObsWrapper import FullRGBImgPartialObsWrapper
from minigrid.wrappers import ImgObsWrapper
import gymnasium as gym

# Init Ray
ray.init(
    num_cpus=10, num_gpus=1,
    include_dashboard=False,
    _system_config={"maximum_gcs_destroyed_actor_cached_count": 200},
)


# Build env
def env_creator(ec):
    gym.register(
        id='MiniGrid-RandomPath',
        entry_point='environments.MutilRoadEnv:RouteEnv'
    )
    env = gym.make("MiniGrid-RandomPath", **ec)
    env = FullRGBImgPartialObsWrapper(env, 10)  # Get pixel observations
    return ImgObsWrapper(env)


env_config = {
    "size": 20,
    "roads": (5, 7),
    "max_steps": 1000,
    "battery": 100,
    "render_mode": "rgb_array",
    "agent_pov": False
}

ray.tune.register_env("MiniGrid-RandomPath-v0")

env_example = env_creator(env_config)
env_example.reset()
observation_space = env_example.observation_space
action_space = env_example.action_space

# Config path
log_path = "./logs/"
checkpoint_path = "./checkpoints"
sub_buffer_size = 16
setting = Dynaconf(envvar_prefix="DYNACONF", settings_files="./drone.yml")

# Set hyper parameters
hyper_parameters = setting.hyper_parameters.to_dict()
hyper_parameters["logger_config"] = {"type": UnifiedLogger, "logdir": checkpoint_path}
print("log path: %s, check_path: %s" % (log_path, checkpoint_path))

hyper_parameters["train_batch_size"] = int(hyper_parameters["train_batch_size"] / sub_buffer_size)
hyper_parameters["env_config"] = env_config
mlflow.set_experiment(experiment_name="mutil_road")
mlflow_client = mlflow.tracking.MlflowClient()
run_name = "image_decoder_DPBER"
mlflow_run = mlflow.start_run(run_name=run_name, tags={"mlflow.user": "Jo.ZHOU"})

mlflow.log_params({
    **hyper_parameters.replay_buffer_config,
    "type": "MultiAgentPrioritizedBlockReplayBuffer",
    "sub_buffer_size": sub_buffer_size,
})
mlflow.log_params(
    {key: hyper_parameters[key] for key in hyper_parameters.keys() if key not in ["replay_buffer_config"]})

hyper_parameters["train_batch_size"] = int(hyper_parameters["train_batch_size"] / sub_buffer_size)

# Set algorithm
algorithm = ApexDQN(config=hyper_parameters, env="RandomPath")
