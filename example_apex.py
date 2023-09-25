import mlflow
import ray
from dynaconf import Dynaconf
from ray.rllib.algorithms.apex_dqn import ApexDQN
from ray.tune.logger import UnifiedLogger
import gymnasium as gym

# Build env
gym.register(
    id='MiniGrid-RandomPath-v0',
    entry_point='environments.MutilRoadEnv:RouteEnv'
)

env_config = {
    "size": 20,
    "roads": (5, 7),
    "max_steps": 1000,
    "battery":100,
    "render_model": "rgb_array"
}


def _make_env():
    return gym.make("MiniGrid-RandomPath-v0", render_mode="human", size=20, roads=(5, 7),
                    max_steps=1000, battery=100)


ray.tune.register_env("MiniGrid-RandomPath-v0")

env = _make_env()
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
    **hyper_parameters.replay_buffer_config,
    "type": "MultiAgentPrioritizedBlockReplayBuffer",
    "sub_buffer_size": sub_buffer_size,
})
mlflow.log_params(
    {key: hyper_parameters[key] for key in hyper_parameters.keys() if key not in ["replay_buffer_config"]})

hyper_parameters["train_batch_size"] = int(hyper_parameters["train_batch_size"] / sub_buffer_size)
algorithm = ApexDQN(config=hyper_parameters, env=env)
