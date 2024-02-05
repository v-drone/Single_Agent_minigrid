import os
import json
import tqdm
import pickle
import subprocess
from ray.tune.registry import register_env
from ray.tune.logger import JsonLogger
from replay_buffer.mpber import MultiAgentPrioritizedBlockReplayBuffer
from utils import minigrid_env_creator, convert_np_arrays, check_path


def set_hyper_parameters(setting, checkpoint_path, env_name):
    # Build env
    hyper_parameters = setting.hyper_parameters.to_dict()
    hyper_parameters["logger_config"] = {"type": JsonLogger, "logdir": checkpoint_path}
    hyper_parameters["env_config"] = {
        "id": env_name,
        "size": 12,
        "routes": (2, 4),
        "max_steps": 300,
        "battery": 100,
        "img_size": 100,
        "tile_size": 15,
        "num_stack": 30,
        "render_mode": "rgb_array",
        "agent_pov": False
    }

    env_example = minigrid_env_creator(hyper_parameters["env_config"])
    obs, _ = env_example.reset()
    print(env_example.action_space, env_example.observation_space)
    register_env("example", minigrid_env_creator)

    # Set BER
    sub_buffer_size = hyper_parameters["rollout_fragment_length"]
    replay_buffer_config = {
        **hyper_parameters["replay_buffer_config"],
        "type": MultiAgentPrioritizedBlockReplayBuffer,
        "capacity": 1000000,
        "obs_space": env_example.observation_space,
        "action_space": env_example.action_space,
        "sub_buffer_size": sub_buffer_size,
        "worker_side_prioritization": False,
        "replay_buffer_shards_colocated_with_driver": True,
        "rollout_fragment_length": hyper_parameters["rollout_fragment_length"]
    }
    hyper_parameters["replay_buffer_config"] = replay_buffer_config
    hyper_parameters["train_batch_size"] = int(hyper_parameters["train_batch_size"] / sub_buffer_size)

    return hyper_parameters, env_example


def train_loop(trainer, run_name, setting, checkpoint_path, log_path):
    checkpoint_path = str(checkpoint_path)
    with open(os.path.join(checkpoint_path, "%s_config.pyl" % run_name), "wb") as f:
        pickle.dump(trainer.config.to_dict(), f)
    with open(os.path.join(checkpoint_path, "%s_model_description.txt" % run_name), "w") as f:
        f.write(str(trainer.get_config().model))

    checkpoint_path = str(os.path.join(checkpoint_path, "results"))
    check_path(checkpoint_path)

    for i in tqdm.tqdm(range(1, setting.log.max_run)):
        result = trainer.train()
        time_used = result["time_total_s"]

        process = subprocess.Popen("nvidia-smi", stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        output, error = process.communicate()
        if process.returncode == 0:
            result["nvidia-smi"] = output.decode()
        else:
            result["nvidia-smi"] = error.decode()

        process = subprocess.Popen("top", stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        output, error = process.communicate()
        if process.returncode == 0:
            result["top"] = output.decode()
        else:
            result["top"] = error.decode()

        if i % setting.log.log == 0:
            trainer.save_checkpoint(checkpoint_path)
        with open(os.path.join(log_path, str(i) + ".json"), "w") as f:
            result["config"] = None
            json.dump(convert_np_arrays(result), f)
        if time_used >= setting.log.max_time:
            break
