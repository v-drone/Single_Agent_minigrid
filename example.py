from config import *
import os
import gym
import ray
import tqdm
import json
import pickle
import argparse
from os import path
from dynaconf import Dynaconf
from algorithms.apex_ddqn import ApexDDQNWithDPBER
from replay_buffer.mpber import MultiAgentPrioritizedBlockReplayBuffer
from ray.tune.logger import UnifiedLogger
from utils import check_path, convert_np_arrays

ray.init(
    num_cpus=6, num_gpus=1,
    include_dashboard=False,
    _system_config={"maximum_gcs_destroyed_actor_cached_count": 200},
)

# Config path
log_path = "./logs/"
checkpoint_path = "./checkpoints"
setting = Dynaconf(envvar_prefix="DYNACONF", settings_files="./drone.yml")

# Set hyper parameters
hyper_parameters = setting.hyper_parameters.to_dict()
hyper_parameters["logger_config"] = {"type": UnifiedLogger, "logdir": checkpoint_path}

if os.path.exists(summary):
    os.remove(summary)
# build models
model = SimpleStack()
print(model)
model.load_parameters("./model_save/model_test.params.best")
# create env
env = SimpleEnv(display=True)
env.reset_env()
memory_pool = Memory(memory_length)
annealing = 0
total_reward = np.zeros(num_episode)
eval_result = []
loss_func = gluon.loss.L2Loss()
for epoch in range(num_episode):
    env.reset_env()
    finish = 0
    cum_clipped_dr = 0
    while not finish:
        if sum(env.step_count) > replay_start:
            annealing += 1
        eps = np.maximum(1 - sum(env.step_count) / annealing_end, epsilon_min)
        by = "Model"
        data = create_input([translate_state(env.map.state())])
        data = [nd.array(i, ctx=ctx) for i in data]
        action = model(data)
        print(action)
        action = int(nd.argmax(action, axis=1).asnumpy()[0])
        old, new, reward_get, finish = env.step(action)
        memory_pool.add(old, new, action, reward_get, finish)
        if finish and epoch > 50:
            cum_clipped_dr += env.detect_rate[-1]
            dr_50 = float(np.mean(env.detect_rate[-50:]))
            dr_all = float(np.mean(env.detect_rate))
            if epoch % 50 == 0:
                text = "DR: %f(50), %f(all), eps: %f" % (dr_50, dr_all, eps)
                print(text)
                with open(summary, "a") as f:
                    f.writelines(text + "\n")
    total_reward[int(epoch) - 1] = cum_clipped_dr
