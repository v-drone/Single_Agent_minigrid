import numpy as np
import mxnet as mx
from model import SimpleStack
from utils import check_dir
from memory import Memory
from config import *
from algorithm.DQN import DQN
from environments.SimpleEnv import SimpleEnv
from utils import copy_params

ctx = mx.gpu()
for i in ["model_save", "data_save"]:
    check_dir(i)
# build models
online_model = SimpleStack(agent_view, map_size)
offline_model = SimpleStack(agent_view, map_size)
online_model.collect_params().initialize(mx.init.Normal(0.02), ctx=ctx)
offline_model.collect_params().initialize(mx.init.Normal(0.02), ctx=ctx)
offline_model.collect_params().zero_grad()
print("create model")
env = SimpleEnv(display=False)
# create pool
memory_pool = Memory(memory_length)
algorithm = DQN([online_model, offline_model], ctx, lr, gamma, memory_pool, action_max, temporary_model, bz=1024)
finish = 0
all_step_counter = 0
annealing_count = 0
cost = []
num_episode = 1000000
tot_reward = np.zeros(num_episode)
moving_average_clipped = 0.
moving_average = 0.
_epoch = 0
negative_addition = 0
tmp_reward = 0
for epoch in range(_epoch, num_episode):
    _epoch += 1
    env.reset_env()
    finish = 0
    cum_clipped_reward = 0
    while not finish:
        if all_step_counter > replay_start_size:
            annealing_count += 1
        if all_step_counter == replay_start_size:
            print('annealing and learning are started')
        eps = np.maximum(1 - all_step_counter / annealing_end, epsilon_min)
        action, by = algorithm.get_action(env.state(), eps)
        old, new, reward_get, finish, original_reward = env.step(action)
        memory_pool.add(old, new, action, reward_get, finish)
        cum_clipped_reward += original_reward
        all_step_counter += 1
        if finish and len(env.finish) > 50:
            sr_50 = sum(env.finish[-50:]) / min(len(env.finish), 50)
            ar_50 = sum(env.total_reward[-50:]) / sum(env.total_step_count[-50:])
            sr_all = sum(env.finish) / len(env.finish)
            ar_all = sum(env.total_reward) / sum(env.total_step_count)
            text = "success rate last 50 %f, avg return %f; success rate total %f, avg return total %f" % (
                sr_50, ar_50, sr_all, ar_all)
            with open(summary, "a") as f:
                f.writelines(text + "\n")
            if epoch % 100 == 0:
                print(text + "; %f" % eps)
        # save model and replace online model each epoch
        if annealing_count > replay_start_size and annealing_count % update_step == 0:
            copy_params(offline_model, online_model)
            offline_model.save_parameters(temporary_model)
    #  train every 4 epoch
    if annealing_count > replay_start_size and epoch % 4 == 0:
        cost.append(algorithm.train())
    tot_reward[int(epoch) - 1] = cum_clipped_reward
