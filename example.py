import matplotlib.pyplot as plt
import os
import numpy as np
import mxnet as mx
from model import Stack, SimpleStack
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
online_model = SimpleStack(7, 10)
offline_model = SimpleStack(7, 10)
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
        old, new, reward, finish, success_text, original_reward = env.step(action)
        memory_pool.add(old, new, action, reward, finish)
        cum_clipped_reward += original_reward
        all_step_counter += 1
        if success_text is not None:
            with open(summary, "a") as f:
                f.writelines(success_text + "\n")
            if epoch % 100 == 0:
                print(success_text)
    #  train every 2 epoch
    if annealing_count > replay_start_size and epoch % 2 == 0:
        cost.append(algorithm.train())
    # save model and replace online model each epoch
    if annealing_count > replay_start_size and annealing_count % update_step == 0:
        copy_params(offline_model, online_model)
        offline_model.save_parameters(temporary_model)
        print("over-write %d, last lost %d" % (_epoch, cost[-1]))
    tot_reward[int(epoch) - 1] = cum_clipped_reward
    if epoch > 50.:
        moving_average = np.mean(tot_reward[int(epoch) - 1 - 50:int(epoch) - 1])
