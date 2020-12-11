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
online_model = SimpleStack(7, 7)
offline_model = SimpleStack(7, 7)
# build models
online_model.load_parameters(temporary_model, ctx=ctx)
offline_model.load_parameters(temporary_model, ctx=ctx)
print("load model")
env = SimpleEnv(display=True)
env.reset_env()
# create pool
memory_pool = Memory(memory_length)
algorithm = DQN([online_model, offline_model], ctx, lr, gamma, memory_pool, action_max, temporary_model)
finish = 0
all_step_counter = 0
annealing_count = 0
cost = []
texts = []
num_episode = 100000
tot_reward = np.zeros(num_episode)
moving_average_clipped = 0.
moving_average = 0.
_epoch = 0
for epoch in range(1, num_episode):
    _epoch += 1
    with open("map_simple.txt", "a") as f:
        f.writelines("\n".join(texts) + "\n")
        texts = []
    env.reset_env()
    finish = 0
    cum_clipped_reward = 0
    while not finish:
        if all_step_counter > replay_start_size:
            annealing_count += 1
        if all_step_counter == replay_start_size:
            print('annealing and learning are started')
        state = env.state()
        action, by = algorithm.get_action(state, 0)
        old, new, reward_get, finish, text, success_text = env.step(action)
        texts.append(text)
        memory_pool.add(old, new, action, sum(reward_get), finish)
        cum_clipped_reward += sum(reward_get)
        all_step_counter += 1
    tot_reward[int(epoch) - 1] = cum_clipped_reward
    if epoch > 50.:
        moving_average = np.mean(tot_reward[int(epoch) - 1 - 50:int(epoch) - 1])
