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
if os.path.exists("./model_save/model.params"):
    online_model.load_parameters("./model_save/model.params", ctx=ctx)
    offline_model.load_parameters("./model_save/model.params", ctx=ctx)
    print("load model")
env = SimpleEnv(display=True)
# create pool
memory_pool = Memory(memory_length)
algorithm = DQN([online_model, offline_model], ctx, lr, gamma, memory_pool, action_max, temporary_model, bz=1024)
finish = 0
all_step_counter = 0
annealing_count = 0
cost = []
texts = []
num_episode = 10000000
tot_reward = np.zeros(num_episode)
moving_average_clipped = 0.
moving_average = 0.
_epoch = 0
for epoch in range(_epoch, num_episode):
    _epoch += 1
    env.reset_env()
    finish = 0
    while not finish:
        eps = 0
        action, by = algorithm.get_action(env.state(), eps)
        old, new, reward_get, finish, original_reward = env.step(action)
        memory_pool.add(old, new, action, reward_get, finish)
        if finish and len(env.finish) > 50:
            sr_50 = sum(env.finish[-50:]) / min(len(env.finish), 50)
            ar_50 = sum(env.total_reward[-50:]) / sum(env.total_step_count[-50:])
            sr_all = sum(env.finish) / len(env.finish)
            ar_all = sum(env.total_reward) / sum(env.total_step_count)
            text = "success rate last 50 %f, avg return %f; success rate total %f, avg return total %f" % (sr_50, ar_50, sr_all, ar_all)
            if epoch % 100 == 0:
                print(text + "; %f" % eps)
