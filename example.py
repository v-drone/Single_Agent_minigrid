import os
import pickle
import numpy as np
import mxnet as mx
from model import Stack, SimpleStack
from utils import check_dir
from memory import Memory
from config import *
from algorithm.DQN import DQN
from environments.SimpleEnv import SimpleEnv
ctx = mx.gpu()
for i in ["model_save", "data_save"]:
    check_dir(i)
# build models
online_model = Stack()
offline_model = Stack()
if os.path.exists(temporary_model):
    online_model.load_parameters(temporary_model, ctx=ctx)
    offline_model.load_parameters(temporary_model, ctx=ctx)
    print("load model")
else:
    online_model.collect_params().initialize(ctx=ctx)
    offline_model.collect_params().initialize(ctx=ctx)
    print("create model")

env = SimpleEnv(display=False)
env.reset_env()
# create pool
memory_pool = Memory(memory_length)
algorithm = DQN([online_model, offline_model], ctx, lr, gamma, memory_pool, action_max, temporary_model)
finish = 0
all_step_counter = 0
while True:
    if finish:
        env.reset_env()
        finish = 0
    else:
        action = algorithm.get_action(env.state(), np.maximum(1 - all_step_counter / annealing_end, epsilon_min))
        old, new, reward, finish = env.step(action[0])
        memory_pool.add(old, new, action[0], sum(reward), finish)
        all_step_counter += 1
    #  train 5 step once
    if all_step_counter % 5 == 0:
        algorithm.train()
    # save model and replace online model each 20 steps
    if all_step_counter % 20 == 0:
        algorithm.reload()
        if all_step_counter % 500 == 0:
            with open(temporary_pool, "wb") as f:
                pickle.dump(memory_pool, f)
