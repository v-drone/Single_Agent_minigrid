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
from utils import copy_params

ctx = mx.gpu()
for i in ["model_save", "data_save"]:
    check_dir(i)
# build models
online_model = Stack()
offline_model = Stack()
# if os.path.exists(temporary_model):
if 1 < 0:
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
cost = []
for epoch in range(10000):
    texts = []
    while True:
        if finish:
            env.reset_env()
            finish = 0
            with open("map_simple.txt", "a") as f:
                f.writelines("\n".join(texts) + "\n")
            break
        else:
            action = algorithm.get_action(env.state(), np.maximum(1 - all_step_counter / annealing_end, epsilon_min))
            old, new, reward, finish, text, success_text = env.step(action[0])
            texts.append(text)
            memory_pool.add(old, new, action[0], sum(reward), finish)
            all_step_counter += 1
        if success_text is not None:
            with open("summary.txt", "a") as f:
                f.writelines(success_text + "\n")
        #  train 100 step once
        if all_step_counter % 100 == 0:
            cost.append(algorithm.train())
        # save model and replace online model each 100 steps
        if all_step_counter % 100 == 0:
            copy_params(offline_model, online_model)
            offline_model.save_parameters(temporary_model)
            # if all_step_counter % 10000 == 0:
            #     with open(temporary_pool, "wb") as f:
            #         pickle.dump(memory_pool, f)
