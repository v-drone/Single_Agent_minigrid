from config import *
import os
import numpy as np
import mxnet as mx
from model.simple_stack import SimpleStack
from utils import check_dir
from memory import Memory
from algorithm.DQN import DQN
from environments.SimpleEnv import SimpleEnv
from utils import copy_params
from evaluation import evaluate

if os.path.exists(summary):
    os.remove(summary)
ctx = mx.gpu()
for i in ["model_save", "data_save"]:
    check_dir(i)
# build models
online_model = SimpleStack(agent_view, map_size)
offline_model = SimpleStack(agent_view, map_size)
online_model.collect_params().initialize(mx.init.Normal(0.02), ctx=ctx)
offline_model.collect_params().initialize(mx.init.Normal(0.02), ctx=ctx)
offline_model.collect_params().zero_grad()
# create env
env = SimpleEnv(display=True)
env.reset_env()
memory_pool = Memory(memory_length, ctx=ctx)
# workflow
algorithm = DQN([online_model, offline_model], ctx, lr, gamma, memory_pool,
                action_max, temporary_model, bz=1024)
annealing = 0
total_reward = np.zeros(num_episode)
eval_result = []
for epoch in range(num_episode):
    env.reset_env()
    finish = 0
    cum_clipped_dr = 0
    if epoch == 100:
        print("Model Structure: ")
        print(offline_model)
    while not finish:
        if sum(env.step_count) > replay_start:
            annealing += 1
        if sum(env.step_count) == replay_start:
            print('annealing and learning are started')
        eps = np.maximum(1 - sum(env.step_count) / annealing_end, epsilon_min)
        action, by = algorithm.get_action(env.map.state(), eps)
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
            if epoch % 100 == 0 and annealing > replay_start:
                eval_result.extend(evaluate(offline_model, 5, ctx))
        # save model and replace online model each update_step
        if annealing > replay_start and annealing % update_step == 0:
            copy_params(offline_model, online_model)
            offline_model.save_parameters(temporary_model)
    #  train every 2 epoch
    if annealing > replay_start and epoch % 2 == 0:
        algorithm.train()
    total_reward[int(epoch) - 1] = cum_clipped_dr
