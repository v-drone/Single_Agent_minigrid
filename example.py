from config import *
import os
import numpy as np
import mxnet as mx
from model.simple_stack import SimpleStack
from utils import check_dir
from memory import Memory
from environments.SimpleEnv import SimpleEnv
from utils import create_input, translate_state
from mxnet import gluon, nd
if os.path.exists(summary):
    os.remove(summary)
ctx = mx.cpu()
for i in ["model_save", "data_save"]:
    check_dir(i)
# build models
model = SimpleStack()
print(model)
model.load_parameters("./model_save/MXNET_view_only.params.best")
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
        action = int(nd.argmax(model(data), axis=1).asnumpy()[0])
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
