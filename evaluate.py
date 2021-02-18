import matplotlib.pyplot as plt
import os
import numpy as np
import mxnet as mx
from model import SimpleStack
from utils import check_dir
from config import *
from algorithm.DQN import DQN
from environments.SimpleEnv import SimpleEnv
from sys import argv

ctx = mx.gpu()
for i in ["model_save", "data_save"]:
    check_dir(i)
# build models
online_model = SimpleStack(7, 10)
offline_model = SimpleStack(7, 10)
if os.path.exists(temporary_model):
    online_model.load_parameters(temporary_model, ctx=ctx)
    offline_model.load_parameters(temporary_model, ctx=ctx)
    print("load model")
env = SimpleEnv(display=False)
algorithm = DQN([online_model, offline_model], ctx, lr, gamma, None, action_max, temporary_model, bz=1024)
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
for epoch in range(10000):
    _epoch += 1
    env.reset_env()
    finish = 0
    cum_clipped_reward = 0
    while not finish:
        eps = 0
        action, by = algorithm.get_action(env.state(), eps)
        old, new, reward_get, finish, original_reward = env.step(action)
        cum_clipped_reward += original_reward
        if finish:
            sr_50 = sum(env.finish[-50:]) / min(len(env.finish), 50)
            ar_50 = sum(env.total_reward[-50:]) / sum(env.total_step_count[-50:])
            sr_all = sum(env.finish) / len(env.finish)
            ar_all = sum(env.total_reward) / sum(env.total_step_count)
            text = "success rate last 50 %f, avg return %f; success rate total %f, avg return total %f" % (
            sr_50, ar_50, sr_all, ar_all)
            if epoch % 100 == 0:
                print(text + "; %f" % eps)
        tot_reward[int(epoch) - 1] = cum_clipped_reward
# Moving average bandwidth
bandwidth = 1000
total_rew = np.zeros(int(_epoch) - bandwidth)
for i in range(int(_epoch) - bandwidth):
    total_rew[i] = np.sum(tot_reward[i:i + bandwidth]) / bandwidth
t = np.arange(int(_epoch) - bandwidth)
belplt = plt.plot(t, total_rew[0:int(_epoch) - bandwidth], "r", label="Return")
# handles[likplt,belplt])
plt.legend()
print('Running after %d number of episodes' % _epoch)
plt.xlabel("Number of episode")
plt.ylabel("Average Reward per episode")
plt.savefig("./default_reward_function_eval_%s.jpg" % argv[1])
np.save("default_%s_eval.array" % argv[1], total_rew)
