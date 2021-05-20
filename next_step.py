from config import *
import os
import mxnet as mx
from model import SimpleStack
from utils import check_dir
from memory import Memory
from algorithm.DQN import DQN
from environments.SimpleEnv import SimpleEnv

if os.path.exists(summary):
    os.remove(summary)
ctx = mx.gpu()
for i in ["model_save", "data_save"]:
    check_dir(i)
# build models
online_model = SimpleStack(agent_view, map_size)
online_model.collect_params().initialize(mx.init.Normal(0.02), ctx=ctx)
env = SimpleEnv(display=True)
env.reset_env()
# create pool
memory_pool = Memory(memory_length)
algorithm = DQN([online_model, online_model], ctx, lr, gamma, memory_pool,
                action_max, temporary_model, bz=1024)
while True:
    eps = 1
    action, by = algorithm.get_action(env.state(), eps)
    old, new, reward_get, finish, original_reward = env.step(action)
