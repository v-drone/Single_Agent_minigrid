from config import *
import os
import numpy as np
import mxnet as mx
from model.simple_stack import SimpleStack
from utils import check_dir
from memory import Memory
from environments.SimpleEnv import SimpleEnv
from mxnet import gluon

if os.path.exists(summary):
    os.remove(summary)
ctx = mx.cpu()
for i in ["model_save", "data_save"]:
    check_dir(i)
# build models
online_model = SimpleStack()
offline_model = SimpleStack()
online_model.collect_params().initialize(mx.init.Normal(0.02), ctx=ctx)
offline_model.collect_params().initialize(mx.init.Normal(0.02), ctx=ctx)
offline_model.collect_params().zero_grad()
# create env
env = SimpleEnv(display=True)
env.reset_env()
memory_pool = Memory(memory_length)
annealing = 0
total_reward = np.zeros(num_episode)
eval_result = []
loss_func = gluon.loss.L2Loss()
trainer = gluon.Trainer(offline_model.collect_params(), 'adam',
                        {'learning_rate': lr})
env.reset_env()
finish = 0
cum_clipped_dr = 0
import pdb
pdb.set_trace()
