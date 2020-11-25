from environments.SimpleEnv import SimpleEnv
import mxnet as mx
from model import Stack
from utils import translate_state
from mxnet import nd
import numpy as np

env = SimpleEnv(display=False)
env.reset_env()
state = translate_state(env.state())
agent_in = nd.array(np.array([state[0], state[0]]))
agent_in = nd.expand_dims(agent_in, 1)
whole_in = nd.array(np.array([state[1], state[1]]))
whole_in = nd.expand_dims(whole_in, 1)
location = nd.array(np.array([state[2], state[2]]))
attitude = nd.array(np.array([state[3], state[3]]))
attitude = nd.expand_dims(attitude, 1)
income = nd.concat(*[agent_in.flatten(), whole_in.flatten(), location.flatten(), attitude.flatten()])
stack = Stack()
stack.collect_params().initialize(ctx=mx.cpu())
x = stack(income)
