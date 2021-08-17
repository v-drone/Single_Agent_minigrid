from utils import create_input, translate_state
from environments.SimpleEnv import SimpleEnv
from model.simple_stack import SimpleStack
from mxnet import nd
import mxnet as mx

# build models
_ctx = mx.cpu()
# agent view
_view = 5
_map_size = 20
# action max
action_max = 3
# temporary files
_temporary_model = "./model_save/TEST-Copy1.params"
_model = SimpleStack()
_model.load_parameters(_temporary_model, ctx=_ctx)
env = SimpleEnv(display=True, agent_view=_view, map_size=_map_size)
env.reset_env()
for epoch in range(5):
    env.reset_env()
    done = 0
    while not done:
        data = create_input([translate_state(env.map.state())])
        data = [nd.array(i, ctx=_ctx) for i in data]
        action = _model(data)
        print(action)
        action = int(nd.argmax(action, axis=1).asnumpy()[0])
        old, new, reward, done = env.step(action)
