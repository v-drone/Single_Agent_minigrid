from utils import create_input, translate_state
from mxnet import nd
import mxnet as mx
from environments.SimpleEnv import SimpleEnv
from model.simple_stack import SimpleStack


def evaluate(ctx, model, env, rounds=5, print_action=False):
    env.reset_env()
    for epoch in range(rounds):
        env.reset_env()
        done = 0
        while not done:
            data = create_input([translate_state(env.map.state())])
            data = [nd.array(i, ctx=ctx) for i in data]
            _action = model(data)
            action = int(nd.argmax(_action, axis=1).asnumpy()[0])
            old, new, reward_get, done = env.step(action)
            if print_action:
                print(_action, reward_get, env.map.battery)
                import pdb
                pdb.set_trace()
    return env.detect_rate


if __name__ == '__main__':
    _agent_view = 7
    _map_size = 20
    _env = SimpleEnv(display=True, agent_view=_agent_view, map_size=_map_size)
    _ctx = mx.cpu()
    _model = SimpleStack()
    _model.load_parameters("./model_save/MXNET.params", _ctx)
    evaluate(_ctx, _model, _env, print_action=False)
