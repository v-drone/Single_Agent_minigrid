from utils import create_input, translate_state
from environments.SimpleEnv import SimpleEnv
from model.simple_stack import SimpleStack
from mxnet import nd
import mxnet as mx
import pandas as pd


def evaluate(ctx, model, agent_view=7, map_size=20, rounds=5, display=False):
    env = SimpleEnv(display=display, agent_view=agent_view, map_size=map_size)
    env.reset_env()
    for epoch in range(rounds):
        env.reset_env()
        done = 0
        while not done:
            data = create_input([translate_state(env.map.state())], ctx)
            action = model(data)
            action = int(nd.argmax(action, axis=1).asnumpy()[0])
            old, new, reward, done = env.step(action)
    return env.detect_rate


if __name__ == '__main__':
    # build models
    _ctx = mx.cpu()
    _order = "TEST"
    # agent view
    _view = 7
    _map_size = 20
    # action max
    action_max = 3
    # temporary files
    _model_save = "./model_save/"
    _temporary_model = "./{}/{}.params".format(_model_save, _order)
    _model = SimpleStack()
    _model.load_parameters(_temporary_model, ctx=_ctx)
    detect_rate = evaluate(_ctx, _model, _view, _map_size, 100, True)
    detect_rate = pd.DataFrame(detect_rate)
    detect_rate.columns = ["detect_rate"]
    detect_rate["round"] = detect_rate.index
    detect_rate[["round", "detect_rate"]].to_csv("./eval.csv")
