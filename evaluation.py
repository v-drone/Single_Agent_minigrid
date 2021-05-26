from utils import create_input, translate_state
from environments.SimpleEnv import SimpleEnv
from model.simple_stack import SimpleStack
from mxnet import nd
from config import *
import mxnet as mx
import pandas as pd


def evaluate(model, test_round, ctx):
    env = SimpleEnv(display=True)
    env.reset_env()
    for epoch in range(test_round):
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
    _ctx = mx.gpu()
    _model = SimpleStack(agent_view, map_size)
    _model.collect_params().initialize(mx.init.Normal(0.02), ctx=_ctx)
    _model.load_parameters(temporary_model)
    detect_rate = evaluate(_model, 100, _ctx)
    detect_rate = pd.DataFrame(detect_rate)
    detect_rate.columns = ["detect_rate"]
    detect_rate["round"] = detect_rate.index
    detect_rate[["round", "detect_rate"]].to_csv("./eval.csv")
