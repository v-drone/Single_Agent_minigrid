import time

from utils import create_input, translate_state
from environments.SimpleEnv import SimpleEnv
from model.simple_stack import SimpleStack
from mxnet import nd
import mxnet as mx
import pandas as pd
import time


def evaluate(ctx, model, agent_view=7, test_round=5, display=False):
    env = SimpleEnv(display=display, agent_view=agent_view)
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
    _ctx = mx.cpu()
    # training cases
    order = "TEST"
    # agent view
    agent_view = 7
    map_size = 10
    # action max
    action_max = 3
    # learning rate
    model_save = "./model_save/"
    lr = 0.005
    num_episode = 1000000
    # start play
    replay_start = 10000
    # update step
    update_step = 1000
    # gamma in q-loss calculation
    gamma = 0.99
    # memory pool size
    memory_length = 100000
    # file to save train log
    summary = "./{}_Reward.csv".format(order)
    eval_statistics = "./{}_CSV.csv".format(order)
    # the number of step it take to linearly anneal the epsilon to it min value
    annealing_end = 200000
    # min level of stochastically of policy (epsilon)-greedy
    epsilon_min = 0.2
    # temporary files
    temporary_model = "./{}/{}.params".format(model_save, order)
    temporary_pool = "./{}/{}.pool".format(model_save, order)
    _model = SimpleStack(agent_view, map_size)
    _model.load_parameters(temporary_model, ctx=_ctx)
    detect_rate = evaluate(_ctx, _model, agent_view, 100, True)
    detect_rate = pd.DataFrame(detect_rate)
    detect_rate.columns = ["detect_rate"]
    detect_rate["round"] = detect_rate.index
    detect_rate[["round", "detect_rate"]].to_csv("./eval.csv")
