from utils import create_input, translate_state
from model.simple_stack_torch import SimpleStack
from environments.SimpleEnv import SimpleEnv
from config import *
import torch


def evaluate(ctx, model, env, rounds=5, print_action=False):
    env.reset_env()
    for epoch in range(rounds):
        env.reset_env()
        done = 0
        while not done:
            data = create_input([translate_state(env.map.state())])
            data = [torch.FloatTensor(i).to(ctx) for i in data]
            action = model.forward(data)
            if print_action:
                print(action)
            action = int(torch.argmax(action).cpu().numpy())
            old, new, reward_get, done = env.step(action)
    return env.detect_rate


if __name__ == '__main__':
    _env = SimpleEnv(display=True, agent_view=agent_view, map_size=map_size)
    _ctx = torch.device("cpu")
    _model = SimpleStack()
    _model.load_state_dict(torch.load(temporary_model, _ctx))
    evaluate(_ctx, _model, _env, print_action=True)
