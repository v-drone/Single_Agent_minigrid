from utils import create_input, translate_state
import torch


def evaluate(ctx, model, env, rounds=5):
    env.reset_env()
    for epoch in range(rounds):
        env.reset_env()
        done = 0
        while not done:
            data = create_input([translate_state(env.map.state())])
            data = [torch.FloatTensor(i).to(ctx) for i in data]
            action = model.forward(data)
            action = int(torch.argmax(action).cpu().numpy())
            old, new, reward_get, done = env.step(action)
    return env.detect_rate
