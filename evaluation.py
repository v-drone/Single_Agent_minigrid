from utils import create_input, translate_state
from environments.SimpleEnv import SimpleEnv
from mxnet import nd


def evaluate(model, test_round, ctx):
    env = SimpleEnv(display=False)
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
