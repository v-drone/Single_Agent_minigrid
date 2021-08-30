from utils import create_input, translate_state
from mxnet import nd
import mxnet as mx
from environments.SimpleEnv import SimpleEnv
from model.simple_stack import SimpleStack
from PIL import Image


def evaluate(ctx, model, env, rounds=5, print_action=False, save=None):
    env.reset_env()
    for epoch in range(rounds):
        env.reset_env()
        done = 0
        step = 0
        while not done:
            step += 1
            data = create_input([translate_state(env.map.state())])
            data = [nd.array(i, ctx=ctx) for i in data]
            pred = model(data)
            action = int(nd.argmax(pred, axis=1).asnumpy()[0])
            old, new, reward, done = env.step(action)
            if print_action:
                print(pred, reward, env.map.battery)
            if save is not None:
                img = Image.fromarray(env.map.render(), 'RGB')
                pred = [str(x)[0:5] for x in pred.asnumpy().tolist()[0]]
                filename = str(epoch) + "-" + str(step) + "-" + str(
                    reward) + "-" + "_".join(pred) + ".jpg"
                img.save(save + "/" + filename)
    return env.detect_rate


if __name__ == '__main__':
    _agent_view = 5
    _map_size = 20
    _env = SimpleEnv(display=True, agent_view=_agent_view, map_size=_map_size)
    _ctx = mx.cpu()
    _model = SimpleStack()
    _model.load_parameters("./model_save/MXNET.params", _ctx)
    evaluate(_ctx, _model, _env, rounds=100, print_action=True,
             save="./data_save/")
