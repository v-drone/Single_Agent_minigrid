from utils import create_input, translate_state
import torch
from PIL import Image
from environments.MutilRoadEnv import SimpleEnv
from model.image_decoder import SimpleStack


def evaluate(ctx, model, env, rounds=5, print_action=False, save=None):
    env.reset_env()
    for epoch in range(rounds):
        env.reset_env()
        done = 0
        step = 0
        while not done:
            step += 1
            data = create_input([translate_state(env.map.state())])
            data = [torch.FloatTensor(i).to(ctx) for i in data]
            pred = model.forward(data)
            action = int(torch.argmax(pred).cpu().numpy())
            old, new, reward, done = env.step(action)
            if print_action:
                print(pred, reward, env.map.battery)
            if save is not None:
                img = Image.fromarray(env.map.render(), 'RGB')
                pred = [str(x)[0:5] for x in pred.detach().numpy().tolist()[0]]
                filename = "torch-" + str(epoch) + "-" + str(step) + "-" + str(
                    reward) + "-" + "_".join(pred) + ".jpg"
                img.save(save + "/" + filename)
    return env.detect_rate


if __name__ == '__main__':
    _agent_view = 5
    _map_size = 20
    _env = SimpleEnv(display=True, agent_view=_agent_view, map_size=_map_size)
    _ctx = torch.device("cpu")
    _model = SimpleStack()
    _model.load_state_dict(torch.load("./model_save/Torch.params", _ctx))
    evaluate(_ctx, _model, _env, rounds=10, print_action=True,
             save=None)
