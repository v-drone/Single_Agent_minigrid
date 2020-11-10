import random
from gym_minigrid.envs.lavagap import LavaGapEnv
from gym_minigrid.envs.distshift import DistShiftEnv
from gym_minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper
from gym_minigrid.window import Window


class Env1(DistShiftEnv):
    def __init__(self, size, seed=1500):
        super().__init__(size)
        self.seed = seed


class Env2(LavaGapEnv):
    def __init__(self, size, seed=1500):
        super().__init__(size)
        self.seed = seed


def redraw():
    img = env.render('rgb_array', tile_size=32)
    window.show_img(img)


def get_env(seed=1500):
    """
    get a environment at the start point
    :return:
    """
    if random.randint(0, 1) == 1:
        size = random.randint(5, 7)
        _env = Env1(size, seed=seed)
    else:
        _env = Env2(size=random.randint(5, 7), seed=seed)
    _env.reset()
    return _env


def step_and_display(action, _env):
    obs, reward, done, info = _env.step(action)
    print('step=%s, reward=%.2f' % (_env.step_count, reward))
    if done:
        print('done!')
        _env.reset()
    else:
        redraw()


def key_handler(event):
    print('pressed', event.key)
    if event.key == 'escape':
        window.close()
        return

    if event.key == 'backspace':
        env.reset()
        return

    if event.key == 'left':
        step_and_display(env.actions.left, env)
        return
    if event.key == 'right':
        step_and_display(env.actions.right, env)
        return
    if event.key == 'up':
        step_and_display(env.actions.forward, env)
        return

    if event.key == ' ':
        step_and_display(env.actions.toggle, env)
        return
    if event.key == 'pageup':
        step_and_display(env.actions.pickup, env)
        return
    if event.key == 'pagedown':
        step_and_display(env.actions.drop, env)
        return

    if event.key == 'enter':
        step_and_display(env.actions.done, env)
        return


if __name__ == '__main__':
    env = get_env()
    env = RGBImgPartialObsWrapper(env)
    env = ImgObsWrapper(env)
    window = Window('GYM_MiniGrid')
    window.reg_key_handler(key_handler)
    window.show(block=True)
