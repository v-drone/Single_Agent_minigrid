import random
from gym_minigrid.envs.lavagap import LavaGapEnv
from gym_minigrid.envs.distshift import DistShiftEnv
from gym_minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper
from gym_minigrid.window import Window


def get_env(seed=1500):
    """
    get a environment at the start point
    :return:
    """
    if random.randint(0, 1) == 1:
        size = random.randint(5, 7)
        env = DistShiftEnv(width=size, height=size)
        env.seed(seed)
    else:
        env = LavaGapEnv(size=random.randint(5, 7))
        env.seed(seed)
    env.reset()
    return env


class Env1(DistShiftEnv):
    def __init__(self, size, seed=1500):
        super().__init__(width=size, height=size)
        self.seed = seed


def key_handler(event):
    print('pressed', event.key)

    if event.key == 'escape':
        window.close()
        return

    if event.key == 'backspace':
        reset()
        return

    if event.key == 'left':
        step(env.actions.left)
        return
    if event.key == 'right':
        step(env.actions.right)
        return
    if event.key == 'up':
        step(env.actions.forward)
        return

    # Spacebar
    if event.key == ' ':
        step(env.actions.toggle)
        return
    if event.key == 'pageup':
        step(env.actions.pickup)
        return
    if event.key == 'pagedown':
        step(env.actions.drop)
        return

    if event.key == 'enter':
        step(env.actions.done)
        return


if __name__ == '__main__':
    env = SimpleEnv()
