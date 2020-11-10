import random
from gym_minigrid.envs.lavagap import LavaGapEnv
from gym_minigrid.envs.distshift import DistShiftEnv
from gym_minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper
from gym_minigrid.window import Window


class SimpleEnv(object):
    def __init__(self, display=True):
        super().__init__()
        self.display = display
        self.env = None
        self.window = Window('GYM_MiniGrid')
        self.window.reg_key_handler(self.key_handler)
        self.reset_env()
        self.window.show(True)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        print('step=%s, reward=%.2f' % (self.env.step_count, reward))
        if done:
            print('done!')
            self.reset_env()
        else:
            if self.display is True:
                self.redraw()

    def key_handler(self, event):
        print('pressed', event.key)
        if event.key == 'escape':
            self.window.close()
            return
        if event.key == 'backspace':
            self.reset_env()
            return
        if event.key == 'left':
            self.step(self.env.actions.left)
            return
        if event.key == 'right':
            self.step(self.env.actions.right)
            return
        if event.key == 'up':
            self.step(self.env.actions.forward)
            return
        if event.key == ' ':
            self.step(self.env.actions.toggle)
            return
        if event.key == 'pageup':
            self.step(self.env.actions.pickup)
            return
        if event.key == 'pagedown':
            self.step(self.env.actions.drop)
            return

        if event.key == 'enter':
            self.step(self.env.actions.done)
            return

    def redraw(self):
        if self.window is not None:
            img = self.env.render('rgb_array', tile_size=32)
            self.window.show_img(img)

    def reset_env(self):
        """
        reset environment to the start point
        :return:
        """
        size = random.randint(10, 15)
        if random.randint(1, 2) >= 1:
            self.env = LavaGapEnv(size)
        else:
            self.env = DistShiftEnv(width=size, height=size)
        self.env.reset()
        if self.display:
            self.env = RGBImgPartialObsWrapper(self.env)
            self.env = ImgObsWrapper(self.env)
        self.redraw()

    def state(self):
        raise NotImplementedError

    def collision_checking(self):
        raise NotImplementedError

    def go_forward(self):
        """
        forward d unit
        :return:
        """
        raise NotImplementedError

    def get_attitude(self):
        """
        get yaw state of drones
        :return:
        """
        raise NotImplementedError

    def turn(self, angle):
        """
        turn angle degree
        :param angle: float
        :return:
        """
        raise NotImplementedError


if __name__ == '__main__':
    env = SimpleEnv()
