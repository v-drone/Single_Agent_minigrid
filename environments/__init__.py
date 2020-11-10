import random
from gym_minigrid.envs.lavagap import LavaGapEnv
from gym_minigrid.envs.distshift import DistShiftEnv
from gym_minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper
from gym_minigrid.window import Window


class SimpleEnv(object):
    def __init__(self, seed=1500, agent_view=True):
        super().__init__()
        self.env = None
        self.window = None
        self.agent_view = agent_view
        self.seed = seed
        self.reset_env()

    def reset_env(self):
        """
        reset environment to the start point
        :return:
        """
        if random.randint(0, 1) == 1:
            size = random.randint(5, 7)
            self.env = DistShiftEnv(width=size, height=size)
            self.env.seed(self.seed)
        else:
            self.env = LavaGapEnv(size=random.randint(5, 7))
            self.env.seed(self.seed)
        self.env = RGBImgPartialObsWrapper(self.env)
        self.env = ImgObsWrapper(self.env)
        self.window = Window('MiniGrid')
        self.window.reg_key_handler(self.key_handler)
        # Blocking event loop
        self.window.show(block=True)

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

        # SpaceBar
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

    def get_location(self):
        """
        get location data
        :return: numpy.array
        """
        raise NotImplementedError

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        print('step=%s, reward=%.2f' % (self.env.step_count, reward))
        if done:
            print('done!')
            self.reset_env()
        else:
            self.redraw(obs)

    def redraw(self, img):
        if not self.agent_view:
            img = self.env.render('rgb_array', tile_size=64)
        self.window.show_img(img)


if __name__ == '__main__':
    env = SimpleEnv()
