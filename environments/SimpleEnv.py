from gym_minigrid.envs.simple2Dv2 import Simple2Dv2
from gym_minigrid.window import Window


class SimpleEnv(object):
    def __init__(self, display=False, agent_view=5, map_size=20, roads=1, max_step=100):
        super().__init__()
        self.display = display
        self.map = Simple2Dv2(map_size, map_size, agent_view=agent_view, roads=roads, max_step=max_step)
        self.window = None
        if self.display:
            self.window = Window('GYM_MiniGrid')
            self.window.reg_key_handler(self.key_handler)
            self.window.show(True)
        self.same_position = 0
        self.detect_rate = []
        self.step_count = []
        self.old = None
        self.new = None

    def short_term_reward(self):
        return self.new["reward"]

    def long_term_reward(self):
        road_detect = self.new["l_reward"]
        if self.map.battery <= 0:
            rate = 0.1
        else:
            rate = 2 - (self.map.step_count / self.map.max_steps)
        return road_detect * rate

    def step(self, action):
        # Turn left, turn right, move forward
        # forward = 0
        # left = 1
        # right = 2
        self.old = self.map.state()
        self.new, done = self.map.step(action)
        reward = self.short_term_reward()
        if self.display is True:
            self.redraw()
        if done:
            self.detect_rate.append(self.new["l_reward"])
            self.step_count.append(self.map.step_count)
            reward += self.long_term_reward()
        return self.old, self.new, reward, done

    def key_handler(self, event):
        print('pressed', event.key)
        if event.key == 'left':
            self.step(0)
            return
        if event.key == 'right':
            self.step(1)
            return
        if event.key == 'up':
            self.step(2)
            return

    def redraw(self):
        if self.window is not None:
            self.map.render('human')

    def reset_env(self):
        """
        reset environment to the start point
        :return:
        """
        self.map.reset()
        if self.display:
            self.redraw()
