from gym_minigrid.envs.search import SearchEnv
from gym_minigrid.window import Window
from algorithm.reward_function import short_term_reward_function
from algorithm.reward_function import long_term_reward_function


class SimpleEnv(object):
    def __init__(self, display=True):
        super().__init__()
        self.display = display
        self.map = SearchEnv(20, 20, max_step=200, goals=50)
        self.window = None
        if self.display:
            self.window = Window('GYM_MiniGrid')
            self.window.reg_key_handler(self.key_handler)
            self.window.show(True)
        self.same_position = 0
        self.detect_rate = []
        self.step_count = []
        # state_memory
        self.old = None
        self.new = None

    def get_short_term_reward(self):
        return short_term_reward_function(self.old, self.new,
                                          self.map.check_history())

    def get_long_term_reward(self):
        find_keys, travel_area = self.map.reward()
        self.detect_rate.append(find_keys)
        return long_term_reward_function(find_keys, travel_area)

    def step(self, action):
        # Done completing task
        # done = 0
        # Turn left, turn right, move forward
        # left = 1
        # right = 2
        # forward = 3
        self.old = self.map.state()
        self.new, done = self.map.step(action)
        r = self.get_short_term_reward()
        if self.display is True:
            self.redraw()
        if done:
            self.step_count.append(self.map.step_count)
            r += self.get_long_term_reward()
        return self.old, self.new, float(r), done

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
        self.redraw()
