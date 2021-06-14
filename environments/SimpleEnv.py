from gym_minigrid.envs.search import SearchEnv
from gym_minigrid.window import Window
from algorithm.reward_function import short_term_reward_function
from algorithm.reward_function import long_term_reward_function


class SimpleEnv(object):
    def __init__(self, display=False):
        super().__init__()
        self.display = display
        self.map = SearchEnv(20, 20, agent_view_size=7)
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
        reward = short_term_reward_function(self.old, self.new,
                                            self.map.check_history())
        rate = self.map.step_count / self.map.max_steps
        if rate < 0.3:
            rate = 2
        elif 0.3 < rate < 0.6:
            rate = 1
        else:
            rate = 0.5
        return rate * reward

    def get_long_term_reward(self):
        find_keys, travel_area = self.map.reward()
        self.detect_rate.append(find_keys)
        rate = self.map.step_count / self.map.max_steps
        if rate < 0.3:
            rate = 2
        elif 0.3 < rate < 0.6:
            rate = 1
        else:
            rate = 0.5
        return rate * long_term_reward_function(find_keys, travel_area)

    def step(self, action):
        # Turn left, turn right, move forward
        # forward = 0
        # left = 1
        # right = 2
        self.old = self.map.state()
        self.new, done = self.map.step(action)
        reward = self.get_short_term_reward()
        if self.display is True:
            self.redraw()
        if done:
            self.step_count.append(self.map.step_count)
            reward += 10 * self.get_long_term_reward()
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
