from gym_minigrid.envs.simple2Dv1 import Simple2Dv1 as Environment
from gym_minigrid.window import Window
import numpy as np


class SimpleEnv(object):
    def __init__(self, display=False, agent_view=5, map_size=20, roads=1, max_step=100):
        super().__init__()
        self.sr = ""
        self.lr = ""
        self.display = display
        self.map = Environment(map_size, map_size, agent_view=agent_view, roads=roads, max_step=max_step)
        self.window = None
        if self.display:
            self.window = Window('GYM_MiniGrid')
            self.window.reg_key_handler(self.key_handler)
            self.window.show(True)
        self.detect_rate = []
        self.rewards = []
        self.step_count = []
        self._rewards = []
        self.bonus_rate = 0.005

    def short_term_reward(self, old, new):
        moving_bonus = 0
        if set([tuple(i) for i in old["history"].tolist()]) != set([tuple(i) for i in new["history"].tolist()]):
            moving_bonus += 10
        return (-2 + moving_bonus) * self.bonus_rate

    def long_term_reward(self, state):
        _extrinsic_reward = state["l_reward"]
        _extrinsic_reward = sum(_extrinsic_reward) / len(_extrinsic_reward)
        self.lr = "er = detect rate (0~1)"
        return _extrinsic_reward

    def step(self, action):
        # forward = 0
        # left = 1
        # right = 2
        old = self.map.state()
        new, done = self.map.step(action)
        reward = self.short_term_reward(old, new)
        if self.display is True:
            self.redraw()
        if done != 0:
            self.detect_rate.append(new["l_reward"])
            self.step_count.append(self.map.step_count)
            reward += self.long_term_reward(new)
            self._rewards.append(reward)
            self.rewards.append(np.mean(self._rewards))
        else:
            self._rewards.append(reward)

        return old, new, reward, done

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
        self._rewards = []
        if self.display:
            self.redraw()
