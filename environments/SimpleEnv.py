import random
import numpy as np
from utils import to_numpy
from gym_minigrid.envs.lavagap import LavaGapEnv
from gym_minigrid.envs.distshift import DistShiftEnv
from gym_minigrid.envs.empty import EmptyEnv
from gym_minigrid.window import Window
from algorithm.reward_function import reward_function


class SimpleEnv(object):
    def __init__(self, display=True):
        super().__init__()
        self.display = display
        self.env = None
        self.window = None
        self.reset_env()
        if self.display:
            self.window = Window('GYM_MiniGrid')
            self.window.reg_key_handler(self.key_handler)
            self.window.show(True)
        self.same_position = 0

    def step(self, action):
        # Turn left, turn right, move forward
        # left = 0
        # right = 1
        # forward = 2
        old = self.state()
        obs, reward_get, done, info = self.env.step(action)
        new = self.state()
        if done == 1 and reward_get == 0:
            reward_get = -0.1
        if np.equal(old["relative_position"], new["relative_position"]).all():
            self.same_position += 1
        else:
            self.same_position = 0
        reward_get = reward_function(old, new, reward_get, self.env.step_count, self.same_position)
        _ = 'step=%s, reward=%.2f, action=%d' % (self.env.step_count, sum(reward_get), action)
        if sum(reward_get) > 10:
            _ = _ + "      ***********"
        elif sum(reward_get) > 0:
            _ = _ + "      *"
        print(_)
        if done:
            print('done!')
            self.reset_env()
            finish = 1
        else:
            if self.display is True:
                self.redraw()
            finish = 0
        return old, new, reward_get, finish

    def key_handler(self, event):
        print('pressed', event.key)
        if event.key == 'escape':
            self.window.close()
            return
        if event.key == 'backspace':
            self.reset_env()
            return
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
            self.env.render('human')
            # self.window.show_img(img)

    def reset_env(self):
        """
        reset environment to the start point
        :return:
        """
        # size = random.randint(9, 15)
        size = 10
        _ = random.randint(-2, 2)
        if _ > 1:
            self.env = LavaGapEnv(size)
        elif _ < -1:
            self.env = DistShiftEnv(width=size, height=size)
        else:
            self.env = EmptyEnv(width=size, height=size)
        self.env.reset()
        if self.display and self.window:
            self.window.close()

    def state(self):
        grid, vis_mask = self.env.gen_obs_grid()
        agent = np.array([self.env.agent_pos[0], self.env.agent_pos[1], self.env.agent_dir])
        view = to_numpy(grid, [3, 6, self.env.agent_dir], vis_mask)[::-1]
        view = np.flip(view[::-1], 1)
        whole_map = to_numpy(self.env.grid, agent, None).T
        goal = np.array(self.env.goal_pos)
        relative_position = agent[0:2] - goal
        data = {
            "agent_view": view,
            "whole_map": whole_map,
            "relative_position": relative_position,
            "attitude": self.env.agent_dir
        }
        # self.redraw()
        return data


if __name__ == '__main__':
    env = SimpleEnv()
