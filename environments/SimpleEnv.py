import random
import numpy as np
from utils import to_numpy, replace_self
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
        self.success = 0
        self.all = 0
        self.total_return = 0

    def step(self, action):
        success_text = None
        # Turn left, turn right, move forward
        # left = 0
        # right = 1
        # forward = 2
        old = self.state()
        obs, original_get, done, info = self.env.step(action)
        new = self.state()
        reward_get = reward_function(old, new, original_get, self.env.step_count, self.same_position)
        self.total_return += sum(reward_get)
        if np.equal(old["relative_position"], new["relative_position"]).all():
            self.same_position += 1
        else:
            self.same_position = 0
        if done:
            self.reset_env()
            finish = 1
            self.all += 1
            if original_get > 0:
                self.success += 1
            success_text = "success rate %f, avg return %f" % (self.success / self.all, self.total_return / self.all)
        else:
            if self.display is True:
                self.redraw()
            finish = 0
        text = 'step=%s, reward=%.2f, action=%d' % (self.env.step_count, sum(reward_get), action)
        if sum(reward_get) > 10:
            text = text + "      ***********"
        elif sum(reward_get) > 0:
            text = text + "      *"
        return old, new, reward_get, finish, text, success_text

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
        size = 7
        # _ = random.randint(-2, 2)
        _ = 0
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
        precision = 10
        attitude = self.env.agent_dir
        agent = np.array([self.env.agent_pos[0], self.env.agent_pos[1], self.env.agent_dir])
        # view = to_numpy(grid, [3, 6, self.env.agent_dir], vis_mask)[::-1]
        # view = np.flip(view[::-1], 1)
        # view = replace_self(view, attitude)
        view = self.env.gen_obs()["image"]

        view = self.env.get_obs_render(view, precision)
        whole_map = self.env.grid.render(precision, (agent[0], agent[1]), agent[2])

        # whole_map = to_numpy(self.env.grid, agent, None).T
        # whole_map = replace_self(whole_map, attitude)
        goal = np.array(self.env.goal_pos)
        relative_position = goal - agent[0:2]
        data = {
            "agent_view": view,
            "whole_map": whole_map,
            "relative_position": relative_position,
            "attitude": attitude
        }
        return data


if __name__ == '__main__':
    env = SimpleEnv()
