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
        self.success = []
        self.all = 0
        self.total_return = []
        self.total_steps = []
        self.this_turn = []
        self.this_steps = 0

    def step(self, action):
        success_text = None
        # Turn left, turn right, move forward
        # forward = 0
        # left = 1
        old = self.state()
        if action == 1:
            obs, original_get, done, info = self.env.step(0)
        else:
            obs, original_get, done, info = self.env.step(2)
        new = self.state()
        reward_get = reward_function(old, new, original_get, self.env.step_count, self.same_position)
        self.this_turn.append(reward_get)
        if np.equal(old["relative_position"], new["relative_position"]).all():
            self.same_position += 1
        else:
            self.same_position = -3
        self.this_steps += 1
        if done:
            self.total_return.append(sum(self.this_turn))
            self.total_steps.append(self.this_steps)
            self.reset_env()
            finish = 1
            self.all += 1
            if original_get > 0:
                self.success.append(1)
            else:
                self.success.append(0)
            success_text = "success rate last 50 %f, avg return %f; total %f, avg return %f" % (
                sum(self.success[-50:]) / min(self.all, 50), sum(self.total_return[-50:]) / sum(self.total_steps[-50:]),
                sum(self.success) / self.all, sum(self.total_return) / sum(self.total_steps))
        else:
            if self.display is True:
                self.redraw()
            finish = 0
        text = 'step=%s, reward=%.2f, action=%d' % (self.env.step_count, reward_get, action)
        if reward_get > 10:
            text = text + "      ***********"
        elif reward_get > 0:
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
        self.this_turn = []
        self.this_steps = 0
        return self.state()

    def state(self):
        precision = 10
        attitude = self.env.agent_dir
        grid, vis_mask = self.env.gen_obs_grid()
        agent = np.array([self.env.agent_pos[0], self.env.agent_pos[1], self.env.agent_dir])
        view = to_numpy(grid, [3, 6, self.env.agent_dir], vis_mask)[::-1]
        view = np.flip(view[::-1], 1)
        # view = self.env.gen_obs()["image"]
        # view = self.env.get_obs_render(view, precision)
        # whole_map = self.env.grid.render(precision, (agent[0], agent[1]), agent[2])
        whole_map = to_numpy(self.env.grid, agent, None).T
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
