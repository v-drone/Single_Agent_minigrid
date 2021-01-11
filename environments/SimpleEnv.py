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
        self.finish = []
        self.total_reward = []
        self.total_step_count = []
        self.current_show_reward = []
        self.current_step_count = 0

    def step(self, action):
        success_text = None
        # Turn left, turn right, move forward
        # forward = 0
        # left = 1
        old = self.state()
        if action == 1:
            obs, original_reward, done, info = self.env.step(0)
        else:
            obs, original_reward, done, info = self.env.step(2)
        new = self.state()
        if np.equal(old["relative_position"], new["relative_position"]).all():
            self.same_position += 1
        else:
            self.same_position = -1
        reward_get = reward_function(old, new, original_reward, self.env.step_count, self.same_position)
        self.current_show_reward.append(sum(reward_get))
        self.current_step_count += 1
        if done:
            self.total_reward.append(sum(self.current_show_reward))
            self.total_step_count.append(self.current_step_count)
            self.reset_env()
            finish = 1
            if original_reward > 0:
                self.finish.append(1)
            else:
                self.finish.append(0)
            if len(self.finish) > 50:
                success_text = "success rate last 50 %f, avg return %f; success rate total %f, avg return total %f" % (
                    sum(self.finish[-50:]) / min(len(self.finish), 50), sum(self.total_reward[-50:]) / sum(self.total_step_count[-50:]),
                    sum(self.finish) / len(self.finish), sum(self.total_reward) / sum(self.total_step_count))
        else:
            if self.display is True:
                self.redraw()
            finish = 0
        return old, new, reward_get, finish, success_text, original_reward

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
        size = 10
        _ = 0
        # _ = random.randint(-2, 2)
        if _ > 1:
            self.env = LavaGapEnv(size)
        elif _ < -1:
            self.env = DistShiftEnv(width=size, height=size)
        else:
            self.env = EmptyEnv(width=size, height=size)
        self.env.reset()
        if self.display and self.window:
            self.window.close()
        self.current_show_reward = []
        self.current_step_count = 0
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
