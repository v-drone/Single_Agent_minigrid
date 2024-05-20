import gymnasium as gym
import csv
import os


class RecordingWrapper(gym.Wrapper):
    def __init__(self, env, filename='record.csv'):
        super().__init__(env)
        self.filename = filename
        self.data = []
        self.episode_count = 0
        self.step_count = 0
        with open(self.filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Episode', 'Step', 'Action', 'Reward'])

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.record_step(action, reward)
        if terminated or truncated:
            self.write_to_file()
            self.data.clear()
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        if self.data:
            self.write_to_file()
            self.data.clear()
        self.episode_count += 1
        self.step_count = 0
        observation = self.env.reset(**kwargs)
        return observation

    def record_step(self, action, reward):
        self.step_count += 1
        self.data.append([self.episode_count, self.step_count, action, reward])

    def write_to_file(self):
        with open(self.filename, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(self.data)
