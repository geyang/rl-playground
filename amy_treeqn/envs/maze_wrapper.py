import gym
import numpy as np
from gym import spaces


class MazeWrapper(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.observation_space = spaces.Box(
            low=np.zeros((2, self.width, self.height)),
            high=np.ones((2, self.width, self.height)))

    def get_obs(self, obs):
        return np.concatenate([obs['img'], obs['goal_img']], axis=0)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if reward == 0:
            done = True
        return self.get_obs(obs), reward, done, info
    
    def reset(self):
        obs = self.env.reset()
        return self.get_obs(obs)