import gym
import numpy as np

from gym import spaces

class RandMDP(gym.Env):
    def __init__(self, seed=0, semi_rand=False):
        super(RandMDP, self).__init__()
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=0, high=1, shape=(), dtype=np.float32)
        self.time = 0
        self.rng = np.random.RandomState(seed)
        self.obs = self.rng.rand()
        if semi_rand:
            self.kinks = np.array([[1/3, 2/3], [1/3, 2/3]])
            self.values = np.array([[0.35*self.rng.rand(), 0.65 + 0.35*self.rng.rand(), 0.35*self.rng.rand(), 0.65 + 0.35*self.rng.rand()],
                                   [0.35*self.rng.rand(), 0.65 + 0.35*self.rng.rand(), 0.35*self.rng.rand(), 0.65 + 0.35*self.rng.rand()]])
        else:
            self.kinks = self.rng.rand(2, 2)
            self.kinks.sort(axis=1)
            self.values = self.rng.rand(2, 4)

    def step(self, action):
        self.time += 1
        kink = self.kinks[action]
        value = self.values[action]

        if self.obs < kink[0]:
            self.obs = value[0] + (value[1]-value[0])/kink[0]*self.obs
        elif self.obs >= kink[0] and self.obs < kink[1]:
            self.obs = value[1] + (value[2] - value[1])/(kink[1]-kink[0])*(self.obs - kink[0])
        else:
            self.obs = value[2] + (value[3] - value[2])/(1 - kink[1])*(self.obs - kink[1])
        assert 0 <= self.obs <= 1

        return self.obs, self.obs, (self.time >= 10), {}

    def reset(self):
        self.obs = self.rng.random.rand()
        return self.obs