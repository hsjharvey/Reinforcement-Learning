# -*- coding:utf-8 -*-
import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import scipy.stats as sts


class fiveStateMDP(gym.Env):
    def __init__(self):
        self.observation_space = spaces.Discrete(1)  # one state
        self.action_space = spaces.Discrete(2)  # two actions
        self.cum_reward = 0.0
        self.game_type = None

        self.seed()
        self.viewer = None
        self.state = int(0)

        self.steps_beyond_done = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        if action == 0:
            reward = sts.expon(loc=0, scale=1).rvs()
        elif action == 1:
            reward = - sts.expon(loc=-1.85, scale=1).rvs()

        self.cum_reward += reward

        done = (self.cum_reward > 100000)  # game termination condition, to be modified
        done = bool(done)

        return self.state, reward, done, {}

    def reset(self):
        self.state = 0
        self.steps_beyond_done = None
        return self.state

    def render(self, mode='human'):
        pass

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
