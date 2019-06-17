# -*- coding:utf-8 -*-
from utils import policies, config, replay_fn
from network import neural_network
import numpy as np
import tensorflow as tf
from collections import deque
import gym
import time


class CategoricalDQNAgent:
    def __init__(self, config):
        self.config = config

        self.input_dim = config.input_dim  # neural network input dimension

        self.quantile_weights = 1.0 / float(config.num_quantiles)
        self.cum_density = np.arange(2 * np.arange(config.num_quantiles) + 1) / (2.0 * config.num_quantiles)

        self.atoms = np.linspace(
            config.Categorical_Vmin,
            config.Categorical_Vmax,
            config.Categorical_n_atoms,
        )  # Z

        self.envs = None
        self.actor_network = None
        self.target_network = None

        self.total_steps = 0
        self.episodes = config.episodes
        self.steps = config.steps
        self.batch_size = config.batch_size

        self.replay_buffer_size = config.replay_buffer_size
        self.replay_buffer = deque()

    def transition(self):
        for each_ep in range(self.episodes):
            current_state = self.envs.reset()

            for step in range(self.steps):
                self.total_steps += 1

                

    def train_by_replay(self):
        pass
