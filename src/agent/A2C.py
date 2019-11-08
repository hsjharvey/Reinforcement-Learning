# -*- coding:utf-8 -*-
from src.utils import *
import numpy as np
import tensorflow as tf
from collections import deque
import gym


class A2Cgent:
    def __init__(self, config, base_network):
        self.base_network = base_network
        self.config = config

        self.input_dim = config.input_dim  # neural network input dimension

        self.envs = None
        self.actor = self.base_network.nn_model()
        self.critic = tf.keras.models.clone_model(self.actor)
        self.critic.set_weights(self.actor.get_weights())

        self.total_steps = 0
        self.episodes = config.episodes
        self.steps = config.steps
        self.batch_size = config.batch_size

        self.replay_buffer_size = config.replay_buffer_size
        self.replay_buffer = deque()

        self.keras_check = config.keras_checkpoint

        self.check = 0
        self.best_max = 0