# -*- coding:utf-8 -*-
import tensorflow as tf


class Config:
    device = tf.device('/gpu:0')

    def __init__(self):
        # categorical DQN parameters
        self.Categorical_Vmin = -10
        self.Categorical_Vmax = 10
        self.Categorical_n_atoms = 51

        # environment
        self.input_dim = (1, 4)
        self.action_dim = 2
