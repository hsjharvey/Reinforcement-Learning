# -*- coding:utf-8 -*-
import tensorflow as tf


class Config:
    device = tf.device('/gpu:0')

    def __init__(self):
        # environment parameters
        self.input_dim = (1, 4)  # input feature dimension
        self.action_dim = 2  # agent action dimension (generally consider part of the environment)

        self.episodes = 1000
        self.steps = 200  # note that OpenAI gym has max environment steps (e.g. max_step = 200 for CartPole)

        # general RL agent parameters
        self.batch_size = 32  # size for each training
        self.replay_buffer_size = 100  # must > batch size
        assert self.replay_buffer_size >= self.batch_size

        self.stop_explore = 300

        self.discount_rate = 0.99  # constant, a super important parameter

        # neural network parameters
        self.weights_initializer = tf.keras.initializers.RandomNormal(mean=0.0,
                                                                      stddev=0.0001,
                                                                      seed=None)

        self.optimizer = tf.keras.optimizers.RMSprop(lr=1e-2,  # learning rate
                                                     clipnorm=1.0)  # gradient clipping

        self.keras_checkpoint = tf.keras.callbacks.ModelCheckpoint('../saved_network_models/harvey.model',
                                                                   save_weights_only=True,
                                                                   mode='auto')

        # categorical DQN parameters
        self.Categorical_Vmin = 0
        self.Categorical_Vmax = 10
        self.Categorical_n_atoms = 50

        # Quantile Regression DQN parameters
        self.num_quantiles = 20
