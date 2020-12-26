# -*- coding:utf-8 -*-
import tensorflow as tf


class Config:
    def __init__(self):
        # environment parameters
        self.input_dim = (1, 4)  # input feature dimension
        self.action_dim = 2  # agent action dimension (generally consider part of the environment)

        self.episodes = 500
        self.evaluate_episodes = 50
        self.steps = 200  # note that OpenAI gym has max environment steps (e.g. max_step = 200 for CartPole)

        # general RL agent parameters
        self.batch_size = 32  # size of the memory buffer drawn for training and replay
        self.replay_buffer_size = 100  # size of the memory buffer, must > batch size
        assert self.replay_buffer_size >= self.batch_size
        self.discount_rate = 0.99

        # e-greedy algorithm
        self.stop_explore = 100

        # neural network parameters
        self.weights_initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)
        self.network_optimizer = tf.keras.optimizers.Adam(lr=1e-4,  # learning rate
                                                          clipnorm=1.0)  # gradient clipping
        self.activity_regularizer = tf.keras.regularizers.l1_l2(1e-3, 1e-3)
        self.keras_checkpoint = [tf.keras.callbacks.ModelCheckpoint(
            filepath='./results/saved_network_models/harvey.model',
            save_weights_only=True, mode='auto'),
            tf.keras.callbacks.TensorBoard(log_dir='./results/logs'),
            tf.keras.callbacks.EarlyStopping(patience=10, monitor='loss'),
        ]

        # Categorical DQN parameters
        self.categorical_Vmin = 0
        self.categorical_Vmax = 10
        self.categorical_n_atoms = 50

        # Quantile Regression DQN parameters
        self.num_quantiles = 20
        self.huber_loss_threshold = 1.0

        # Expetile ER-DQN parameters
        self.num_expectiles = 11
        self.z_val_limits = (-10, 10)
        self.imputation_distribution_bounds = tuple(self.z_val_limits for _ in range(self.num_expectiles))
        self.imputation_method = "root"  # root or minimization

        # a2c parameters
        self.head_out_dim = 20
