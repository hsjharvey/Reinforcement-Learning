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
        self.replay_buffer_size = 500  # size of the memory buffer, must > batch size
        assert self.replay_buffer_size >= self.batch_size
        self.discount_rate = 0.9

        # e-greedy algorithm
        self.stop_explore = 50

        # neural network parameters
        # self.weights_initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=1e-2, seed=None)
        # self.activity_regularizer = tf.keras.regularizers.l1_l2(1e-3, 1e-3)
        self.network_optimizer = tf.keras.optimizers.Adam(lr=1e-4,  # learning rate
                                                          clipnorm=1.0)  # gradient clipping
        self.keras_checkpoint = [tf.keras.callbacks.ModelCheckpoint(
            filepath='./results/saved_network_models/harvey.model',
            save_weights_only=True, mode='auto'),
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
        self.num_expectiles = 10
        self.z_val_limits = (0, 10)
        self.num_imputed_samples = 10
        self.imputation_distribution_bounds = tuple(self.z_val_limits for _ in range(self.num_imputed_samples))
        self.imputation_method = "root"  # root or minimization

        # the default root method is "hybr", it requires the input shape of x to be the same as
        # the output shape of the root results
        # in this case, it means that the imputed sample size to be exactly the same
        # as the number of expectiles
        # this is also the assumption in the paper if you look closely at Algorithm 2 and appendix D.1
        if self.imputation_method == "root":
            assert self.num_expectiles == self.num_imputed_samples, \
                "if you use root method, the number of imputed samples must be equal to the number of expectiles"

        # a2c parameters
        self.head_out_dim = 20
