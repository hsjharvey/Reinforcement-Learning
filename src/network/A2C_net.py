# -*- coding:utf-8 -*-
import tensorflow as tf
from src.utils.config import *
import numpy as np
from tensorflow.keras.layers import Input, Dense, Reshape, Softmax


class ActorCriticNet:
    def __int__(self, config):
        self.config = config
        self.num_atoms = config.Categorical_n_atoms
        self.input_dim = config.input_dim
        self.action_dim = config.action_dim

        self.optimizer = config.optimizer
        self.net_model = None

    def actor_net(self):
        input_layer = Input(shape=self.input_dim, name='state_tensor_input')

        output_layers = Dense(units=self.action_dim,
                              use_bias=False,
                              activation='linear',
                              kernel_initializer=self.config.weights_initializer,
                              activity_regularizer=self.config.regularizer,
                              name='actor_net'
                              )(input_layer)

        return output_layers

    def critic(self, input_layer):
        output_layers = Dense(units=1,  # critic value
                              use_bias=False,
                              activation='linear',
                              kernel_initializer=self.config.weights_initializer,
                              activity_regularizer=self.config.regularizer,
                              name='critic_net'
                              )(input_layer)

        return output_layers

    def nn_model(self):
        actor_output = tf.keras.layers
