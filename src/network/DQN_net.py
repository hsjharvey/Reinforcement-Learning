# -*- coding:utf-8 -*-
import tensorflow as tf
from src.utils.config import *
import numpy as np
from tensorflow.keras.layers import Input, Dense, Reshape, Softmax


class DQNNet:
    def __init__(self, config):
        self.config = config
        self.input_dim = config.input_dim
        self.action_dim = config.action_dim
        self.output_dim = self.action_dim

        self.net_model = None

        self.optimizer = config.network_optimizer

    def nn_model(self):
        input_layer = Input(shape=self.input_dim, name='state_tensor_input')
        output_layers = Dense(units=24, activation="linear", name='hidden_layer_1')(input_layer)
        output_layers = Dense(units=24, activation="linear", name='hidden_layer_2')(output_layers)
        output_layers = Dense(units=self.output_dim, activation='softmax', name='output_layer')(output_layers)

        self.net_model = tf.keras.models.Model(
            inputs=[input_layer],
            outputs=[output_layers]
        )

        # we update the weights according to the loss of quantiles of optimal actions from both
        # action network and target network
        self.net_model.compile(
            loss=['mean_squared_error'],  # apply loss function only to the second output
            optimizer=self.optimizer
        )

        self.net_model.summary()  # print out the network structure

        return self.net_model
