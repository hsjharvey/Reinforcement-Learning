# -*- coding:utf-8 -*-
import tensorflow as tf
from src.utils.config import *
import numpy as np
from tensorflow.keras.layers import Input, Dense, Reshape, Softmax


class ExpectileNet:
    def __init__(self, config):
        self.config = config
        self.input_dim = config.input_dim
        self.action_dim = config.action_dim
        self.num_expectiles = config.num_expectiles
        self.output_dim = self.action_dim * self.num_expectiles

        self.optimizer = config.optimizer
        self.net_model = None

        # note that tau_6 = 0.5 and thus this expectile statistic is in fact the mean
        # tau
        self.cum_density = np.linspace(0.01, 0.99, config.num_expectiles)

    def nn_model(self):
        input_layer = Input(shape=self.input_dim, name='state_tensor_input')

        output_layers = Dense(units=self.output_dim,
                              use_bias=False,
                              input_shape=self.input_dim,  # input
                              activation='linear',
                              kernel_initializer=self.config.weights_initializer,
                              activity_regularizer=self.config.activity_regularizer,
                              name='fully_connect'
                              )(input_layer)

        # processing layers ==> reshape, no training variables
        output_layers = Reshape((self.action_dim, self.num_expectiles))(output_layers)

        # get the action values
        # tf.cast is to cast the action values to int32
        action_values = tf.gather_nd(params=output_layers, indices=[[0, 0, 6], [0, 1, 6]])
        action = tf.cast(tf.argmax(action_values, axis=1), dtype=tf.int32)

        # to get the optimal action expectiles
        # size = [batch_size, 2 actions, expectiles per action]
        # we need to generate the correct index, size = [batch_size, argmax index]
        idx = tf.transpose([tf.range(tf.shape(output_layers)[0]), action])

        # the final result is a [batch_size, expectiles] tensor for optimal actions
        actor_net_output_argmax = tf.gather_nd(params=output_layers, indices=idx)

        # tensorflow keras: to set up the neural network itself.
        self.net_model = tf.keras.models.Model(
            inputs=input_layer,
            outputs=[output_layers, actor_net_output_argmax]
        )

        # we update the weights according to the loss of expectiles of optimal actions from both
        # action network and target network
        self.net_model.compile(
            loss=[None, self.expectile_regression_loss],  # apply loss function only to the second output
            optimizer=self.optimizer

        )

        self.net_model.summary()  # print out the network structure

        return self.net_model

    def expectile_regression_loss(self, y_true, y_predict):
        """
        The loss function that is passed to the network
        :param y_true: True label, expectiles_next
        :param y_predict: predicted label, expectiles
        :return: expectile huber loss between the target expectile and the expectile
        """
        diff = y_true - y_predict
