# -*- coding:utf-8 -*-
import tensorflow as tf
from src.utils.config import *
import numpy as np
from tensorflow.keras.layers import Input, Dense, Reshape, Softmax


class QuantileNet:
    def __init__(self, config):
        self.config = config
        self.num_quantiles = config.num_quantiles
        self.input_dim = config.input_dim
        self.action_dim = config.action_dim
        self.output_dim = self.action_dim * self.num_quantiles

        self.optimizer = config.optimizer
        self.net_model = None

        self.k = config.huber_loss_threshold

        self.cum_density = (2 * np.arange(config.num_quantiles) + 1) / (2.0 * config.num_quantiles)

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

        # processing layers ==> reshape and softmax, no training variables
        output_layers = Reshape((self.action_dim, self.num_quantiles))(output_layers)

        # get the action values
        # tf.cast is to cast the action values to int32
        action_values = tf.reduce_sum(output_layers, axis=2)
        idx = tf.cast(tf.argmax(action_values, axis=1), dtype=tf.int32)

        # to get the optimal action quantiles
        # size = [batch_size, 2 actions, quantiles per action]
        # we need to generate the correct index, size = [batch_size, argmax index]
        idx = tf.transpose([tf.range(tf.shape(output_layers)[0]), idx])

        # the final result is a [batch_size, quantiles] tensor for optimal actions
        actor_net_output_argmax = tf.gather_nd(params=output_layers, indices=idx)

        # tensorflow keras: to set up the neural network itself.
        self.net_model = tf.keras.models.Model(
            inputs=input_layer,
            outputs=[output_layers, actor_net_output_argmax]
        )

        # we update the weights according to the loss of quantiles of optimal actions from both
        # action network and target network
        self.net_model.compile(
            loss=[None, self.quantile_huber_loss],  # apply loss function only to the second output
            optimizer=self.optimizer

        )

        self.net_model.summary()  # print out the network structure

        return self.net_model

    def quantile_huber_loss(self, quantile_next, quantile_predict):
        """
        The loss function that is passed to the network
        :param quantile_next: True label, quantiles_next
        :param quantile_predict: predicted label, quantiles
        :return: quantile huber loss between the target quantiles and the quantiles
        """
        diff = quantile_next - quantile_predict

        target_loss = tf.reduce_mean(
            (self.huber_loss(diff) *
             tf.abs(self.cum_density - tf.cast(diff < 0, dtype=tf.float32))),
            axis=0)

        regularization_loss = tf.add_n(self.net_model.losses)

        total_loss = tf.add_n([tf.reduce_sum(target_loss), regularization_loss])

        return total_loss

    def huber_loss(self, item):
        return tf.where(
            tf.abs(item) < self.k,
            0.5 * np.power(item, 2),
            self.k * (tf.abs(item) - 0.5 * self.k)
        )
