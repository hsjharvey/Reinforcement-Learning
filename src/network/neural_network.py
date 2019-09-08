# -*- coding:utf-8 -*-
import tensorflow as tf
from src.utils.config import *
import numpy as np


class DQNNet:
    def __init__(self, config):
        self.config = config
        self.input_dim = config.input_dim
        self.action_dim = config.action_dim
        self.output_dim = self.action_dim

        self.net_model = None

        self.optimizer = config.optimizer

    def nn_model(self):
        input_layer = tf.keras.layers.Input(shape=self.input_dim, name='state_tensor_input')

        output_layers = tf.keras.layers.Dense(units=self.output_dim,
                                              use_bias=False,
                                              input_shape=self.input_dim,  # input
                                              kernel_initializer=self.config.weights_initializer,
                                              activation='linear',
                                              activity_regularizer=self.config.regularizer,
                                              name='fully_connect'
                                              )(input_layer)

        actorNet_output_argmax = tf.reduce_max(output_layers, axis=2, name='argmax')

        self.net_model = tf.keras.models.Model(
            inputs=[input_layer],
            outputs=[output_layers, actorNet_output_argmax]
        )

        # we update the weights according to the loss of quantiles of optimal actions from both
        # action network and target network
        self.net_model.compile(
            loss=[None, 'mean_squared_error'],  # apply loss function only to the second output
            optimizer=self.optimizer

        )

        self.net_model.summary()  # print out the network structure

        return self.net_model


class CategoricalNet:
    def __init__(self, config):
        self.config = config
        self.num_atoms = config.Categorical_n_atoms
        self.input_dim = config.input_dim
        self.action_dim = config.action_dim
        self.output_dim = self.action_dim * self.num_atoms

        self.optimizer = config.optimizer
        self.net_model = None

        self.atoms = tf.linspace(
            float(config.Categorical_Vmin),
            float(config.Categorical_Vmax),
            config.Categorical_n_atoms,
        )  # Z

    def nn_model(self):
        input_layer = tf.keras.layers.Input(shape=self.input_dim, name='state_tensor_input')

        output_layers = tf.keras.layers.Dense(units=self.output_dim,
                                              use_bias=False,
                                              input_shape=self.input_dim,  # input
                                              activation='linear',
                                              kernel_initializer=self.config.weights_initializer,
                                              activity_regularizer=self.config.regularizer,
                                              name='fully_connect'
                                              )(input_layer)

        # processing layers ==> reshape and softmax, no training variables
        output_layers = tf.keras.layers.Reshape((self.action_dim, self.num_atoms))(output_layers)
        output_layers = tf.keras.layers.Softmax(axis=-1)(output_layers)

        action_values = tf.tensordot(output_layers, self.atoms, axes=1)

        # create an index of the max action value in each batch
        idx = tf.cast(tf.argmax(action_values, axis=1), dtype=tf.int32)

        # adjust the index to: [[0, 1], [1, 0], [2, 1], [3, 1]...etc.]
        # first number is row (batch) number, second number is the argmax max index
        idx = tf.transpose([tf.range(tf.shape(output_layers)[0]), idx])

        # gather probability histogram for actions with max action_values
        actorNet_output_argmax = tf.gather_nd(params=output_layers, indices=idx)

        # tensorflow keras: to set up the neural network itself.
        self.net_model = tf.keras.models.Model(
            inputs=input_layer,
            outputs=[output_layers, actorNet_output_argmax]
        )

        self.net_model.compile(
            loss=[None, 'categorical_cross_entropy'],  # apply loss function only to the second output
            optimizer=self.optimizer

        )

        self.net_model.summary()  # print out the network structure

        return self.net_model


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
        input_layer = tf.keras.layers.Input(shape=self.input_dim, name='state_tensor_input')

        output_layers = tf.keras.layers.Dense(units=self.output_dim,
                                              use_bias=False,
                                              input_shape=self.input_dim,  # input
                                              activation='linear',
                                              kernel_initializer=self.config.weights_initializer,
                                              activity_regularizer=self.config.regularizer,
                                              name='fully_connect'
                                              )(input_layer)

        # processing layers ==> reshape and softmax, no training variables
        output_layers = tf.keras.layers.Reshape((self.action_dim, self.num_quantiles))(output_layers)

        # get the action values
        # tf.cast is to cast the action values to int32
        action_values = tf.reduce_sum(output_layers, axis=2)
        idx = tf.cast(tf.argmax(action_values, axis=1), dtype=tf.int32)

        # to get the optimal action quantiles
        # size = [batch_size, 2 actions, quantiles per action]
        # we need to generate the correct index, size = [batch_size, argmax index]
        idx = tf.transpose([tf.range(tf.shape(output_layers)[0]), idx])

        # the final result is a [batch_size, quantiles] tensor for optimal actions
        actorNet_output_argmax = tf.gather_nd(params=output_layers, indices=idx)

        # tensorflow keras: to set up the neural network itself.
        self.net_model = tf.keras.models.Model(
            inputs=input_layer,
            outputs=[output_layers, actorNet_output_argmax]
        )

        # we update the weights according to the loss of quantiles of optimal actions from both
        # action network and target network
        self.net_model.compile(
            loss=[None, self.quantile_huber_loss],  # apply loss function only to the second output
            optimizer=self.optimizer

        )

        self.net_model.summary()  # print out the network structure

        return self.net_model

    def quantile_huber_loss(self, y_true, y_predict):
        """
        The loss function that is passed to the network
        :param y_true: True label, quantiles_next
        :param y_predict: predicted label, quantiles
        :return: quantile huber loss between the target quantiles and the quantiles
        """
        diff = y_true - y_predict

        model_loss = tf.reduce_mean(
            (self.huber_loss(diff) *
             tf.abs(self.cum_density - tf.cast(diff < 0, dtype=tf.float32))),
            axis=0)

        regularization_loss = tf.add_n(self.net_model.losses)

        total_loss = tf.add_n([tf.reduce_sum(model_loss), regularization_loss])

        return total_loss

    def huber_loss(self, item):
        return tf.where(
            tf.abs(item) < self.k,
            0.5 * np.power(item, 2),
            self.k * (tf.abs(item) - 0.5 * self.k)
        )


class ExpectileNet:
    def __init__(self, config):
        pass

    def nn_model(self):
        pass
