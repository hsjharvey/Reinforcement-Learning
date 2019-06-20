# -*- coding:utf-8 -*-
import tensorflow as tf
from utils.config import *
import numpy as np


class DQNNet:
    def __init__(self, config):
        self.config = config
        self.input_dim = config.input_dim
        self.action_dim = config.action_dim
        self.output_dim = self.action_dim

        self.optimizer = config.optimizer

    def nn_model(self):
        input_layer = tf.keras.layers.Input(shape=self.input_dim)

        hidden_1 = tf.keras.layers.Dense(units=16, activation='relu')(input_layer)

        hidden_2 = tf.keras.layers.Dense(units=16, activation='relu')(hidden_1)

        output_layers = tf.keras.layers.Dense(units=self.output_dim,
                                              use_bias=True,
                                              input_shape=self.input_dim,  # input
                                              activation='linear',
                                              activity_regularizer=tf.keras.regularizers.l2(1e-3)
                                              )(hidden_2)

        output_layer_2 = tf.reduce_max(output_layers, axis=2)

        self.net_model = tf.keras.models.Model(inputs=[input_layer], outputs=[output_layers, output_layer_2])

        # we update the weights according to the loss of quantiles of optimal actions from both
        # action network and target network
        self.net_model.compile(
            loss=[None, 'mean_squared_error'],
            optimizer=self.optimizer

        )

        self.net_model.summary()  # print out the network summary

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

    def nn_model(self):
        self.net_model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(units=self.output_dim,
                                  use_bias=True,
                                  input_shape=self.input_dim,  # input
                                  kernel_initializer='random_uniform',
                                  activity_regularizer=tf.keras.regularizers.l1_l2(1e-3, 1e-3)
                                  ),

            # processing layers ==> reshape and softmax, no training variables
            tf.keras.layers.Reshape((self.action_dim, self.num_atoms)),
            tf.keras.layers.Softmax(axis=-1)
        ])

        self.net_model.compile(
            loss='categorical_crossentropy',
            optimizer=self.optimizer

        )

        self.net_model.summary()  # print out the network summary

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

        self.cum_density = (2 * np.arange(config.num_quantiles) + 1) / (2.0 * config.num_quantiles)

    def nn_model(self):
        input_layer = tf.keras.layers.Input(shape=self.input_dim)

        output_layers = tf.keras.layers.Dense(units=self.output_dim,
                                              use_bias=True,
                                              input_shape=self.input_dim,  # input
                                              kernel_initializer='random_uniform',
                                              activity_regularizer=tf.keras.regularizers.l1_l2(1e-3, 1e-3)
                                              )(input_layer)

        # processing layers ==> reshape and softmax, no training variables
        output_layers = tf.keras.layers.Reshape((self.action_dim, self.num_quantiles))(output_layers)

        # get the action values
        # tf.cast is to cast the action values to in32
        action_values = tf.reduce_sum(output_layers, axis=2)
        idx = tf.cast(tf.argmax(action_values, axis=1), dtype=tf.int32)

        # to get the optimal action quantiles [batch_size, 2 actions, quantiles per action]
        # we need to generate the correct index
        idx = tf.transpose([tf.range(tf.shape(output_layers)[0]), idx])

        # the final result is a [batch_size, quantiles] tensor for optimal actions
        output_layer_2 = tf.gather_nd(params=output_layers, indices=idx)

        self.net_model = tf.keras.models.Model(inputs=input_layer, outputs=[output_layers, output_layer_2])

        # we update the weights according to the loss of quantiles of optimal actions from both
        # action network and target network
        self.net_model.compile(
            loss=[None, self.quantile_huber_loss],
            optimizer=self.optimizer

        )

        self.net_model.summary()  # print out the network summary

        return self.net_model

    def quantile_huber_loss(self, y_true, y_predict):
        diff = y_true - y_predict

        loss = tf.reduce_mean(
            (self.huber_loss(diff) *
             tf.abs(self.cum_density - tf.cast(diff < 0, dtype=tf.float32))),
            axis=0)

        loss = tf.reduce_sum(loss)

        return loss

    def huber_loss(self, item, k=1.0):
        return tf.where(tf.abs(item) < k, 0.5 * np.power(item, 2), k * (tf.abs(item) - 0.5 * k))


if __name__ == '__main__':
    # C = Config()
    #
    # x = np.random.randn(30, 1, 4)
    # cat = CategoricalNet(config=C)
    # cat_nn = cat.nn_model()
    # predictions = cat_nn.predict(x)

    # np.random.seed(1000)
    #
    # C = Config()
    # cat = QuantileNet(config=C)
    # cat.action_next = np.array([1, 1, 1, 1, 0])
    # cat.action = np.array([0, 0, 0, 1, 1])
    # y_true = np.random.randn(5, 2, 30)
    # y_predict = np.random.randn(5, 2, 30)
    #
    # print(cat.my_loss(y_true, y_predict))
    #
    # C = Config()
    # x = np.random.randn(30, 1, 4)
    # cat = QuantileNet(config=C)
    # cat_nn = cat.nn_model()
    # predictions = cat_nn.predict(x)
    # print(predictions[0].shape)
    # print(predictions[1].shape)

    C = Config()
    x = np.random.randn(30, 1, 4)
    cat = DQNNet(config=C)
    cat_nn = cat.nn_model()
    predictions = cat_nn.predict(x)
    print(predictions[0].shape)
    print(predictions[1].shape)
