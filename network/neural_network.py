# -*- coding:utf-8 -*-
import tensorflow as tf
from utils.config import *
import numpy as np
import keras.backend as k


class CategoricalNet:
    def __init__(self, config):
        self.config = config
        self.num_atoms = config.Categorical_n_atoms
        self.input_dim = config.input_dim
        self.action_dim = config.action_dim
        self.output_dim = self.action_dim * self.num_atoms

        self.optimizer = None
        self.net_model = None

    def nn_model(self):
        self.net_model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(units=self.output_dim,
                                  use_bias=True,
                                  input_shape=self.input_dim,  # input
                                  kernel_initializer='random_uniform',
                                  activity_regularizer=tf.keras.regularizers.l1_l2(1e-2, 1e-2)
                                  ),

            # processing layers ==> reshape and softmax, no training variables
            tf.keras.layers.Reshape((self.action_dim, self.num_atoms)),
            tf.keras.layers.Softmax(axis=-1)
        ])

        self.net_model.compile(
            loss='categorical_crossentropy',
            optimizer=tf.keras.optimizers.Adam(1e-2)

        )

        self.net_model.summary()  # print out the network summary

        return self.net_model


class QuantileNet_new:
    def __init__(self, config):
        self.config = config
        self.num_quantiles = config.num_quantiles
        self.input_dim = config.input_dim
        self.action_dim = config.action_dim
        self.output_dim = self.action_dim * self.num_quantiles

        self.optimizer = None
        self.net_model = None

        self.action = None
        self.action_next = None

        self.cum_density = (2 * np.arange(config.num_quantiles) + 1) / (2.0 * config.num_quantiles)

    def nn_model(self):
        input_layer = tf.keras.layers.Input(shape=self.input_dim)

        output_layers = tf.keras.layers.Dense(units=self.output_dim,
                                              use_bias=True,
                                              input_shape=self.input_dim,  # input
                                              kernel_initializer='random_uniform',
                                              activity_regularizer=tf.keras.regularizers.l1_l2(1e-2, 1e-2)
                                              )(input_layer)

        # processing layers ==> reshape and softmax, no training variables
        output_layers = tf.keras.layers.Reshape((self.action_dim, self.num_quantiles))(output_layers)
        output_layers = tf.keras.layers.Softmax(axis=-1)(output_layers)

        output_layer_2 = tf.reduce_mean(output_layers, axis=-1)
        output_layer_2 = tf.reduce_max(output_layer_2, axis=-1)

        self.net_model = tf.keras.models.Model(inputs=input_layer, outputs=[output_layers, output_layer_2])

        self.net_model.compile(
            loss=[None, self.my_loss],
            optimizer=tf.keras.optimizers.Adam(1e-2)

        )

        self.net_model.summary()  # print out the network summary

        return self.net_model

    def my_loss(self, y_true, y_predict):
        diff = y_true - y_predict
        print(diff)

        loss = tf.reduce_mean((self.huber_loss(diff) * tf.abs(self.cum_density - tf.cast(diff < 0, dtype=tf.float32))),
                              axis=0)
        loss = tf.reduce_sum(loss)

        return loss

    def huber_loss(self, x, k=1.0):
        return tf.where(tf.abs(x) < k, 0.5 * np.power(x, 2), k * (tf.abs(x) - 0.5 * k))


class QuantileNet_old:
    def __init__(self, config):
        self.config = config
        self.num_quantiles = config.num_quantiles
        self.input_dim = config.input_dim
        self.action_dim = config.action_dim
        self.output_dim = self.action_dim * self.num_quantiles

        self.optimizer = None
        self.net_model = None

        self.action = None
        self.action_next = None

        self.cum_density = (2 * np.arange(config.num_quantiles) + 1) / (2.0 * config.num_quantiles)

    def nn_model(self):
        self.net_model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(units=self.output_dim,
                                  use_bias=True,
                                  input_shape=self.input_dim,  # input
                                  kernel_initializer='random_uniform',
                                  activity_regularizer=tf.keras.regularizers.l1_l2(1e-2, 1e-2)
                                  ),

            # processing layers ==> reshape and softmax, no training variables
            tf.keras.layers.Reshape((self.action_dim, self.num_quantiles)),
            tf.keras.layers.Softmax(axis=-1)
        ])

        self.net_model.compile(
            loss=self.my_loss(action_list=self.action),
            optimizer=tf.keras.optimizers.Adam(1e-2)

        )

        self.net_model.summary()  # print out the network summary

        return self.net_model

    def my_loss(self, action_list):
        def loss(y_true, y_predict):
            y_predict = y_predict[np.arange(y_predict.shape[0]), action_list, :]

            x = y_true - y_predict

            # k = 1.0
            huber_loss = np.where(np.abs(x) < 1.0, 0.5 * np.power(x, 2), 1.0 * (np.abs(x) - 0.5 * 1.0))

            return (huber_loss * np.abs((self.cum_density - (x < 0)))).mean(1).sum()

        return loss


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

    C = Config()
    x = np.random.randn(30, 1, 4)
    cat = QuantileNet_new(config=C)
    cat_nn = cat.nn_model()
    predictions = cat_nn.predict(x)
    print(predictions[0].shape)
    print(predictions[1].shape)
