# -*- coding:utf-8 -*-
import tensorflow as tf
from utils.config import *
import numpy as np


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
                                  activation='softmax',
                                  use_bias=False,
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

        self.net_model.summary()  # printout the network summary

        return self.net_model


class QuantileNet:
    pass


if __name__ == '__main__':
    C = Config()

    x = np.random.randn(30, 1, 4)
    cat = CategoricalNet(config=C)
    cat_nn = cat.nn_model()
    predictions = cat_nn.predict(x)
