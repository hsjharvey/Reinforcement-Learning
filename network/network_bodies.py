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
        self.check = tf.keras.callbacks.ModelCheckpoint('results/harvey.model',
                                                        monitor='val_acc', verbose=1, save_best_only=True,
                                                        save_weights_only=False,
                                                        mode='auto')

        self.optimizer = None
        self.net_model = None

    def nn_model(self):
        self.net_model = tf.keras.models.Sequential([
            # output layer
            tf.keras.layers.Dense(units=self.output_dim,
                                  activation='softmax',
                                  use_bias=False,
                                  input_shape=self.input_dim,
                                  kernel_initializer='random_uniform',
                                  activity_regularizer=tf.keras.regularizers.l1_l2(1e-2, 1e-2)
                                  ),

            # processing layers ==> reshape and softmax, no training variables
            tf.keras.layers.Reshape((self.action_dim, self.num_atoms)),
            tf.keras.layers.Softmax(axis=-1)
        ])

        self.optimizer = tf.keras.optimizers.Adam(1e-2)

        self.net_model.compile(
            loss=self.customize_loss_fn,
            opimizer=self.optimizer
        )

        return self.net_model

    def customize_loss_fn(self, y_predict, y_target):
        return -(np.log(y_predict) * y_target).sum(-1).mean()


if __name__ == '__main__':
    C = Config()

    x = np.random.randn(30, 1, 4)
    cat = CategoricalNet(config=C)
    cat_nn = cat.nn_model()
    cat_nn.summary()
    predictions = cat_nn.predict(x)
