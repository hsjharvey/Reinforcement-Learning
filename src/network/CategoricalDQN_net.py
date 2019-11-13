# -*- coding:utf-8 -*-
import tensorflow as tf
from src.utils.config import *
import numpy as np
from tensorflow.keras.layers import Input, Dense, Reshape, Softmax


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
        input_layer = Input(shape=self.input_dim, name='state_tensor_input')

        output_layers = Dense(units=self.output_dim,
                              use_bias=False,
                              input_shape=self.input_dim,  # input
                              activation='linear',
                              kernel_initializer=self.config.weights_initializer,
                              activity_regularizer=self.config.regularizer,
                              name='fully_connect'
                              )(input_layer)

        # processing layers ==> reshape and softmax, no training variables
        output_layers = Reshape((self.action_dim, self.num_atoms))(output_layers)
        output_layers = Softmax(axis=-1)(output_layers)

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
