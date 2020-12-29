# -*- coding:utf-8 -*-
import tensorflow as tf
from src.utils.config import *
import numpy as np
from tensorflow.keras.layers import Input, Dense, Reshape, Softmax, Lambda


class ExpectileNet:
    def __init__(self, config):
        self.config = config
        self.input_dim = config.input_dim
        self.action_dim = config.action_dim
        self.num_expectiles = config.num_expectiles
        self.output_dim = self.action_dim * self.num_expectiles

        self.batch_size = self.config.batch_size

        self.optimizer = config.network_optimizer
        self.net_model = None

        # note that middle expectile statistic is in fact the mean, i.e. tau_{middle}
        self.cum_density = tf.linspace(0.01, 0.99, config.num_expectiles)

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

        # get the action values <=> mid expectile
        # tf.cast is to cast the action values to int32
        action_values = output_layers[:, :, int(self.num_expectiles / 2) + 1]

        action = tf.cast(tf.argmax(action_values, axis=1), dtype=tf.int32)

        # to get the optimal action expectiles
        # size = [batch_size, 2 actions, expectiles per action]
        # we need to generate the correct index, size = [batch_size, argmax index]
        idx = tf.transpose([tf.range(tf.shape(output_layers)[0]), action])

        # the final result is a [batch_size, expectiles] tensor for optimal actions
        optimal_action_expectiles = tf.gather_nd(params=output_layers, indices=idx)

        # tensorflow keras: to set up the neural network itself.
        self.net_model = tf.keras.models.Model(
            inputs=input_layer,
            outputs=[output_layers, optimal_action_expectiles]
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
        :param y_true: True label, distribution after imputation (batch_size, number of z values)
        :param y_predict: predicted label, expectile_predict (batch_size, number of expectiles)
        :return: expectile loss between the target expectile and the predicted expectile
        """
        batch_loss = []
        for i in range(self.batch_size):
            expectile_predict = y_predict[i]
            z = y_true[i]
            loss_val = 0

            for k in range(self.num_expectiles):
                diff = z - expectile_predict[k]
                diff_square = tf.square(diff)

                er_loss = tf.reduce_mean(tf.where(diff > 0, self.cum_density[k] * diff_square,
                                                  (1 - self.cum_density[k]) * diff_square))
                # sum over all kth statistics
                loss_val += er_loss

            # get batch loss size=(32, 1)
            batch_loss.append(loss_val)

        return tf.reduce_mean(batch_loss)
