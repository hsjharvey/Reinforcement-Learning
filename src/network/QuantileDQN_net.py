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

        self.optimizer = config.network_optimizer
        self.net_model = None

        self.k = config.huber_loss_threshold
        # quantiles, e.g. [0.125, 0.375, 0.625, 0.875]
        self.cum_density = (2 * np.arange(config.num_quantiles) + 1) / (2.0 * config.num_quantiles)
        # append r_0 = 0, [0.   , 0.125, 0.375, 0.625, 0.875]
        temp = np.sort(np.append(self.cum_density, 0))
        # calculate r_hat [0.0625, 0.25, 0.5, 0.75], see lemma 2 in the original paper.
        self.r_hat = np.array([(temp[j] + temp[j - 1]) / 2 for j in range(1, temp.shape[0])])

    def nn_model(self):
        input_layer = Input(shape=self.input_dim, name='state_tensor_input')
        output_layers = Dense(units=24, activation="relu", name='hidden_layer_1')(input_layer)
        output_layers = Dense(units=self.output_dim, activation='linear', name='output_layer')(output_layers)

        # processing layers ==> reshape and softmax, no training variables
        output_layers = Reshape((self.action_dim, self.num_quantiles))(output_layers)

        # get the action values
        # tf.cast is to cast the action values to int32
        action_values = tf.reduce_sum(output_layers, axis=2)
        action = tf.cast(tf.argmax(action_values, axis=1), dtype=tf.int32)

        # to get the optimal action quantiles
        # size = [batch_size, 2 actions, quantiles per action]
        # we need to generate the correct index, size = [batch_size, argmax index]
        idx = tf.transpose([tf.range(tf.shape(output_layers)[0]), action])

        # the final result is a [batch_size, quantiles] tensor for optimal actions
        optimal_action_quantiles = tf.gather_nd(params=output_layers, indices=idx)

        # tensorflow keras: to set up the neural network itself.
        self.net_model = tf.keras.models.Model(
            inputs=input_layer,
            outputs=[output_layers, optimal_action_quantiles]
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
        see algorithm 1 in the original paper for more details
        :param y_true: true label, quantiles_next, [batch_size, num_quantiles]
        :param y_predict: predicted label, quantiles, [batch_size, num_quantiles]
        :return: quantile huber loss between the target quantiles and the quantiles
        """
        batch_loss = []
        for each_batch in range(self.config.batch_size):
            each_transition_sample_loss = 0
            for i in range(self.num_quantiles):
                diff = y_true[each_batch] - y_predict[each_batch, i]

                # calculate the expected value over j
                target_loss = tf.reduce_mean(
                    (self.huber_loss(diff) *
                     tf.abs(self.r_hat[i] - tf.cast(diff < 0, dtype=tf.float32))))

                # sum over i in algorithm 1
                each_transition_sample_loss += target_loss

            # get batch loss size=(32, 1)
            batch_loss.append(each_transition_sample_loss)
        return tf.reduce_mean(batch_loss)

    def huber_loss(self, mu):
        """
        equation 10 of the original paper
        :return:
        """
        return tf.where(
            tf.abs(mu) < self.k,
            0.5 * tf.square(mu),
            self.k * (tf.abs(mu) - 0.5 * self.k)
        )
