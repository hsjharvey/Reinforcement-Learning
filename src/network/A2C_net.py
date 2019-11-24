# -*- coding:utf-8 -*-
import tensorflow as tf
from src.utils.config import *
import numpy as np
from tensorflow.keras.layers import Input, Dense, Reshape, Softmax


class ActorCriticNet:
    def __init__(self, config):
        self.config = config
        self.num_atoms = config.Categorical_n_atoms
        self.input_dim = config.input_dim
        self.action_dim = config.action_dim
        self.log_action_prob = 0.0

        self.optimizer = config.optimizer

    def nn_model(self):
        input_layer = Input(shape=self.input_dim, name='state_representation_head')

        shared_net_head = Dense(units=self.config.head_out_dim,
                                use_bias=False,
                                activation='linear',
                                kernel_initializer=self.config.weights_initializer,
                                activity_regularizer=self.config.activity_regularizer,
                                name='shared_network_head'
                                )(input_layer)

        actor_output = Dense(units=self.action_dim,
                             use_bias=False,
                             activation='linear',
                             kernel_initializer=self.config.weights_initializer,
                             activity_regularizer=self.config.activity_regularizer,
                             name='actor_net'
                             )(shared_net_head)

        actor_output = Softmax(axis=-1)(actor_output)

        critic_output = Dense(units=1,  # critic value
                              use_bias=False,
                              activation='linear',
                              kernel_initializer=self.config.weights_initializer,
                              activity_regularizer=self.config.activity_regularizer,
                              name='critic_net'
                              )(shared_net_head)

        # log probability

        self.net_model = tf.keras.models.Model(
            inputs=[input_layer],
            outputs=[actor_output, critic_output]
        )

        # the loss values in a2c are very complicated.
        self.net_model.compile(
            loss=[self.actor_net_loss, self.critic_net_loss],
            optimizer=self.optimizer,
        )

        self.net_model.summary()  # print out the network structure

        return self.net_model

    def actor_net_loss(self, y_true, y_predict):
        """
        actor network loss function
        :param y_true: returns (discounted rewards)
        :param y_predict: value predicted by critic_net
        :return: loss function
        """
        return - tf.reduce_mean(self.log_action_prob * (y_true - y_predict))

    def critic_net_loss(self, y_true, y_predict):
        """
        critic network loss function
        :param y_true: returns (discounted rewards)
        :param y_predict: value predicted by critic_net
        :return: loss function
        """
        return tf.keras.losses.mean_squared_error(y_true, y_predict)
