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
        self.log_action_prob = None

        self.optimizer = config.optimizer

    def nn_model(self):
        input_layer = Input(shape=self.input_dim, name='state_tensor_input')

        shared_net_head = Dense(units=self.action_dim,
                                use_bias=False,
                                activation='linear',
                                kernel_initializer=self.config.weights_initializer,
                                activity_regularizer=self.config.regularizer,
                                name='actor_net'
                                )(input_layer)

        actor_output = Softmax(axis=-1)(shared_net_head)

        critic_output = Dense(units=1,  # critic value
                              use_bias=False,
                              activation='linear',
                              kernel_initializer=self.config.weights_initializer,
                              activity_regularizer=self.config.regularizer,
                              name='critic_net'
                              )(input_layer)

        # log probability
        log_action_prob = tf.math.log(actor_output)

        self.net_model = tf.keras.models.Model(
            inputs=[input_layer],
            outputs=[log_action_prob, actor_output, critic_output]
        )

        # the loss values in a2c are very complicated.
        self.net_model.compile(
            loss=[None, self.actor_net_loss, self.critic_net_loss],
            optimizer=self.optimizer
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
