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
        self.net_model = None

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
                             activation='softmax',
                             kernel_initializer=self.config.weights_initializer,
                             activity_regularizer=self.config.activity_regularizer,
                             name='actor_net'
                             )(shared_net_head)

        critic_output = Dense(units=1,  # critic value
                              use_bias=False,
                              activation='linear',
                              kernel_initializer=self.config.weights_initializer,
                              activity_regularizer=self.config.activity_regularizer,
                              name='critic_net'
                              )(shared_net_head)

        self.net_model = tf.keras.models.Model(
            inputs=[input_layer],
            outputs=[actor_output, critic_output]
        )

        # the loss values in a2c are very complicated.
        self.net_model.compile(
            loss={'actor_net': self.actor_net_loss, 'critic_net': self.critic_net_loss},
            optimizer=self.optimizer,
        )

        self.net_model.summary()  # print out the network structure

        return self.net_model

    def actor_net_loss(self, y_true, y_predict):
        """
        actor network loss function
        cite: https://github.com/germain-hug/Deep-RL-Keras
        :param y_true: advantage (returns/discounted reward - critic_value)
        :param y_predict: predicted_probability of actions from the actor net
        :return: loss function
        """
        weighted_actions = tf.reduce_sum(np.arange(self.config.action_dim) * y_predict, axis=2)
        entropy = tf.reduce_sum(weighted_actions * tf.math.log(weighted_actions))
        eligibility = tf.math.log((weighted_actions + 1e-10) * y_true)
        return 0.0001 * entropy - 0.0001 * tf.reduce_sum(eligibility)

    def critic_net_loss(self, y_true, y_predict):
        """
        critic network loss function
        :param y_true: returns (discounted rewards)
        :param y_predict: value predicted by critic_net
        :return: loss function
        """
        return tf.keras.losses.mean_squared_error(y_true, y_predict)
