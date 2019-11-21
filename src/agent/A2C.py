# -*- coding:utf-8 -*-
from src.utils import *
import numpy as np
import tensorflow as tf
from collections import deque
import gym


class A2Cgent:
    def __init__(self, config, base_network):
        self.base_network = base_network
        self.config = config

        self.input_dim = config.input_dim  # neural network input dimension

        self.envs = None
        self.actor = self.base_network.nn_model()
        self.critic = tf.keras.models.clone_model(self.actor)
        self.critic.set_weights(self.actor.get_weights())

        self.total_steps = 0
        self.episodes = config.episodes
        self.steps = config.steps
        self.batch_size = config.batch_size

        self.replay_buffer_size = config.replay_buffer_size
        self.replay_buffer = deque()

        self.keras_check = config.keras_checkpoint

        self.check = 0
        self.best_max = 0

    def transition(self):
        """
        In transition, the agent simply plays and record
        [current_state, action, reward, next_state, done]
        in the replay_buffer (or memory pool)

        Updating the weights of the neural network happens
        every single time the replay buffer size is reached.

        done: boolean, whether the game has end or not.
        """
        for each_ep in range(self.episodes):
            current_state = self.envs.reset()

            print('Episode: {}  Reward: {} Max_Reward: {}'.format(each_ep, self.check, self.best_max))
            print('-' * 64)
            self.check = 0

            for step in range(self.steps):
                quantile_values, _ = self.actor.predict(
                    np.array(current_state).reshape((1, self.input_dim[0], self.input_dim[1])))
                action_value = quantile_values.mean(-1)

                # choose action according to the E-greedy policy
                action = policies.epsilon_greedy(action_values=action_value[0],
                                                 episode=each_ep,
                                                 stop_explore=self.config.stop_explore,
                                                 total_actions=self.config.action_dim)

                next_state, reward, done, _ = self.envs.step(action=action)

                # record the per step history into replay buffer
                self.replay_buffer.append([current_state.reshape(self.input_dim).tolist(), action,
                                           next_state.reshape(self.input_dim).tolist(), reward, done])

                # when we collect certain number of batches, perform replay and
                # update the weights in the actor network (Backpropagation)
                # reset the replay buffer
                if len(list(self.replay_buffer)) == self.replay_buffer_size:
                    self.train_by_replay()
                    self.replay_buffer = deque()

                # if episode is finished, break the inner loop
                # otherwise, continue
                if done:
                    self.total_steps += 1
                    break
                else:
                    current_state = next_state
                    self.total_steps += 1
                    self.check += reward

    def train_by_replay(self):
        pass

    def eval_step(self, render=True):
        pass
