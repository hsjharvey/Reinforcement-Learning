# -*- coding:utf-8 -*-
from src.utils import *
import numpy as np
import tensorflow as tf
from collections import deque
import gym


class A2Cagent:
    def __init__(self, config, base_network):
        self.base_network = base_network
        self.config = config

        self.input_dim = config.input_dim  # neural network input dimension

        self.envs = None
        self.A2C_network = self.base_network.nn_model()
        self.best_network = tf.keras.models.clone_model(self.A2C_network)

        self.total_steps = 0
        self.episodes = config.episodes
        self.steps = config.steps
        self.batch_size = config.batch_size

        self.replay_buffer_size = config.replay_buffer_size
        self.replay_buffer = deque()

        self.keras_check = config.keras_checkpoint

        self.check_model_improved = 0
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

            print('Episode: {} Reward: {} Max_Reward: {}'.format(each_ep, self.check_model_improved, self.best_max))
            print('-' * 64)
            self.check_model_improved = 0

            for step in range(self.steps):
                actor_output, _ = self.A2C_network.predict(
                    np.array(current_state).reshape((1, self.input_dim[0], self.input_dim[1])))

                # choose action according to the E-greedy policy
                action = policies.epsilon_greedy(action_values=actor_output.reshape(self.config.action_dim),
                                                 episode=each_ep,
                                                 stop_explore=self.config.stop_explore,
                                                 total_actions=self.config.action_dim)

                next_state, reward, done, _ = self.envs.step(action=action)
                returns = self.config.discount_rate * reward  # calculate returns/discounted rewards

                # record the per step history into replay buffer
                self.replay_buffer.append([current_state.reshape(self.input_dim).tolist(), action,
                                           next_state.reshape(self.input_dim).tolist(), returns, done])

                # when we collect certain number of batches, perform replay and
                # update the weights in the actor network (Backpropagation)
                # reset the replay buffer
                if len(self.replay_buffer) == self.replay_buffer_size:
                    self.train_by_replay()
                    self.replay_buffer.clear()

                # if episode is finished, break the inner loop
                # otherwise, continue
                if done:
                    self.total_steps += 1
                    break
                else:
                    current_state = next_state
                    self.total_steps += 1
                    self.check_model_improved += reward
            # for any episode where the reward is higher
            # we copy the actor network weights to the target network
            if self.check_model_improved > self.best_max:
                self.best_max = self.check_model_improved
                self.best_network.set_weights(self.A2C_network.get_weights())

    def train_by_replay(self):
        """
        TD update by replaying the history.
        """
        # step 1: generate replay samples (size = self.batch_size) from the replay buffer
        # e.g. uniform random replay or prioritize experience replay
        current_states, actions, next_states, returns, done = \
            replay_fn.uniform_random_replay(self.replay_buffer, self.batch_size)

        returns = returns.reshape((-1, 1, 1))

        actor_output, critic_values = self.A2C_network.predict(current_states)

        advantage = returns - critic_values
        self.A2C_network.fit(x=current_states, y=[advantage, returns], verbose=2, callbacks=self.keras_check)

    def eval_step(self, render=True):
        """
        Evaluation using the trained target network, no training involved
        :param render: whether to visualize the evaluation or not
        """
        for each_ep in range(self.config.evaluate_episodes):
            current_state = self.envs.reset()

            print('Episode: {}  eward: {} Training_Max_Reward: {}'.format(each_ep, self.check_model_improved,
                                                                          self.best_max))
            print('-' * 64)
            self.check_model_improved = 0

            for step in range(self.steps):
                action_values, _ = self.best_network.predict(
                    np.array(current_state).reshape((1, self.input_dim[0], self.input_dim[1])))
                action = np.argmax(action_values.reshape(self.config.action_dim))

                next_state, reward, done, _ = self.envs.step(action=action)

                if render:
                    self.envs.render(mode=['human'])

                if done:
                    break
                else:
                    current_state = next_state
                    self.check_model_improved += 1
