# -*- coding:utf-8 -*-
from src.utils import *
import numpy as np
import tensorflow as tf
from collections import deque
import gym


class CategoricalDQNAgent:
    def __init__(self, config, base_network):
        self.config = config
        self.base_network = base_network

        self.input_dim = config.input_dim  # neural network input dimension
        self.n_atoms = config.categorical_n_atoms
        self.vmin = config.categorical_Vmin
        self.vmax = config.categorical_Vmax

        self.atoms = np.linspace(
            config.categorical_Vmin,
            config.categorical_Vmax,
            config.categorical_n_atoms,
        )  # Z

        self.envs = None
        self.actor_network = self.base_network.nn_model()
        self.target_network = tf.keras.models.clone_model(self.actor_network)
        self.target_network.set_weights(self.actor_network.get_weights())

        self.total_steps = 0
        self.episodes = config.episodes
        self.steps = config.steps
        self.batch_size = config.batch_size

        self.replay_buffer_size = config.replay_buffer_size
        self.replay_buffer = deque()

        self.delta_z = (config.categorical_Vmax - config.categorical_Vmin) / float(config.categorical_n_atoms - 1)

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
            print('Episode: {}  Reward: {} Max_Reward: {}'.format(each_ep, self.check_model_improved, self.best_max))
            print('-' * 64)
            self.check_model_improved = 0

            for step in range(self.steps):
                # reshape the input state to a tensor ===> Network ===> action probabilities
                # size = (1, action dimension, number of atoms)
                # e.g. size = (1, 2, 51)
                action_prob, _ = self.actor_network.predict(
                    np.array(current_state).reshape((1, self.input_dim[0], self.input_dim[1])))

                # calculate action value (Q-value)
                action_value = np.dot(np.array(action_prob), self.atoms)

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
                    self.check_model_improved += reward

            # for any episode where the reward is higher
            # we copy the actor network weights to the target network
            if self.check_model_improved > self.best_max:
                self.best_max = self.check_model_improved
                self.target_network.set_weights(self.actor_network.get_weights())

    def train_by_replay(self):
        """
        TD update by replaying the history.
        Implementation of algorithm 1 in the paper.
        """
        # step 1: generate replay samples (size = self.batch_size) from the replay buffer
        # e.g. uniform random replay or prioritize experience replay
        current_states, actions, next_states, rewards, terminals = \
            replay_fn.uniform_random_replay(self.replay_buffer, self.batch_size)

        # step 2:
        # generate next state probability, size = (batch_size, action_dimension, number_of_atoms)
        # e.g. (32, 2, 51) where batch_size =  32,
        # each batch contains 2 actions,
        # each action distribution contains 51 bins.
        prob_next, _ = self.target_network.predict(next_states)

        # step 3:
        # calculate next state Q values, size = (batch_size, action_dimension, 1).
        # e.g. (32, 2, 1), each action has one Q value.
        # then choose the higher value out of the 2 for each of the 32 batches.
        action_value_next = np.dot(np.array(prob_next), self.atoms)
        action_next = np.argmax(action_value_next, axis=1)

        # step 4:
        # use the optimal actions as index, pick out the probabilities of the optimal action
        prob_next = prob_next[np.arange(self.batch_size), action_next, :]

        # match the rewards from the memory to the same size as the prob_next
        rewards = np.tile(rewards.reshape(self.batch_size, 1), (1, self.n_atoms))

        # perform TD update
        discount_rate = self.config.discount_rate * (1 - terminals)
        atoms_next = rewards + np.dot(discount_rate.reshape(self.batch_size, 1),
                                      self.atoms.reshape(1, self.n_atoms))
        # constrain atoms_next to be within Vmin and Vmax
        atoms_next = np.clip(atoms_next, self.vmin, self.vmax)

        # calculate the floors and ceilings of atom_next
        b = (atoms_next - self.config.categorical_Vmin) / self.delta_z
        l, u = np.floor(b).astype(int), np.ceil(b).astype(int)

        # it is important to check if l == u, to avoid histogram collapsing.
        d_m_l = (u + (l == u) - b) * prob_next
        d_m_u = (b - l) * prob_next

        # step 5: redistribute the target probability histogram (calculation of m)
        # Note that there is an implementation issue
        # The loss function requires current histogram and target histogram to have the same size
        # Generally, the loss function should be the categorical cross entropy loss between
        # P(x, a*): size = (32, 1, 51) and P(x(t+1), a*): size = (32, 1, 51), i.e. only for optimal actions
        # However, the network generates P(x, a): size = (32, 2, 51), i.e. for all actions
        # Therefore, I create a tensor with zeros (size = (32, 2, 51)) and update only the probability histogram
        target_histo = np.zeros(shape=(self.batch_size, self.n_atoms))

        for i in range(self.batch_size):
            target_histo[i][action_next[i]] = 0.0  # clear the histogram that needs to be updated
            np.add.at(target_histo[i], l[i], d_m_l[i])  # update d_m_l
            np.add.at(target_histo[i], l[i], d_m_u[i])  # update d_m_u

        # update actor network weights
        self.actor_network.fit(x=current_states, y=target_histo, verbose=2, callbacks=self.keras_check)

    def eval_step(self, render=True):
        """
        Evaluation using the trained target network, no training involved
        :param render: whether to visualize the evaluation or not
        """
        for each_ep in range(self.config.evaluate_episodes):
            current_state = self.envs.reset()

            print('Episode: {} Reward: {} Training_Max_Reward: {}'.format(each_ep, self.check_model_improved,
                                                                          self.best_max))
            print('-' * 64)
            self.check_model_improved = 0

            for step in range(self.steps):
                action_prob, _ = self.target_network.predict(
                    np.array(current_state).reshape((1, self.input_dim[0], self.input_dim[1])))

                action_value = np.dot(np.array(action_prob), self.atoms)
                action = np.argmax(action_value[0])

                next_state, reward, done, _ = self.envs.step(action=action)

                if render:
                    self.envs.render(mode=['human'])

                if done:
                    break
                else:
                    current_state = next_state
                    self.check_model_improved += 1
