# -*- coding:utf-8 -*-
from src.utils import *
import numpy as np
from scipy.optimize import minimize, root
import tensorflow as tf
from collections import deque
import gym


class ExpectileDQNAgent:
    def __init__(self, config, base_network):
        self.base_network = base_network
        self.config = config

        self.input_dim = config.input_dim  # neural network input dimension

        self.num_expectiles = config.num_expectiles
        self.expectile_mean_idx = int(config.num_expectiles / 2) + 1

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

        self.keras_check = config.keras_checkpoint

        self.check_model_improved = 0
        self.best_max = 0
        # note that tau_6 = 0.5 and thus this expectile statistic is in fact the mean
        # tau
        self.cum_density = np.linspace(0.01, 0.99, config.num_expectiles)
        self.imputation_method = config.imputation_method

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
                # neural network returns quantile value
                # action value (Q): middle of all expectile values
                expectile_values, _ = self.actor_network.predict(
                    np.array(current_state).reshape((1, self.input_dim[0], self.input_dim[1])))
                action_value = expectile_values[0, :, self.expectile_mean_idx]

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
        """
        # step 1: generate replay samples (size = self.batch_size) from the replay buffer
        # e.g. uniform random replay or prioritize experience replay
        current_states, actions, next_states, rewards, terminals = \
            replay_fn.uniform_random_replay(self.replay_buffer, self.batch_size)

        # step 2: get the next state expectiles
        # and choose the optimal actions from next state quantiles
        expectile_next, _ = self.target_network.predict(next_states)

        # different from the quantile approach, in which the q values are calculated by mean over quantiles
        # in expectile approach, the middle expectile value is the mean (i.e. the action value)
        action_value_next = expectile_next[:, :, self.expectile_mean_idx]
        action_next = np.argmax(action_value_next, axis=1)

        # choose the optimal expectile next
        expectile_next = expectile_next[np.arange(self.batch_size), action_next, :]

        # The following part corresponds to Algorithm 2 in the paper
        # after getting the target expectile (or expectile_next), we need to impute the distribution
        # from the target expectile. This imputation step effectively re-generate the distribution
        # from the statistics (expectile)
        # Note that in the paper the authors assume dirac form to approximate a continuous PDF.
        # Therefore, the following steps generate several points on the x-axis of the PDF, each with an equal height
        # A visualization of this process is in figure 10 of the appendix section A of the paper
        z = self.imputation_strategy(expectile_next)

        # match the rewards and the discount rates from the memory to the same size as the expectile_next
        rewards = np.tile(rewards.reshape(self.batch_size, 1), (1, self.num_expectiles))
        discount_rate = np.tile(self.config.discount_rate * (1 - terminals).reshape(self.batch_size, 1),
                                (1, self.num_expectiles))

        # TD update
        z = rewards + discount_rate * z

        # update actor network weights
        self.actor_network.fit(x=current_states, y=z, verbose=2, callbacks=self.keras_check)

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
                expectile_value, _ = self.target_network.predict(
                    np.array(current_state).reshape((1, self.input_dim[0], self.input_dim[1])))
                action_value = expectile_value[:, :, self.expectile_mean_idx]

                action = np.argmax(action_value[0])

                next_state, reward, done, _ = self.envs.step(action=action)

                if render:
                    self.envs.render(mode=['human'])

                if done:
                    break
                else:
                    current_state = next_state
                    self.check_model_improved += 1

    def imputation_strategy(self, expectile_next_batch):
        result_collection = np.zeros(shape=(self.batch_size, self.num_expectiles))
        for idx in range(self.batch_size):
            start_vals = np.linspace(self.config.z_val_limits[0], self.config.z_val_limits[1], self.num_expectiles)

            if self.imputation_method == "minimization":
                # To be discussed, I think this is pretty much problem-dependent
                # The bounds here limit the possible options of z
                # Having bounds could potentially prevent crazy z
                bnds = self.config.imputation_distribution_bounds
                optimization_results = minimize(self.minimize_objective_fc, args=(expectile_next_batch[idx, :]),
                                                x0=start_vals, bounds=bnds, method="SLSQP")
            elif self.imputation_method == "root":
                optimization_results = root(self.root_objective_fc, args=(expectile_next_batch[idx, :]), x0=start_vals)

            result_collection[idx, :] = optimization_results.x
        return result_collection

    def minimize_objective_fc(self, x, expect_set):
        vals = 0
        for idx, each_expectile in enumerate(expect_set):
            diff = x - each_expectile
            diff = np.where(diff > 0, - self.cum_density[idx] * diff, (self.cum_density[idx] - 1) * diff)
            vals += np.square(np.mean(diff))

        return vals

    def root_objective_fc(self, x, expect_set):
        vals = []
        for idx, each_expectile in enumerate(expect_set):
            diff = x - each_expectile
            diff = np.where(diff > 0, - self.cum_density[idx] * diff, (self.cum_density[idx] - 1) * diff)
            vals.append(np.mean(diff))
        return vals
