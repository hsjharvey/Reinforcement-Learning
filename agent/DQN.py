# -*- coding:utf-8 -*-
from utils import policies, config, replay_fn
from network import neural_network
import numpy as np
import tensorflow as tf
from collections import deque
import gym
import time


class DQN:
    def __init__(self, config, base_network):
        self.base_network = base_network
        self.config = config

        self.input_dim = config.input_dim  # neural network input dimension

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

        self.check = 0
        self.best_max = 0

    def transition(self):
        for each_ep in range(self.episodes):
            current_state = self.envs.reset()

            print('max_step: {}'.format(self.check))
            self.check = 0

            for step in range(self.steps):
                action_values, _ = self.actor_network.predict(
                    np.array(current_state).reshape((1, self.input_dim[0], self.input_dim[1])))

                action = policies.epsilon_greedy(action_values=action_values.reshape(self.config.action_dim),
                                                 episode=each_ep,
                                                 stop_explore=self.config.stop_explore)

                next_state, reward, done, _ = self.envs.step(action=action)

                # record the history to replay buffer
                self.replay_buffer.append([current_state.reshape(self.input_dim).tolist(), action,
                                           next_state.reshape(self.input_dim).tolist(), reward, done])

                # when we collect certain number of batches, perform replay and update
                # the weights in actor network and clear the replay buffer
                if each_ep > self.config.stop_explore and len(list(self.replay_buffer)) >= self.replay_buffer_size:
                    self.train_by_replay()
                    self.replay_buffer = deque()

                # if episode is finished, break the inner loop
                # otherwise, continue
                if done:
                    self.total_steps += 1
                    break

                else:
                    current_state = next_state
                    self.check += 1
                    self.total_steps += 1

            # for certain period, we copy the actor network weights to the target network
            if self.check >= self.best_max:
                self.best_max = self.check
                self.target_network.set_weights(self.actor_network.get_weights())

    def train_by_replay(self):
        # step 1: generate replay samples (size = self.batch_size) from the replay buffer
        # e.g. prioritize experience replay
        current_states, actions, next_states, rewards, terminals = \
            replay_fn.uniform_random_replay(self.replay_buffer, self.batch_size)

        action_values, _ = self.target_network.predict(next_states)
        action_values_next = np.max(action_values, axis=2)

        rewards = rewards.reshape(action_values_next.shape)
        terminals = terminals.reshape((action_values_next.shape))

        # TD update
        action_values_next = rewards + self.config.discount_rate * action_values_next * (1 - terminals)

        self.actor_network.fit(x=current_states, y=action_values_next, verbose=2)

    def eval_step(self, render=True):
        for each_ep in range(100):
            current_state = self.envs.reset()

            print('max_step: {}'.format(self.check))
            self.check = 0

            for step in range(200):
                action_values, _ = self.target_network.predict(
                    np.array(current_state).reshape((1, self.input_dim[0], self.input_dim[1])))
                action = np.argmax(action_values.reshape(self.config.action_dim))

                next_state, reward, done, _ = self.envs.step(action=action)

                if render:
                    self.envs.render(mode=['human'])

                if done:
                    break
                else:
                    current_state = next_state
                    self.check += 1


if __name__ == '__main__':
    C = config.Config()
    quant = DQN(config=C, base_network=neural_network.DQNNet(config=C))
    quant.envs = gym.make('CartPole-v0')
    quant.transition()

    print("finish training")
    print("evaluating.....")
    quant.eval_step(render=True)