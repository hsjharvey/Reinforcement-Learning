# -*- coding:utf-8 -*-
from utils import policies, config, replay_fn
from network import network_bodies
import numpy as np
import tensorflow as tf
from collections import deque
from copy import copy, deepcopy
import gym


class CategoricalDQNAgent:
    def __init__(self, config):

        self.config = config

        self.atoms = np.linspace(
            config.Categorical_Vmin,
            config.Categorical_Vmax,
            config.Categorical_n_atoms,
        )  # Z
        self.envs = None
        self.actor_network = None
        self.target_network = None
        self.episodes = config.episodes
        self.steps = config.steps
        self.batch_size = config.batch_size

        self.replay_buffer_size = config.replay_buffer_size
        self.replay_buffer = deque(maxlen=self.replay_buffer_size)

        self.delta_z = (config.Categorical_Vmax - config.Categorical_Vmin) / float(config.Categorical_n_atoms - 1)

    def train_step(self):
        for each_ep in range(self.episodes):
            current_state = self.envs.reset()

            for step in range(self.steps):

                action_prob = self.actor_network.predict(np.array(current_state).reshape(1, 1, 4))
                action_value = np.dot(np.array(action_prob), self.atoms)
                action = policies.epsilon_greedy(action_value=action_value[0])

                next_state, reward, done, _ = self.envs.step(action=action)

                self.replay_buffer.append([current_state.reshape(self.config.input_dim).tolist(), action,
                                           next_state.reshape(self.config.input_dim).tolist(), reward, done])

                if len(list(self.replay_buffer)) == self.replay_buffer_size:
                    next_states, rewards, terminals = replay_fn.uniform_random_replay(self.replay_buffer,
                                                                                      self.batch_size)
                    prob_next = self.target_network.predict(np.asarray(next_states))

                    print('=' * 64)
                    print(np.array(prob_next).shape)
                    print(self.atoms.shape)
                    q_next = np.dot(np.array(prob_next), self.atoms)
                    action_next = np.argmax(q_next, axis=1)
                    print(action_next.shape)
                    prob_next = prob_next[np.arange(self.batch_size), action_next, :]
                    print('prob_next {}'.format(prob_next.shape))

                    print(q_next.shape)
                    print(rewards.shape)
                    print(terminals.shape)
                    print(self.atoms.shape)
                    print(self.config.discount_rate * (1 - terminals))
                    rewards = np.tile(rewards.reshape(self.batch_size, 1), (1, self.config.Categorical_n_atoms))
                    print(rewards.shape)

                    discount_rate = self.config.discount_rate * (1 - terminals)
                    atoms_next = rewards + np.dot(discount_rate.reshape(self.batch_size, 1),
                                                  self.atoms.reshape(1, self.config.Categorical_n_atoms))

                    atoms_next = np.clip(atoms_next, self.config.Categorical_Vmin, self.config.Categorical_Vmax)

                    b = (atoms_next - self.config.Categorical_Vmin) / self.delta_z

                    l = np.floor(b)
                    u = np.ceil(b)
                    d_m_l = (u + (l == u) - b)
                    print('dml {}'.format(d_m_l.shape))

                current_state = next_state

    def eval_step(self):
        pass


if __name__ == '__main__':
    C = config.Config()
    envs = gym.make('CartPole-v0')
    cat = CategoricalDQNAgent(config=C)
    cat.envs = envs
    cat.actor_network = network_bodies.CategoricalNet(config=C).nn_model()
    cat.target_network = network_bodies.CategoricalNet(config=C).nn_model()
    cat.actor_network.summary()
    cat.train_step()
