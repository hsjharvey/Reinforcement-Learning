# -*- coding:utf-8 -*-
from utils import policies, config
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

                self.replay_buffer.append([current_state, action, next_state, reward, done])

                if len(list(self.replay_buffer)) == self.replay_buffer_size:
                    print(list(self.replay_buffer)[2])
                    exp = np.random.choice(list(self.replay_buffer), size=self.batch_size)
                    next_states = exp[:, 2]
                    prob_next = self.target_network.predict(next_states)
                    print(prob_next)

                    # rewards = exp[:, 3]

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