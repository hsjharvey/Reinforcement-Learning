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
                    current_states, actions, next_states, rewards, terminals = replay_fn.uniform_random_replay(
                        self.replay_buffer,
                        self.batch_size)
                    prob_next = self.target_network.predict(next_states)

                    q_next = np.dot(np.array(prob_next), self.atoms)
                    action_next = np.argmax(q_next, axis=1)

                    prob_next = prob_next[np.arange(self.batch_size), action_next, :]

                    rewards = np.tile(rewards.reshape(self.batch_size, 1), (1, self.config.Categorical_n_atoms))

                    discount_rate = self.config.discount_rate * (1 - terminals)
                    atoms_next = rewards + np.dot(discount_rate.reshape(self.batch_size, 1),
                                                  self.atoms.reshape(1, self.config.Categorical_n_atoms))

                    atoms_next = np.clip(atoms_next, self.config.Categorical_Vmin, self.config.Categorical_Vmax)

                    b = (atoms_next - self.config.Categorical_Vmin) / self.delta_z

                    l = np.floor(b).astype(int)
                    u = np.ceil(b).astype(int)
                    d_m_l = (u + (l == u) - b) * prob_next
                    d_m_u = (b - l) * prob_next

                    target_histo = np.zeros(prob_next.shape)
                    for i in range(self.batch_size):
                        np.add.at(target_histo[i], l[i], d_m_l[i])
                        np.add.at(target_histo[i], l[i], d_m_u[i])

                    log_prob = np.log(self.actor_network.predict(current_states))
                    log_prob = log_prob[np.arange(self.batch_size), actions, :]

                    self.actor_network.fit(x=current_states, y=prob_next)

                    # updates = optimizer.get_updates(loss=loss_fn, params=self.actor_network.trainable_variables)
                    # self.weights_update(loss_fn, self.actor_network, optimizer)

                current_state = next_state

    def eval_step(self):
        pass

    @tf.function
    def weights_update(self, loss, model, optimizer):
        print('loss {}'.format(loss))
        with tf.GradientTape() as tape:
            tape.watch(model.trainable_variables)
            regularization_loss = tf.math.add_n(model.losses)
            print(regularization_loss)

            total_loss = regularization_loss + loss

        gradients = tape.gradient(total_loss, model.trainable_variables)
        print(gradients)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))


if __name__ == '__main__':
    C = config.Config()
    envs = gym.make('CartPole-v0')
    cat = CategoricalDQNAgent(config=C)
    cat.envs = envs
    cat.actor_network = network_bodies.CategoricalNet(config=C).nn_model()
    cat.target_network = tf.keras.models.clone_model(cat.actor_network)
    cat.target_network.set_weights(cat.actor_network.get_weights())
    cat.actor_network.summary()
    cat.train_step()
