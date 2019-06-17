# -*- coding:utf-8 -*-
import tensorflow as tf
import gym
import numpy as np

from collections import deque

envs = gym.make('CartPole-v0')
batch_size = 32
state_dim = 4  # len(envs.observation_space.low)
action_dim = 2  # envs.action_space.n
num_of_atoms = 51
output_dim = action_dim * num_of_atoms

NN_model = tf.keras.models.Sequential([
    # output layer
    tf.keras.layers.Dense(units=output_dim,
                          activation='softmax',
                          use_bias=True,
                          kernel_initializer='random_uniform',
                          activity_regularizer=tf.keras.regularizers.l1_l2(1e-2, 1e-2)
                          )

])

NN_model.compile(optimizer='adam',
                 loss='crossentropy',
                 metrics='accuracy')

NN_model.summary()

batch = deque([])

for episode in range(1000):
    current_state = envs.reset()
    action = np.random.random_integers(0, 1)
    experience_buffer = []

    for step in range(1000):
        next_state, reward, done, _ = envs.step(action=action)

        # in the future replay bt replay_function
        experience_buffer.append([current_state, action, reward, next_state, done])

    if episode % 100 == 0:
        pass


def train_actor(model):
    pass


def epsilon_greedy(action_prob):
    temp = np.random.random()
    if temp < 0.1:
        return np.random.random_integers(0, 1)

    else:
        return tf.argmax(action_prob, axis=0)


optimizer = tf.keras.optimizers.Adam(0.001)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
