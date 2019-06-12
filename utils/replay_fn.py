# -*- coding:utf-8 -*-
import numpy as np
import random


def uniform_random_replay(replay_buffer, batch_size):
    current_states = []
    actions = []
    next_states = []
    rewards = []
    done = []
    other_info = []

    replay_buffer = list(replay_buffer)
    idx = random.sample(range(len(replay_buffer)), batch_size)

    for i in idx:
        current_states.append(replay_buffer[i][0])
        actions.append(replay_buffer[i][1])
        next_states.append(replay_buffer[i][2])
        rewards.append(replay_buffer[i][3])
        done.append(replay_buffer[i][4])

    return np.asarray(next_states), np.asarray(rewards), np.array(done)
