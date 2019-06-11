# -*- coding:utf-8 -*-
import numpy as np


def epsilon_greedy(action_value):
    temp = np.random.random()
    if temp < 0.1:
        return np.random.random_integers(0, 1)

    else:
        return np.argmax(action_value, axis=0)
