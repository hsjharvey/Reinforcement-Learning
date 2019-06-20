# -*- coding:utf-8 -*-
import numpy as np


def epsilon_greedy(action_values, episode=0, stop_explore=10):
    """
    exponential decay exploration E-greedy
    :param action_values:
    :param episode:
    :param stop_explore:
    :return:
    """

    if episode < stop_explore:
        random_draw = np.random.random()
        exploration = 0.9 ** np.exp(-0.1 * episode)

        if random_draw < exploration:
            return np.random.random_integers(0, 1)

        else:
            return np.argmax(action_values, axis=0)
    else:
        return np.argmax(action_values, axis=0)
