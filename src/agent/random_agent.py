# -*- coding:utf-8 -*-
import gym
import time
import numpy as np

if __name__ == '__main__':

    envs = gym.make('CartPole-v0')

    check = 0
    for each_ep in range(1000):
        current_state = envs.reset()
        print('max_step: {}'.format(check))
        check = 0

        for each_step in range(1000):
            check += 1
            envs.render(mode=['human'])
            action = np.random.randint(0, 2)  # random action, either left or right

            next_state, reward, done, _ = envs.step(action=action)

            if done:
                break
            else:
                current_state = next_state
