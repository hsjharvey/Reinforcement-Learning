# -*- coding:utf-8 -*-
from src import *


def run_DQN_example():
    C = config.Config()
    base_network = DQN_net.DQNNet(config=C)
    cat = DQNAgent(config=C, base_network=base_network)
    cat.envs = gym.make('CartPole-v0')

    cat.transition()
    print("finish training")
    print('=' * 64)
    print("evaluating.....")
    cat.eval_step(render=True)


def run_CategoricalDQN_example():
    C = config.Config()
    base_network = CategoricalDQN_net.CategoricalNet(config=C)
    cat = CategoricalDQNAgent(config=C, base_network=base_network)
    cat.envs = gym.make('CartPole-v0')

    cat.transition()
    print("finish training")
    print('=' * 64)
    print("evaluating.....")
    cat.eval_step(render=True)


def run_QuantileDQN_example():
    C = config.Config()
    base_network = QuantileDQN_net.QuantileNet(config=C)
    cat = QuantileDQNAgent(config=C, base_network=base_network)
    cat.envs = gym.make('CartPole-v0')

    cat.transition()
    print("finish training")
    print('=' * 64)
    print("evaluating.....")
    cat.eval_step(render=True)


if __name__ == '__main__':
    # run_DQN_example()
    # run_CategoricalDQN_example()
    run_QuantileDQN_example()
