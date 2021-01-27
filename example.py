# -*- coding:utf-8 -*-
from src import *


def run_DQN_example(game):
    C = config.Config()
    base_network = DQN_net.DQNNet(config=C)
    cat = DQNAgent(config=C, base_network=base_network)
    cat.envs = gym.make(game)

    cat.transition()
    print("finish training")
    print('=' * 64)
    print("evaluating.....")
    cat.eval_step(render=True)


def run_CategoricalDQN_example(game):
    C = config.Config()
    base_network = CategoricalDQN_net.CategoricalNet(config=C)
    cat = CategoricalDQNAgent(config=C, base_network=base_network)
    cat.envs = gym.make(game)

    cat.transition()
    print("finish training")
    print('=' * 64)
    print("evaluating.....")
    cat.eval_step(render=True)


def run_QuantileDQN_example(game):
    C = config.Config()
    base_network = QuantileDQN_net.QuantileNet(config=C)
    Quant = QuantileDQNAgent(config=C, base_network=base_network)
    Quant.envs = gym.make(game)

    Quant.transition()
    print("finish training")
    print('=' * 64)
    print("evaluating.....")
    Quant.eval_step(render=True)


def run_ExpectileDQN_example(game):
    C = config.Config()
    base_network = ExpectileDQN_net.ExpectileNet(config=C)
    expect = ExpectileDQNAgent(config=C, base_network=base_network)
    expect.envs = gym.make(game)

    expect.transition()
    print("finish training")
    print('=' * 64)
    print("evaluating.....")
    expect.eval_step(render=True)


def run_A2C_example(game):
    C = config.Config()
    base_network = A2C_net.ActorCriticNet(config=C)
    A2C = A2Cagent(config=C, base_network=base_network)
    A2C.envs = gym.make(game)

    A2C.transition()
    print("finish training")
    print('=' * 64)
    print("evaluating.....")
    A2C.eval_step(render=True)


if __name__ == '__main__':
    game = 'CartPole-v0'

    run_DQN_example(game)
    # run_CategoricalDQN_example(game)
    # run_QuantileDQN_example(game)
    # run_ExpectileDQN_example(game)
    # run_A2C_example(game)
