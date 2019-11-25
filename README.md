# Not-so-deep reinforcement learning
A practice module of some RL algorithms, implemented in 
[Tensorflow Keras](https://www.tensorflow.org/guide/keras) 
and [OpenAI gym](https://github.com/openai/gym) framework. 

## General
I implement RL algorithms in this repo to better understand them.
 
## Environment:
So far I've only tested all algorithms on [CartPole](https://github.com/openai/gym/wiki/CartPole-v0).

 
## Algorithms:
- [x] A2C
- [ ] A3C
- [x] Deep Q Network
- [x] Categorical DQN (C51)
- [x] Quantile DQN
- [ ] Expectile DQN

## Prerequisites
* [Python 3.6](https://www.python.org/)
* [Numpy](http://www.numpy.org/)
* [Scipy](https://www.scipy.org/)
* [Tensorflow](https://www.tensorflow.org/)
* [OpenAI-gym](https://github.com/openai/gym)


```examples.py``` contains examples for all implemented algorithms.

## References
* [Reinforcement Learning](http://incompleteideas.net/book/RLbook2018.pdf)
* [Deep Q-Learning (DQN)](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)
* [Advantage Actor Critic (A2C)](https://papers.nips.cc/paper/1786-actor-critic-algorithms.pdf)
* (C51) [A Distributional Perspective on Reinforcement Learning](https://arxiv.org/abs/1707.06887)
* (Expectile) [Statistics and Samples in Distributional Reinforcement Learning](https://arxiv.org/abs/1902.08102)
* (Quantile) [Distributional Reinforcement Learning with Quantile Regression](https://arxiv.org/abs/1710.10044)
* [An Analysis of Categorical Distributional Reinforcement Learning](https://arxiv.org/abs/1802.08163)
* [Implicit Quantile Networks for Distributional Reinforcement Learning](https://arxiv.org/abs/1806.06923)
* [A Comparative Analysis of Expected and Distributional Reinforcement Learning](https://arxiv.org/abs/1806.06923)




## Acknowledgements:
* (PyTorch) Deep RL by [Shangtong Zhang](https://github.com/ShangtongZhang/DeepRL)
* (Keras) Deep-RL-Keras by [Hugo Germain](https://github.com/germain-hug/Deep-RL-Keras)