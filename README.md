# Not-so-deep reinforcement learning
A practice module of some RL algorithms, implemented in 
[Tensorflow Keras](https://www.tensorflow.org/guide/keras) 
and [OpenAI gym](https://github.com/openai/gym) framework. 


## General:
- I implement RL algorithms in this repo to better understand them. 
- I tend to focus on the RL part rather than the network structure. Therefore, the network structure of the algorithms in this report is pretty simple.
- My own research is related to distributional RL so I spend most of my effort implementing distributional RL algorithms. 
- Implementation of neural networks has improved/changed dramatically over the past three years. Unfortunately I was not able to keep up with the trends since my own research did not focus on the network structure. For sure some code are not efficient, everytime there is a major shift in the packages (OpenAI gym, tensorflow), I merely change the code so that they can work.
- I hope the generic algorithm structure helps.

 
## Environment:
So far I've only tested all algorithms on [CartPole](https://github.com/openai/gym/wiki/CartPole-v0).

 
## Algorithms:
- [x] A2C
- [ ] A3C
- [x] Deep Q Network
- [x] Categorical DQN (C51)
- [x] Quantile DQN
- [x] Expectile DQN

## Prerequisites (testing environment)
* [Python 3.8](https://www.python.org/)
* [Numpy](http://www.numpy.org/)
* [Scipy](https://www.scipy.org/)
* [Tensorflow](https://www.tensorflow.org/)
* [OpenAI-gym](https://github.com/openai/gym)


[```examples.py```](./example.py) contains examples for all implemented algorithms.

## References
* [Reinforcement Learning](http://incompleteideas.net/book/RLbook2018.pdf)
* [Deep Q-Learning (DQN)](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)
* [Advantage Actor Critic (A2C)](https://papers.nips.cc/paper/1786-actor-critic-algorithms.pdf)
* (C51) [A Distributional Perspective on Reinforcement Learning](https://arxiv.org/abs/1707.06887)
* (ER-DQN) [Statistics and Samples in Distributional Reinforcement Learning](https://arxiv.org/abs/1902.08102)
* (QR-DQN) [Distributional Reinforcement Learning with Quantile Regression](https://arxiv.org/abs/1710.10044)
* [An Analysis of Categorical Distributional Reinforcement Learning](https://arxiv.org/abs/1802.08163)
* [Implicit Quantile Networks for Distributional Reinforcement Learning](https://arxiv.org/abs/1806.06923)
* [A Comparative Analysis of Expected and Distributional Reinforcement Learning](https://arxiv.org/abs/1806.06923)
* [Fully Parameterized Quantile Function for Distributional Reinforcement Learning](https://arxiv.org/abs/1911.02140)
* [Non-crossing quantile regression for deep reinforcement learning](https://proceedings.neurips.cc//paper/2020/file/b6f8dc086b2d60c5856e4ff517060392-Paper.pdf)
* [Distributional Reinforcement Learning with Maximum Mean Discrepancy](https://arxiv.org/abs/2007.12354)




## Acknowledgements:
* (PyTorch) Deep RL by [Shangtong Zhang](https://github.com/ShangtongZhang/DeepRL)
* (Keras) Deep-RL-Keras by [Hugo Germain](https://github.com/germain-hug/Deep-RL-Keras)