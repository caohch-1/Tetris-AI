# Agents<!-- omit in toc -->

- [1. Uniform](#1-uniform)
- [2. Q-learning](#2-q-learning)
- [3. Heuristic](#3-heuristic)

## 1. Uniform

The uniform agent selects actions uniformly at random. See [run_uniform_agent.py](https://github.com/OliverOverend/gym-simplifiedtetris/blob/master/run_uniform_agent.py) for an example of how to use it.

<p align="center">
    <img src="https://github.com/OliverOverend/gym-simplifiedtetris/raw/master/assets/20x10_4.gif" width="500">
</p>

## 2. Q-learning

Due to the curse of dimensionality, this agent struggles to learn as the grid's dimensions are increased; the size of the state-action space grows exponentially. The exploration rate parameter, epsilon, is linearly annealed over the training period.  Following the training period, the Q-learning agent selects the action with the highest state-action value.  See [run_q_learning_agent.py](https://github.com/OliverOverend/gym-simplifiedtetris/blob/master/run_q_learning_agent.py) for an example of how to use it.

<p align="center">
    <img src="https://github.com/OliverOverend/gym-simplifiedtetris/raw/master/assets/7x4_3_q_learning.gif" width="500">
</p>

## 3. Heuristic

The heuristic agent selects the action with the highest heuristic score, based on the [Dellacherie feature set](https://arxiv.org/abs/1905.01652).  The heuristic score for each possible action is computed using the following heuristic, crafted by Pierre Dellacherie [Colin Fahey's website](https://colinfahey.com):

***- landing height + eroded cells - row transitions - column transitions -4 x holes - cumulative wells***

Similarly to how Colin Fahey implemented Dellacherie's agent, ties are broken by selecting the action with the largest priority.  Deviations from and to the left of the centre of the grid are rewarded, and rotations are punished.  See [run_heuristic_agent.py](https://github.com/OliverOverend/gym-simplifiedtetris/blob/master/run_heuristic_agent.py) for an example of how to use it.

<p align="center">
    <img src="https://github.com/OliverOverend/gym-simplifiedtetris/raw/master/assets/20x10_4_heuristic.gif" width="500">
</p>