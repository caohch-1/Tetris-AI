#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""A script for running training and evaluating a Q-learning agent."""


from gym_simplifiedtetris.agents import QLearningAgent
from gym_simplifiedtetris.envs import SimplifiedTetrisBinaryEnv as Tetris
from gym_simplifiedtetris.helpers import train_q_learning, eval_agent


def main() -> None:
    """Train and evaluate a Q-learning agent."""
    grid_dims = (7, 4)
    env = Tetris(grid_dims=grid_dims, piece_size=3)
    agent = QLearningAgent(
        grid_dims=grid_dims, num_pieces=env._num_pieces_, num_actions=env._num_actions_
    )

    agent = train_q_learning(env=env, agent=agent, num_eval_timesteps=100, render=True)
    eval_agent(agent=agent, env=env, num_episodes=30, render=True)


if __name__ == "__main__":
    main()
