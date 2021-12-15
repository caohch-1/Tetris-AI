#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""A script for running a heuristic agent."""


import numpy as np

from gym_simplifiedtetris.agents import HeuristicAgent
from gym_simplifiedtetris.envs import SimplifiedTetrisBinaryEnv as Tetris


def main() -> None:
    """Evaluate the agent that selects action according to a heuristic."""
    num_episodes = 30
    episode_num = 0
    ep_returns = np.zeros(num_episodes)

    agent = HeuristicAgent()
    env = Tetris(grid_dims=(10, 10), piece_size=4)

    obs = env.reset()

    while episode_num < num_episodes:
        env.render()

        heuristic_scores = env._engine._get_dellacherie_scores()
        action = agent.predict(heuristic_scores)

        obs, reward, done, info = env.step(action)
        ep_returns[episode_num] += info["num_rows_cleared"]

        if done:
            print(f"Episode {episode_num + 1} has terminated.")
            episode_num += 1
            obs = env.reset()

    env.close()

    print(
        f"""\nScore obtained from averaging over {num_episodes} games:\nMean = {np.mean(ep_returns):.1f}\nStandard deviation = {np.std(ep_returns):.1f}"""
    )


if __name__ == "__main__":
    main()
