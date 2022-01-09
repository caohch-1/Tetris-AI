#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""A script for running a heuristic agent."""


import numpy as np
from gym_simplifiedtetris.envs import SimplifiedTetrisBinaryEnv as Tetris


def predict(ratings_or_priorities: np.array) -> int:
    """
    Return that action yielding the largest heuristic score. Separate ties
    using a priority rating, which is based on the translation and
    rotation.

    :param ratings_or_priorities: either the ratings or priorities for all available actions.
    :return: the action with the largest rating; ties are separated based on the priority.
    """
    return np.argmax(ratings_or_priorities)

def main() -> None:
    """Evaluate the agent that selects action according to a heuristic."""
    num_episodes = 30
    episode_num = 0
    ep_returns = np.zeros(num_episodes)

    env = Tetris(grid_dims=(10, 10), piece_size=4)

    obs = env.reset()

    while episode_num < num_episodes:
        env.render()

        heuristic_scores = env._engine._get_dellacherie_scores()
        print(heuristic_scores)
        action = predict(heuristic_scores)

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
