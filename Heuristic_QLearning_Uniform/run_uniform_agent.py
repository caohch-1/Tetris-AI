#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""A script for running a uniform agent on Tetris."""

import gym
import numpy as np

from gym_simplifiedtetris.agents import UniformAgent


def main() -> None:
    """Run ten games, selecting actions uniformly at random."""
    ep_returns = np.zeros(10)

    env = gym.make("simplifiedtetris-binary-20x10-4-v0")
    agent = UniformAgent(env.action_space.n)

    obs = env.reset()

    num_episodes = 0
    while num_episodes < 10:
        env.render()
        action = agent.predict()
        obs, reward, done, info = env.step(action)
        ep_returns[num_episodes] += info["num_rows_cleared"]

        if done:
            print(f"Episode {num_episodes + 1} has terminated.")
            num_episodes += 1
            obs = env.reset()

    env.close()

    print(
        f"""\nScore obtained from averaging over {num_episodes} games:\nMean = {np.mean(ep_returns):.1f}\nStandard deviation = {np.std(ep_returns):.1f}"""
    )


if __name__ == "__main__":
    main()
