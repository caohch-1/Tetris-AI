"""Contains a function that evaluates an agent."""

from typing import Tuple

import gym
import numpy as np
from tqdm import tqdm

from gym_simplifiedtetris.agents import QLearningAgent


def eval_agent(
    agent: QLearningAgent, env: gym.Env, num_episodes: int, render: bool
) -> Tuple[float, float]:
    """
    Evaluate the agent's performance on the game of Tetris and return the mean
    score and standard deviation.

    :param agent: the agent to evaluate on the env.
    :param env: the agent will be evaluated on this env.
    :param num_episodes: the number of games to evaluate the trained agent.
    :param render: renders the agent playing SimplifiedTetris after training.
    :return: the mean and std score obtained from letting the agent play num_episodes games.
    """
    ep_returns = np.zeros(num_episodes, dtype=int)
    env._engine._final_scores = np.array([], dtype=int)

    for episode_id in tqdm(range(num_episodes), desc="No. of episodes completed"):

        obs = env.reset()
        done = False

        while not done:

            if render:
                env.render()

            action = agent.predict(obs)

            obs, _, done, info = env.step(action)
            ep_returns[episode_id] += info["num_rows_cleared"]

    env.close()

    mean_score = np.mean(ep_returns)
    std_score = np.std(ep_returns)

    print(
        f"""\nScore obtained from averaging over {num_episodes} games:\nMean = {np.mean(ep_returns):.1f}\nStandard deviation = {np.std(ep_returns):.1f}"""
    )

    return mean_score, std_score
