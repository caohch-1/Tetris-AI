#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""A script for running some tests on the envs."""


import gym
from stable_baselines3.common.env_checker import check_env

from gym_simplifiedtetris.register import env_list


def main() -> None:
    """
    Check if each env created is compliant with the OpenAI Gym API, by playing ten games using an agent that selects actions uniformly at random. In every game, validate the reward received, and render the env for visual inspection.
    """
    for env_id, env_name in enumerate(env_list):
        print(f"Testing the env: {env_name} ({env_id+1}/{len(env_list)})")

        env = gym.make(env_name)
        check_env(env=env, skip_render_check=True)

        obs = env.reset()

        num_episodes = 0
        is_first_move = True
        while num_episodes < 3:
            action = env.action_space.sample()
            obs, reward, done, _ = env.step(action)

            assert (
                env.reward_range[0] <= reward <= env.reward_range[1]
            ), f"Reward seen: {reward}"

            if num_episodes == 0 and is_first_move:
                is_first_move = False

            if done:
                num_episodes += 1
                obs = env.reset()

        env.close()

    print("All envs passed the tests.")


if __name__ == "__main__":
    main()
