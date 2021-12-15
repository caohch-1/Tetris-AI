<p align="center">
  <img src="https://github.com/OliverOverend/gym-simplifiedtetris/raw/master/assets/20x10_4.gif" width="500">
</p>

<h1 align="center">Gym-SimplifiedTetris </h1>

<p align="center">
  <a href="https://www.codefactor.io/repository/github/oliveroverend/gym-simplifiedtetris">
    <img src="https://img.shields.io/codefactor/grade/github/OliverOverend/gym-simplifiedtetris?color=ff69b4&style=for-the-badge">
  </a>
  <a href="https://pypi.org/">
    <img src="https://img.shields.io/pypi/v/gym-simplifiedtetris?style=for-the-badge">
  </a>
  <a href="https://pypi.org/project/gym-simplifiedtetris/">
    <img src="https://img.shields.io/pypi/pyversions/gym-simplifiedtetris?style=for-the-badge">
  </a>
  <a href="/LICENSE.md">
    <img src="https://img.shields.io/github/license/OliverOverend/gym-simplifiedtetris?color=darkred&style=for-the-badge">
  </a>
  <a href="https://github.com/OliverOverend/gym-simplifiedtetris/commits/dev">
    <img src="https://img.shields.io/github/last-commit/OliverOverend/gym-simplifiedtetris/dev?style=for-the-badge">
  </a>
  <a href="https://github.com/OliverOverend/gym-simplifiedtetris/releases">
    <img src="https://img.shields.io/github/release-date/OliverOverend/gym-simplifiedtetris?color=teal  &style=for-the-badge">
  </a>
  <a href="https://github.com/OliverOverend/gym-simplifiedtetris/issues">
    <img src="https://img.shields.io/github/issues-raw/OliverOverend/gym-simplifiedtetris?color=blueviolet&style=for-the-badge">
  </a>
</p>

<p align="center">
  <a href="https://github.com/OliverOverend/gym-simplifiedtetris/issues/new?assignees=OliverOverend&labels=bug&late=BUG_REPORT.md&title=%5BBUG%5D%3A">Report Bug
  </a>
  ·
  <a href="https://github.com/OliverOverend/gym-simplifiedtetris/issues/new?assignees=OliverOverend&labels=enhancement&late=FEATURE_REQUEST.md&title=%5BFEATURE%5D%3A">Request Feature
  </a>
  ·
  <a href="https://github.com/OliverOverend/gym-simplifiedtetris/discussions/new">Suggestions
  </a>
</p>

---

> 🟥 Simplified Tetris environments compliant with OpenAI Gym's API

Gym-SimplifiedTetris is a pip installable package that creates simplified Tetris environments compliant with [OpenAI Gym's API](https://github.com/openai/gym). Currently, Gym's API is the field standard for developing and comparing reinforcement learning algorithms.

The environments implemented in this package are simplified because the player must select the column and piece's rotation before the piece is dropped vertically downwards.  If one looks at the previous approaches to the game of Tetris, most of them use this simplified setting.

---

- [1. Installation](#1-installation)
- [2. Usage](#2-usage)
- [3. Agents and environments](#3-agents-and-environments)
- [4. Future work](#4-future-work)
- [5. Acknowledgements](#5-acknowledgements)

## 1. Installation

The package is pip installable:
```bash
pip install gym-simplifiedtetris
```

Or, you can copy the repository by forking it and then downloading it using:

```bash
git clone https://github.com/INSERT-YOUR-USERNAME-HERE/gym-simplifiedtetris
```

Packages can be installed using pip:

```bash
cd gym-simplifiedtetris
pip install -r requirements.txt
```

Here is a list of dependencies:

- numpy
- gym
- tqdm
- opencv_python
- matplotlib
- Pillow
- stable_baselines3
- dataclasses

## 2. Usage

The file [examples.py](https://github.com/OliverOverend/gym-simplifiedtetris/blob/master/examples.py) shows two examples of using an instance of the `simplifiedtetris-binary-20x10-4-v0` environment for ten games. You can create an environment using `gym.make`, supplying the environment's ID as an argument.

```python
import gym
import gym_simplifiedtetris

env = gym.make("simplifiedtetris-binary-20x10-4-v0")
obs = env.reset()

# Run 10 games of Tetris, selecting actions uniformly at random.
episode_num = 0
while episode_num < 10:
    env.render()
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)

    if done:
        print(f"Episode {episode_num + 1} has terminated.")
        episode_num += 1
        obs = env.reset()

env.close()
```

Alternatively, you can import the environment directly:

```python
from gym_simplifiedtetris.envs import SimplifiedTetrisBinaryEnv as Tetris

env = Tetris(grid_dims=(20, 10), piece_size=4)
```

## 3. Agents and environments

Three agents — described in [gym_simplifiedtetris/agents](https://github.com/OliverOverend/gym-simplifiedtetris/blob/master/gym_simplifiedtetris/agents) — are provided. There are currently 64 environments provided; a description can be found in [gym_simplifiedtetris/envs](https://github.com/OliverOverend/gym-simplifiedtetris/blob/master/gym_simplifiedtetris/envs).

## 4. Future work

- Normalise the observation spaces
- Implement an action space that only permits non-terminal actions to be taken
- Implement more shaping rewards: potential-style, potential-based, dynamic potential-based, and non-potential. Optimise their weights using an optimisation algorithm.

## 5. Acknowledgements

This package utilises several methods from the [codebase](https://github.com/andreanlay/tetris-ai-deep-reinforcement-learning) developed by andreanlay (2020) and the [codebase](https://github.com/Benjscho/gym-mdptetris) developed by Benjscho (2021).
