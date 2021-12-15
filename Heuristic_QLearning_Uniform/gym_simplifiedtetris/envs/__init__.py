"""Initialise the envs package."""

from gym_simplifiedtetris.envs.simplified_tetris_binary_env import (
    SimplifiedTetrisBinaryEnv,
)
from gym_simplifiedtetris.envs.simplified_tetris_engine import SimplifiedTetrisEngine
from gym_simplifiedtetris.envs.simplified_tetris_base_env import SimplifiedTetrisBaseEnv
from gym_simplifiedtetris.envs.simplified_tetris_part_binary_env import (
    SimplifiedTetrisPartBinaryEnv,
)
from gym_simplifiedtetris.envs.reward_shaping import (
    SimplifiedTetrisBinaryShapedEnv,
    SimplifiedTetrisPartBinaryShapedEnv,
)


# Affect 'from envs import *'.
__all__ = [
    "SimplifiedTetrisBinaryEnv",
    "SimplifiedTetrisEngine",
    "SimplifiedTetrisBinaryShapedEnv",
    "SimplifiedTetrisPartBinaryEnv",
    "SimplifiedTetrisPartBinaryShapedEnv",
]
