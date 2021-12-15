"""Contains a simplified Tetris env with a binary obs space and shaped reward function."""

from gym_simplifiedtetris.register import register
from gym_simplifiedtetris.envs.simplified_tetris_binary_env import (
    SimplifiedTetrisBinaryEnv,
)
from .potential_based_shaping_reward import PotentialBasedShapingReward


class SimplifiedTetrisBinaryShapedEnv(
    PotentialBasedShapingReward, SimplifiedTetrisBinaryEnv
):
    """
    A simplified Tetris environment, where the reward function is a
    potential-based shaping reward and the observation space is the grid's
    binary representation plus the current piece's id.

    :param grid_dims: the grid's dimensions.
    :param piece_size: the size of the pieces in use.
    :param seed: the rng seed.
    """

    def __init__(self, **kwargs):
        """Extend the two superclasses."""
        super().__init__()
        SimplifiedTetrisBinaryEnv.__init__(self, **kwargs)


register(
    incomplete_id="simplifiedtetris-binary-shaped",
    entry_point="gym_simplifiedtetris.envs:SimplifiedTetrisBinaryShapedEnv",
)
