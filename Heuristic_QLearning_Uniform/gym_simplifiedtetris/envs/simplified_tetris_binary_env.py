"""Contains a simplified Tetris env class with a binary obs space.
"""

import numpy as np
from gym import spaces

from gym_simplifiedtetris.register import register
from gym_simplifiedtetris.envs.simplified_tetris_base_env import SimplifiedTetrisBaseEnv


class SimplifiedTetrisBinaryEnv(SimplifiedTetrisBaseEnv):
    """
    A custom Gym env for Tetris, where the observation space is the binary
    representation of the grid plus the current piece's id.

    :param grid_dims: the grid dimensions.
    :param piece_size: the size of every piece.
    :param seed: the rng seed.
    """

    @property
    def observation_space(self) -> spaces.Box:
        """
        Override the superclass property.

        :return: a Box obs space.
        """
        return spaces.Box(
            low=np.append(np.zeros(self._width_ * self._height_), 0),
            high=np.append(
                np.ones(self._width_ * self._height_), self._num_pieces_ - 1
            ),
            dtype=np.int,
        )

    def _get_obs(self) -> np.array:
        """
        Override superclass method and return a flattened NumPy array
        containing the grid's binary representation, plus the current piece's
        id.

        :return: the current obs.
        """
        current_grid = self._engine._grid.flatten()

        return np.append(current_grid, self._engine._piece._idx)


register(
    incomplete_id=f"simplifiedtetris-binary",
    entry_point=f"gym_simplifiedtetris.envs:SimplifiedTetrisBinaryEnv",
)
