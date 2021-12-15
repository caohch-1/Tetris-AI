"""Contains a simplified Tetris env class with a part-binary obs space."""

import numpy as np
from gym import spaces

from gym_simplifiedtetris.register import register
from gym_simplifiedtetris.envs.simplified_tetris_base_env import SimplifiedTetrisBaseEnv


class SimplifiedTetrisPartBinaryEnv(SimplifiedTetrisBaseEnv):
    """
    A simplified Tetris environment, where the observation space is a
    flattened NumPy array containing the grid's binary representation
    excluding the top piece_size rows, plus the current piece's id.

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
            low=np.append(
                np.zeros(self._width_ * (self._height_ - self._piece_size_)), 0
            ),
            high=np.append(
                np.ones(self._width_ * (self._height_ - self._piece_size_)),
                self._num_pieces_ - 1,
            ),
            dtype=np.int,
        )

    def _get_obs(self) -> np.array:
        """
        Override superclass method and return a flattened NumPy array
        containing the grid's binary representation excluding the top
        piece_size rows, plus the current piece's id.

        :return: the current observation.
        """
        current_grid = self._engine._grid[:, self._piece_size_ :].flatten()
        return np.append(current_grid, self._engine._piece._idx)


register(
    incomplete_id="simplifiedtetris-partbinary",
    entry_point="gym_simplifiedtetris.envs:SimplifiedTetrisPartBinaryEnv",
)
