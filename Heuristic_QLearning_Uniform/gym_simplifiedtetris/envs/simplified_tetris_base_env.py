from abc import abstractmethod
from typing import Any, Dict, Optional, Sequence, Tuple

import gym
import numpy as np
from gym import spaces
from gym.utils import seeding

from gym_simplifiedtetris.envs.simplified_tetris_engine import SimplifiedTetrisEngine


class SimplifiedTetrisBaseEnv(gym.Env):
    """
    All custom envs inherit from gym.Env and implement the essential methods
    and spaces.

    :param grid_dims: the grid dimensions.
    :param piece_size: the size of every piece.
    :param seed: the rng seed.
    """

    metadata = {"render.modes": ["human", "rgb_array"]}
    reward_range = (0, 4)

    @property
    def action_space(self) -> spaces.Discrete:
        # Set the discrete action space.
        return spaces.Discrete(self._num_actions_)

    @property
    @abstractmethod
    def observation_space(self):
        raise NotImplementedError()

    def __init__(
        self, grid_dims: Sequence[int], piece_size: int, seed: Optional[int] = 8191
    ) -> None:

        if not isinstance(grid_dims, (list, tuple, np.array)) or len(grid_dims) != 2:
            raise TypeError(
                "Inappropriate format provided for grid_dims. It should be a list, tuple or numpy array of length 2 containing integers."
            )

        assert piece_size in [
            1,
            2,
            3,
            4,
        ], "piece_size should be either 1, 2, 3, or 4."

        assert list(grid_dims) in [
            [20, 10],
            [10, 10],
            [8, 6],
            [7, 4],
        ], f"Grid dimensions must be one of (20, 10), (10, 10), (8, 6), or (7, 4)."

        self._height_, self._width_ = grid_dims
        self._piece_size_ = piece_size

        self._num_actions_, self._num_pieces_ = {
            1: (grid_dims[1], 1),
            2: (2 * grid_dims[1] - 1, 1),
            3: (4 * grid_dims[1] - 4, 2),
            4: (4 * grid_dims[1] - 6, 7),
        }[piece_size]

        self._seed(seed=seed)

        self._engine = SimplifiedTetrisEngine(
            grid_dims=grid_dims,
            piece_size=piece_size,
            num_pieces=self._num_pieces_,
            num_actions=self._num_actions_,
        )

    def __str__(self) -> str:
        return np.array(self._engine._grid.T, dtype=int).__str__()

    def __repr__(self) -> str:
        return f"""{self.__class__.__name__}(({self._height_!r}, {self.
        _width_!r}), {self._piece_size_!r})"""

    def reset(self) -> np.array:
        """
        Reset the env.

        :return: the current obs.
        """
        self._engine._reset()

        return self._get_obs()

    def step(self, action: int) -> Tuple[np.array, float, bool, Dict[str, Any]]:
        """
        Hard drop the current piece according to the action. Terminate the
        game if the piece cannot fit into the bottom 'height-piece_size' rows.
        Otherwise, select a new piece and reset the anchor.

        :param action: the action to be taken.
        :return: the next observation, reward, game termination indicator, and env info.
        """
        info = {}

        translation, rotation = self._engine._get_translation_rotation(action)

        self._engine._rotate_piece(rotation)
        self._engine._anchor = [translation, self._piece_size_ - 1]
        info["anchor"] = (translation, rotation)

        self._engine._hard_drop()
        self._engine._update_grid(True)

        # The game terminates when any of the dropped piece's blocks occupies
        # any of the top 'piece_size' rows, before any full rows are cleared.
        if np.any(self._engine._grid[:, : self._piece_size_]):

            info["num_rows_cleared"] = 0

            self._engine._final_scores = np.append(
                self._engine._final_scores, self._engine._score
            )

            return self._get_obs(), self._get_terminal_reward(), True, info

        reward, num_rows_cleared = self._get_reward()
        self._engine._score += num_rows_cleared

        self._engine._update_coords_and_anchor()

        info["num_rows_cleared"] = num_rows_cleared

        return self._get_obs(), reward, False, info

    def render(self, mode: Optional[str] = "human") -> np.ndarray:
        """
        Render the env.

        :param mode: the render mode.
        :return: the image pixel values.
        """
        return self._engine._render(mode)

    def close(self) -> None:
        """Close the open windows."""
        return self._engine._close()

    def _seed(self, seed: Optional[int] = 8191) -> None:
        """
        Seed the env.

        :param seed: an optional seed to seed the rng with.
        """
        self._np_random, _ = seeding.np_random(seed)

    def _get_reward(self) -> Tuple[float, int]:
        """
        Return the reward.

        :return: the reward and the number of lines cleared.
        """
        return self._engine._get_reward()

    @staticmethod
    def _get_terminal_reward() -> float:
        """
        Return the terminal reward.

        :return: the terminal reward.
        """
        return 0.0

    @abstractmethod
    def _get_obs(self):
        raise NotImplementedError()
