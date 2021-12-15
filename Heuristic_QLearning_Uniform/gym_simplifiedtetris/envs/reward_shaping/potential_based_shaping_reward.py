"""Contains a potential-based shaping reward class."""

from typing import Tuple

import numpy as np


class PotentialBasedShapingReward(object):
    """A potential-based shaping reward object."""

    reward_range = (-1, 5)

    def __init__(self):
        self._heuristic_range = {"min": 1000, "max": -1}

        # The old potential is 1 because there are no holes at the start of a
        # game.
        self._old_potential = 1
        self._initial_potential = self._old_potential

    def _get_reward(self) -> Tuple[float, int]:
        """
        Override superclass method and return the potential-based shaping reward.

        :return: the potential-based shaping reward and the number of lines cleared.
        """
        num_lines_cleared = self._engine._clear_rows()
        heuristic_value = self._engine._get_holes()
        self._update_range(heuristic_value)

        new_potential = np.clip(
            1
            - (heuristic_value - self._heuristic_range["min"])
            / (self._heuristic_range["max"] + 1e-9),
            0,
            1,
        )
        shaping_reward = (new_potential - self._old_potential) + num_lines_cleared
        self._old_potential = new_potential

        return shaping_reward, num_lines_cleared

    def _get_terminal_reward(self) -> float:
        """
        Override superclass method and return the terminal potential-based shaping reward.

        :return: the terminal potential-based shaping reward.
        """
        terminal_shaping_reward = -self._old_potential
        self._old_potential = self._initial_potential

        return terminal_shaping_reward

    def _update_range(self, heuristic_value: int) -> None:
        """
        Update the heuristic range.

        :param heuristic_value: the computed heuristic value.
        """
        if heuristic_value > self._heuristic_range["max"]:
            self._heuristic_range["max"] = heuristic_value

        if heuristic_value < self._heuristic_range["min"]:
            self._heuristic_range["min"] = heuristic_value
