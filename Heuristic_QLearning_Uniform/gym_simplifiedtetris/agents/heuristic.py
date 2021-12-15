"""
A heuristic agent class.
"""

import numpy as np


class HeuristicAgent(object):
    """
    An agent that selects actions according to a heuristic.
    """

    @staticmethod
    def predict(ratings_or_priorities: np.array) -> int:
        """
        Return that action yielding the largest heuristic score. Separate ties
        using a priority rating, which is based on the translation and
        rotation.

        :param ratings_or_priorities: either the ratings or priorities for all available actions.
        :return: the action with the largest rating; ties are separated based on the priority.
        """
        return np.argmax(ratings_or_priorities)
