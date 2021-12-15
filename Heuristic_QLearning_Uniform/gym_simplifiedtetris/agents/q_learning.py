"""
A Q-learning agent class.
"""

from typing import Optional, Sequence

import numpy as np


class QLearningAgent(object):
    """
    An agent that learns a Q-value for each of the state-action pairs it visits.

    :param grid_dims: the grid dimensions.
    :param num_pieces: the number of pieces in use.
    :param num_actions: the number of actions available in each state.
    :param alpha: the learning rate parameter.
    :param gamma: the discount rate parameter.
    :param epsilon: the exploration rate of the epsilon-greedy policy.
    """

    def __init__(
        self,
        grid_dims: Sequence[int],
        num_pieces: int,
        num_actions: int,
        alpha: Optional[float] = 0.2,
        gamma: Optional[float] = 0.99,
        epsilon: Optional[float] = 1.0,
    ):

        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma

        q_table_dims = [2 for _ in range(grid_dims[0] * grid_dims[1])]
        q_table_dims += [num_pieces] + [num_actions]
        self._q_table = np.zeros((q_table_dims), dtype="double")

        self._num_actions = num_actions

    def predict(self, obs: np.array) -> int:
        """
        Return the action whilst following an epsilon-greedy policy.

        :param obs: a NumPy array containing the observation given to the agent by the env.
        :return: an integer corresponding to the action chosen by the Q-learning agent.
        """
        # Choose an action at random with probability epsilon.
        if np.random.rand(1)[0] <= self.epsilon:
            return np.random.choice(self._num_actions)

        # Choose greedily from the set of all actions.
        return np.argmax(self._q_table[tuple(obs)])

    def learn(
        self, reward: float, obs: np.array, next_obs: np.array, action: int
    ) -> None:
        """
        Update the Q-learning agent's Q-table.

        :param reward: the reward given to the agent by the env after taking action.
        :param obs: the old observation given to the agent by the env.
        :param next_obs: the next observation given to the agent by the env having taken action.
        :param action: the action taken that generated next_obs.
        """
        current_obs_action = tuple(list(obs) + [action])
        max_q_value = np.max(self._q_table[tuple(next_obs)])

        # Update the Q-table using the stored Q-value.
        self._q_table[current_obs_action] += self.alpha * (
            reward + self.gamma * max_q_value - self._q_table[current_obs_action]
        )
