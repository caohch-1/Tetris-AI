"""Initialise the agents module."""

from gym_simplifiedtetris.agents.heuristic import HeuristicAgent
from gym_simplifiedtetris.agents.q_learning import QLearningAgent
from gym_simplifiedtetris.agents.uniform import UniformAgent

__all__ = ["HeuristicAgent", "QLearningAgent", "UniformAgent"]
