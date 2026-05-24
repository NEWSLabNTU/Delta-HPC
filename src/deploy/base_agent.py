"""
src/deploy/base_agent.py

Abstract base class for policy agents in the deployment.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.deploy.act_controller import ActionController


class BasePolicyAgent(ABC):
    """
    Abstract parent class for policy agents (RL or Heuristic).
    """

    def __init__(self, act_ctrl: ActionController):
        self.act_ctrl = act_ctrl

    @abstractmethod
    async def run_loop(self, duration_s: float) -> None:
        """
        Run the policy control loop for the specified duration.

        Parameters
        ----------
        duration_s:
            The duration of the benchmark in seconds.
        """
        raise NotImplementedError
