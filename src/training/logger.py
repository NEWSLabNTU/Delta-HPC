from pathlib import Path
from datetime import datetime
from typing import Dict

import src.simulation.models as m


class TrainingLogger:
    """
    An efficient, buffered logger for RL training steps.
    Reduces I/O overhead by keeping the file handle open and
    supporting conditional logging (sampling).
    """

    def __init__(
        self, log_dir: str = "logs", enabled: bool = True, log_frequency: int = 1
    ):
        self.enabled = enabled
        self.log_frequency = log_frequency  # Log every N steps
        self.log_dir = (
            Path(log_dir) / "train" / datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        self.file = None
        self._ensure_dir()

    def _ensure_dir(self):
        if self.enabled:
            self.log_dir.mkdir(parents=True, exist_ok=True)

    def start_episode(self, episode_idx: int):
        """Opens a new log file for a specific episode."""
        if not self.enabled:
            return

        self.close()  # Ensure previous is closed
        filename = f"episode_{episode_idx}.log"
        filepath = self.log_dir / filename
        self.file = open(filepath, "w", buffering=1)  # Line-buffered

    def log_step(
        self,
        step: int,
        action_name: str,
        budget: float,
        rates: Dict[m.AgentId, float],
        pattern: int,
        agents: Dict[m.AgentId, m.Agent],
    ):
        if not self.enabled or self.file is None:
            return

        if step % self.log_frequency != 0:
            return

        # Use a list to accumulate strings (more efficient than += for large strings)
        rts = {a.name: r for a, r in rates.items()}
        lines = [
            f"--- Step {step} ---\n",
            f"Action: {action_name}\n",
            f"Budget: {budget}\n",
            f"Rates: {rts} (Pattern: {pattern})\n",
            "Simulation State (After-Action):\n",
        ]

        # Build the agent/engine mapping
        for aid, ag in agents.items():
            agent_id = getattr(aid, "value", aid)
            lines.append(f"Agent {agent_id}:\n")

            for eng in ag.engines:
                lines.append(f"  {eng.engine_id}\n")
            lines.append("\n")

        lines.append("\n")

        # Perform a single I/O write operation
        self.file.write("".join(lines))

    def close(self):
        """Safely closes the current log file."""
        if self.file:
            self.file.flush()
            self.file.close()
            self.file = None
