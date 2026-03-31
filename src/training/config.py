import yaml
from pathlib import Path
from typing import Literal

import src.simulation.models as m


class TrainingConfig:
    def __init__(self, config_path: Path):
        with open(config_path, "r") as f:
            data = yaml.safe_load(f)
            self._data = data["training"]

    @classmethod
    def load(cls, config_path: Path):
        return cls(config_path)

    @property
    def action_interval(self) -> float:
        return self._data["action-interval"]

    @property
    def reconfig_budget(self) -> float:
        return self._data["reconfig"]["budget"]

    @property
    def refresh_period(self) -> float:
        return self._data["reconfig"]["refresh"] * 60

    def qf(self, mig: m.MIGProfile) -> float:
        return self._data["reward"]["Q"][mig.string]

    def alpha(self, agent: m.AgentId) -> float:
        return self._data["reward"]["alpha"][agent.value]

    def w(self, latency: Literal["ttft", "tpot"]) -> float:
        k = f"w_{latency}"
        return self._data["reward"][k]

    @property
    def gamma(self) -> float:
        return self._data["reward"]["gamma"]


# Automatically load from default path
project_root = Path(".")
TRAINING_CONFIG = TrainingConfig.load(project_root / "configs" / "training_config.yaml")
