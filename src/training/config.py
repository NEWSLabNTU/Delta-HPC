import yaml
from pathlib import Path


class TrainingConfig:
    def __init__(self, config_path: Path):
        with open(config_path, "r") as f:
            self._data = yaml.safe_load(f)

    @classmethod
    def load(cls, config_path: Path):
        return cls(config_path)

    @property
    def action_interval(self) -> float:
        return self._data["training"]["action-interval"]

    @property
    def reconfig_budget(self) -> float:
        return self._data["training"]["reconfig"]["budget"]

    @property
    def refresh_period(self) -> float:
        return self._data["training"]["reconfig"]["refresh"] * 60


# Automatically load from default path
project_root = Path(".")
TRAINING_CONFIG = TrainingConfig.load(project_root / "configs" / "training_config.yaml")
