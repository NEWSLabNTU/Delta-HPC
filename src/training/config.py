import yaml
from pathlib import Path


class TrainingConfig:
    def __init__(self, config_path: Path):
        with open(config_path, "r") as f:
            data = yaml.safe_load(f)
            self.reconfig_budget = data["training"]["reconfig-budget"]
            self.refresh_period = (
                data["training"]["refresh"] * 60
            )  # Convert minutes to seconds

    @classmethod
    def load(cls, config_path: Path):
        return cls(config_path)


# Automatically load from default path
project_root = Path(".")
TRAINING_CONFIG = TrainingConfig.load(project_root / "configs" / "training_config.yaml")
