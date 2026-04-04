import yaml
from pathlib import Path
from typing import Literal, Tuple, List

import src.simulation.models as m
from src.training.models import AgentPattern


class TrainingConfig:
    def __init__(self, config_path: Path):
        with open(config_path, "r") as f:
            data = yaml.safe_load(f)
            self._data = data["training"]

    @classmethod
    def load(cls, config_path: Path):
        return cls(config_path)

    @property
    def phase(self) -> int:
        return self._data.get("phase", 0)

    @property
    def arrival_rate_history_length(self) -> int:
        return self._data.get("arrival_rate_history_length", 5)

    @property
    def norm_arrival_rate(self) -> float:
        return float(self._data.get("normalization", {}).get("arrival_rate", 10.0))

    @property
    def norm_avg_queue_length(self) -> float:
        return float(self._data.get("normalization", {}).get("avg_queue_length", 500.0))

    @property
    def norm_queue_delta(self) -> float:
        return float(self._data.get("normalization", {}).get("queue_delta", 200.0))

    @property
    def norm_p99_ttft(self) -> float:
        return float(self._data.get("normalization", {}).get("p99_ttft", 50.0))

    @property
    def norm_avg_running_requests(self) -> float:
        return float(self._data.get("normalization", {}).get("avg_running_requests", 100.0))

    @property
    def norm_current_budget(self) -> float:
        return float(self._data.get("normalization", {}).get("current_budget", 150.0))

    @property
    def norm_mig_total_ratio(self) -> float:
        return float(self._data.get("normalization", {}).get("mig_total_ratio", 7.0))

    @property
    def episode_length(self) -> int:
        return self._data["PPO"]["episode_length"]

    @property
    def rl_learning_rate(self) -> float:
        return float(self._data["PPO"]["learning_rate"])

    @property
    def rl_batch_size(self) -> int:
        return self._data["PPO"]["batch_size"]

    @property
    def rl_n_epochs(self) -> int:
        return self._data["PPO"]["n_epochs"]

    @property
    def rl_gamma(self) -> float:
        return float(self._data["PPO"]["gamma"])

    @property
    def rl_gae_lambda(self) -> float:
        return float(self._data["PPO"]["gae_lambda"])

    @property
    def rl_clip_range(self) -> float:
        return float(self._data["PPO"]["clip_range"])

    @property
    def rl_ent_coef(self) -> float:
        return float(self._data["PPO"]["ent_coef"])

    @property
    def rl_net_arch_pi(self) -> List[int]:
        return self._data["PPO"]["net_arch"]["pi"]

    @property
    def rl_net_arch_vf(self) -> List[int]:
        return self._data["PPO"]["net_arch"]["vf"]

    def pattern_duration(self, pattern: AgentPattern) -> Tuple[float, float]:
        match pattern:
            case AgentPattern.BUSY:
                cfg = self._data["patterns"]["busy"]["duration"]
                return float(cfg[0]), float(cfg[1])
            case AgentPattern.IDLE:
                cfg = self._data["patterns"]["idle"]["duration"]
                return float(cfg[0]), float(cfg[1])
            case AgentPattern.BALANCED:
                cfg = self._data["patterns"]["balanced"]["duration"]
                return float(cfg[0]), float(cfg[1])

    def pattern_rate(
        self, pattern: AgentPattern, agent_id: m.AgentId
    ) -> Tuple[float, float]:
        match pattern:
            case AgentPattern.BUSY:
                cfg = self._data["patterns"]["busy"]["rate"][agent_id.value]
                return float(cfg[0]), float(cfg[1])
            case AgentPattern.IDLE:
                cfg = self._data["patterns"]["idle"]["rate"][agent_id.value]
                return float(cfg[0]), float(cfg[1])
            case AgentPattern.BALANCED:
                cfg = self._data["patterns"]["balanced"]["rate"][agent_id.value]
                return float(cfg[0]), float(cfg[1])

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

    @property
    def scaling_factor(self) -> float:
        return self._data["reward"]["scaling"]

    @property
    def clip_threshold(self) -> float:
        return self._data["reward"]["clipping"]


# Automatically load from default path
project_root = Path(".")
TRAINING_CONFIG = TrainingConfig.load(project_root / "configs" / "training_config.yaml")
