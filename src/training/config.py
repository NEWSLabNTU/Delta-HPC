import yaml
from pathlib import Path
from typing import Any, Dict, Literal, Tuple, List

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
    def sb3_norm(self) -> bool:
        return self._data.get("sb3_norm", False)

    @property
    def queue_length_trend_clamp(self) -> float:
        return float(self._data["normalization"]["queue_length_trend_clamp"])

    @property
    def arrival_rate_history_length(self) -> int:
        return self._data["arrival_rate_history_length"]

    @property
    def norm_avg_queue_length(self) -> float:
        return float(self._data["normalization"]["avg_queue_length"])

    @property
    def default_waiting_qj(self) -> float:
        return float(self._data["normalization"]["default_waiting_qj"])

    @property
    def norm_avg_running_requests(self) -> float:
        return float(self._data["normalization"]["avg_running_requests"])

    @property
    def norm_arrival_rate(self) -> float:
        return float(self._data["normalization"]["arrival_rate"])

    @property
    def norm_avg_composite_latency(self) -> float:
        return float(self._data["normalization"]["avg_composite_latency"])

    @property
    def norm_current_budget(self) -> float:
        return float(self._data["normalization"]["current_budget"])

    @property
    def norm_total_sm_ratio(self) -> float:
        return float(self._data["normalization"]["total_sm_ratio"])

    @property
    def norm_total_vram_ratio(self) -> float:
        return float(self._data["normalization"]["total_vram_ratio"])

    @property
    def norm_mig_geometry(self) -> float:
        return float(self._data["normalization"]["mig_geometry"])

    @property
    def norm_vram_transfer_amount(self) -> float:
        return float(self._data["normalization"]["vram_transfer_amount"])

    @property
    def _ppo_cfg(self) -> Dict[str, Any]:
        return self._data["PPO"]

    @property
    def episode_length(self) -> int:
        return self._ppo_cfg["episode_length"]

    @property
    def rl_n_steps(self) -> int:
        return self._ppo_cfg["n_steps"]

    @property
    def rl_batch_size(self) -> int:
        return self._ppo_cfg["batch_size"]

    @property
    def rl_n_epochs(self) -> int:
        return self._ppo_cfg["n_epochs"]

    @property
    def rl_lr_max(self) -> float:
        return float(self._ppo_cfg["lr_max"])

    @property
    def rl_lr_min(self) -> float:
        return float(self._ppo_cfg["lr_min"])

    @property
    def rl_gamma(self) -> float:
        return float(self._ppo_cfg["gamma"])

    @property
    def rl_gae_lambda(self) -> float:
        return float(self._ppo_cfg["gae_lambda"])

    @property
    def rl_clip_range(self) -> float:
        return float(self._ppo_cfg["clip_range"])

    @property
    def rl_enable_ent_coef_schd(self) -> bool:
        return bool(self._ppo_cfg["ent_coef_schd"])

    @property
    def rl_ent_coef(self) -> float:
        return float(self._ppo_cfg["ent_coef"])

    @property
    def rl_min_ent_coef(self) -> float:
        return float(self._ppo_cfg["min_ent_coef"])

    @property
    def rl_max_ent_coef(self) -> float:
        return float(self._ppo_cfg["max_ent_coef"])

    @property
    def rl_net_arch_pi(self) -> List[int]:
        return self._ppo_cfg["net_arch"]["pi"]

    @property
    def rl_net_arch_vf(self) -> List[int]:
        return self._ppo_cfg["net_arch"]["vf"]

    @property
    def action_cooldown(self) -> int:
        return int(self._data["action_mask"]["action_cooldown"])

    def pattern_duration(self, pattern: AgentPattern) -> Tuple[float, float]:
        match pattern:
            case AgentPattern.BUSY:
                cfg = self._data["patterns"]["busy"]["duration"]
                return float(cfg[0]), float(cfg[1])
            case AgentPattern.IDLE:
                cfg = self._data["patterns"]["idle"]["duration"]
                return float(cfg[0]), float(cfg[1])
            case AgentPattern.EVEN:
                cfg = self._data["patterns"]["even"]["duration"]
                return float(cfg[0]), float(cfg[1])
            case AgentPattern.BURST:
                cfg = self._data["patterns"]["burst"]["duration"]
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
            case AgentPattern.EVEN:
                cfg = self._data["patterns"]["even"]["rate"][agent_id.value]
                return float(cfg[0]), float(cfg[1])
            case AgentPattern.BURST:
                cfg = self._data["patterns"]["burst"]["rate"][agent_id.value]
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

    def qf_concrete(self, mig_obj: m.MIGProfileBase, agent_id: m.AgentId) -> float:
        """Retrieve the Quality Factor (Q) for a concrete MIG profile."""
        key = mig_obj.string
        try:
            return self._data["reward"]["Q"][agent_id.value][key]
        except KeyError:
            raise ValueError(
                f"Missing Q-value for profile {key} (GPU: {mig_obj.gpu_model}) and agent {agent_id.value}. "
                f"Please define it in configs/training_config.yaml"
            )

    def qf(self, mig_obj: m.MIGProfileBase, agent_id: m.AgentId) -> float:
        """Alias for qf_concrete."""
        return self.qf_concrete(mig_obj, agent_id)

    def alpha(self, agent: m.AgentId) -> float:
        return self._data["reward"]["alpha"][agent.value]

    def w(self, latency: Literal["ttft", "tpot"]) -> float:
        k = f"w_{latency}"
        return self._data["reward"][k]

    @property
    def gamma(self) -> float:
        return self._data["reward"]["gamma"]

    @property
    def scaling(self) -> float:
        return self._data["reward"]["scaling"]

    @property
    def clip_threshold(self) -> float:
        return self._data["reward"]["clipping"]

    @property
    def use_quality_bonus(self) -> bool:
        return self._data["reward"].get("use_quality_bonus", False)

    @property
    def use_affinity_bonus(self) -> bool:
        return self._data["reward"].get("use_affinity_bonus", False)

    @property
    def affinity_bonus_weight(self) -> float:
        return float(self._data["reward"].get("affinity_bonus_weight", 1.0))

    @property
    def total_timesteps(self) -> int:
        return self._data["total_timesteps"]


config_path_env = "configs/training_config.yaml"
TRAINING_CONFIG = TrainingConfig.load(Path(config_path_env))
