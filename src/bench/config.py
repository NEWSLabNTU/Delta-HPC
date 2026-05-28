import yaml
from pathlib import Path
from typing import Tuple, Union

from src.bench.models import Workload
import src.share.models as m
from src.simulation.config import GPU_MIG_PROFILE


class BenchConfig:
    def __init__(self, config_path: Path):
        with open(config_path, "r") as f:
            data = yaml.safe_load(f)
            self._workloads = data["workloads"]
            self._length = data["benchmark-length"]
            self._seed = data.get("seed", 42)
            self._heuristic = data.get(
                "heuristic", {"watermark_high": 20.0, "watermark_low": 5.0}
            )

    @property
    def utilization_factor(self) -> float:
        return float(self._heuristic.get("utilization_factor", 0.8))

    @property
    def high_threshold(self) -> float:
        return float(self._heuristic.get("high_threshold", 1.2))

    @property
    def low_threshold(self) -> float:
        return float(self._heuristic.get("low_threshold", 0.8))

    def get_service_rate(
        self,
        agent_id: m.AgentId,
        mig_profile: Union[m.MIGProfile, m.MIGProfileBase],
        gpu_id: int = 0,
    ) -> float:
        rates = self._heuristic.get("service_rates", {})

        if isinstance(mig_profile, m.MIGProfile):
            # Resolve logical profile to concrete hardware profile for the given gpu_id
            hw_prof = next(
                p for p in GPU_MIG_PROFILE[gpu_id] if p.profile_type == mig_profile
            )
        else:
            hw_prof = mig_profile

        gpu_model = hw_prof.gpu_model

        # Structure: service_rates[gpu_model][agent_id][mig_str]
        model_rates = rates.get(gpu_model, {})
        agent_rates = model_rates.get(agent_id.value, {})

        prof_str = hw_prof.string
        original_rate = float(agent_rates.get(prof_str, 0.0))

        match hw_prof.profile_type:
            case m.MIGProfile.MIG_7G:
                factor = 1.0
            case m.MIGProfile.MIG_4G | m.MIGProfile.MIG_3G:
                factor = 0.8
            case (
                m.MIGProfile.MIG_2G
                | m.MIGProfile.MIG_1G_LARGE
                | m.MIGProfile.MIG_1G_SMALL
            ):
                factor = 0.5
            case _:
                raise ValueError(f"Unknown MIG profile type: {hw_prof.profile_type}")

        return original_rate * factor

    def get_rate_range(
        self, workload: Workload, agent_id: m.AgentId
    ) -> Tuple[float, float]:
        cfg = self._workloads[workload.value]["rate"]
        if isinstance(cfg, dict):
            cfg = cfg[agent_id.value]
        return float(cfg[0]), float(cfg[1])

    def get_duration_range(self, workload: Workload) -> Tuple[float, float]:
        cfg = self._workloads[workload.value]["duration"]
        return float(cfg[0]), float(cfg[1])

    @property
    def benchmark_length(self) -> int:
        return int(self._length)

    @property
    def seed(self) -> int:
        return int(self._seed)


# Automatically load from default path
project_root = Path(".")
BENCH_CONFIG = BenchConfig(project_root / "configs" / "bench_config.yaml")
