import yaml
from pathlib import Path
from typing import Tuple, Union

from src.bench.models import Workload
from src.training.models import TrainingPhase
import src.simulation.models as m


class BenchConfig:
    def __init__(self, config_path: Path):
        with open(config_path, "r") as f:
            data = yaml.safe_load(f)
            self._workloads = data["workloads"]
            self._length = data["benchmark-length"]
            self._phase = data.get("phase", 2)
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
            from src.simulation.config import GPU_MIG_PROFILE

            hw_prof = next(
                p for p in GPU_MIG_PROFILE[gpu_id] if p.profile_type == mig_profile
            )
        else:
            hw_prof = mig_profile

        gpu_model = hw_prof.gpu_model
        mig_str = hw_prof.string

        # Structure: service_rates[gpu_model][agent_id][mig_str]
        model_rates = rates.get(gpu_model, {})
        agent_rates = model_rates.get(agent_id.value, {})

        return float(agent_rates.get(mig_str, 0.0))

    def get_rate_range(self, workload: Workload) -> Tuple[float, float]:
        cfg = self._workloads[workload.value]["rate"]
        return float(cfg[0]), float(cfg[1])

    def get_duration_range(self, workload: Workload) -> Tuple[float, float]:
        cfg = self._workloads[workload.value]["duration"]
        return float(cfg[0]), float(cfg[1])

    @property
    def benchmark_length(self) -> int:
        return int(self._length)

    @property
    def phase(self) -> TrainingPhase:
        return TrainingPhase(self._phase)

    @property
    def seed(self) -> int:
        return int(self._seed)


# Automatically load from default path
project_root = Path(".")
BENCH_CONFIG = BenchConfig(project_root / "configs" / "bench_config.yaml")
