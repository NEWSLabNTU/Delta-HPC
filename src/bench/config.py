import yaml
from pathlib import Path
from typing import Tuple

from src.bench.models import Workload
from src.training.models import TrainingPhase


class BenchConfig:
    def __init__(self, config_path: Path):
        with open(config_path, "r") as f:
            data = yaml.safe_load(f)
            self._workloads = data["workloads"]
            self._length = data["benchmark-length"]
            self._phase = data.get("phase", 2)
            self._seed = data.get("seed", 42)
            self._heuristic = data.get("heuristic", {"watermark_high": 20.0, "watermark_low": 5.0})

    @property
    def q_threshold_high(self) -> float:
        return float(self._heuristic["q_threshold_high"])

    @property
    def q_threshold_low(self) -> float:
        return float(self._heuristic["q_threshold_low"])

    @property
    def busy_threshold(self) -> float:
        return float(self._heuristic["busy_threshold"])

    @property
    def idle_threshold(self) -> float:
        return float(self._heuristic["idle_threshold"])

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
