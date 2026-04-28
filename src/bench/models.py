from enum import Enum
from typing import TypedDict


class BenchMode(Enum):
    RL = "RL"
    STATIC_NO_MIG = "static_no_mig"
    STATIC_SPLIT_EXTREME = "static_split_extreme"
    BASELINE_STATIC = "static"
    BASELINE_HEURISTIC = "heuristic"


class Workload(Enum):
    IDLE = "idle"
    EVEN = "even"
    BUSY = "busy"
    HYBRID = "hybrid"


class PhaseHistoryType(TypedDict):
    pattern: str
    avg_rate: float
    duration: float
    start_time: float
