from enum import Enum
from typing import TypedDict


class BenchMode(Enum):
    RL = "RL"
    BASELINE_7G = "7g"
    BASELINE_2_2_2_1 = "2_2_2_1"
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
