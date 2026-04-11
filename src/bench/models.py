from enum import Enum


class BenchMode(Enum):
    RL = "RL"
    BASELINE_7G = "7g"
    BASELINE_2_2_2_1 = "2_2_2_1"


class Workload(Enum):
    IDLE = "idle"
    BALANCED = "balanced"
    BUSY = "busy"
    HYBRID = "hybrid"
