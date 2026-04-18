from enum import Enum


class AgentPattern(Enum):
    BUSY = "busy"
    IDLE = "idle"
    EVEN = "even"
    BURST = "burst"


class TrainingPhase(Enum):
    PHASE_1 = 1
    PHASE_2 = 2
