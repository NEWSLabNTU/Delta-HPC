from enum import Enum


class AgentPattern(Enum):
    BUSY = "busy"
    IDLE = "idle"
    BALANCED = "balanced"


class TrainingPhase(Enum):
    PHASE_1 = 1
    PHASE_2 = 2
