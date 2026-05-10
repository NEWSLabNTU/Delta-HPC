from enum import Enum

__all__ = [
    "AgentPattern",
]


class AgentPattern(Enum):
    BUSY = "busy"
    IDLE = "idle"
    EVEN = "even"
    BURST = "burst"
