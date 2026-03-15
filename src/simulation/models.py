import os
import json
import yaml
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum


class RequestState(Enum):
    PENDING = "PENDING"
    PREFILLING = "PREFILLING"
    DECODING = "DECODING"
    COMPLETED = "COMPLETED"


class EngineStatus(Enum):
    ACTIVE = "ACTIVE"
    DRAINING = "DRAINING"
    RESTARTING = "RESTARTING"


class AgentId(Enum):
    CODING = "CodingAgent"
    RAG = "RAGAgent"


class EventType(Enum):
    REQUEST_ARRIVAL = "REQUEST_ARRIVAL"
    ENGINE_STEP_COMPLETE = "ENGINE_STEP_COMPLETE"
    REALLOCATION_TRIGGER = "REALLOCATION_TRIGGER"
    ENGINE_RESTART_COMPLETE = "ENGINE_RESTART_COMPLETE"


@dataclass
class Request:
    id: str
    agent_id: AgentId
    prompt_tokens: int = 0
    completion_tokens: int = 0
    arrival_time: float = 0.0

    # Simulation state
    state: RequestState = RequestState.PENDING
    prefilled_tokens: int = 0
    generated_tokens: int = 0
    start_time: Optional[float] = None
    first_token_time: Optional[float] = None
    finish_time: Optional[float] = None

    @property
    def is_finished(self) -> bool:
        return self.generated_tokens >= self.completion_tokens

    @property
    def remaining_prefill_tokens(self) -> int:
        return self.prompt_tokens - self.prefilled_tokens

    @property
    def prefill_completed(self) -> bool:
        return self.prefilled_tokens >= self.prompt_tokens


@dataclass(order=True)
class SimulationEvent:
    time: float
    event_type: EventType = field(compare=False)
    payload: dict = field(compare=False, repr=False)


@dataclass
class RunningRequests:
    prefill_requests: List[Request] = field(default_factory=list)
    decoding_requests: List[Request] = field(default_factory=list)

    def __len__(self):
        return len(self.decoding_requests) + len(self.prefill_requests)

    @property
    def all_requests(self) -> List[Request]:
        return self.decoding_requests + self.prefill_requests
