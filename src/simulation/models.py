from __future__ import annotations

from enum import Enum
from dataclasses import dataclass, field
from typing import (
    List,
    Optional,
    Literal,
    Dict,
    Union,
    TypedDict,
    Mapping,
)
from abc import ABC, abstractmethod

type ParamDict = Dict[Literal["alpha", "beta", "sigma"], float]


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


# --- Interfaces ---


class Request(ABC):
    @property
    @abstractmethod
    def id(self) -> str: ...

    @property
    @abstractmethod
    def agent_id(self) -> AgentId: ...

    @property
    @abstractmethod
    def prompt_tokens(self) -> int: ...

    @property
    @abstractmethod
    def completion_tokens(self) -> int: ...

    @completion_tokens.setter
    @abstractmethod
    def completion_tokens(self, value: int) -> None: ...

    @property
    @abstractmethod
    def arrival_time(self) -> float: ...

    @arrival_time.setter
    @abstractmethod
    def arrival_time(self, value: float) -> None: ...

    @property
    @abstractmethod
    def original_id(self) -> str: ...

    @property
    @abstractmethod
    def decode_time(self) -> float: ...

    @decode_time.setter
    @abstractmethod
    def decode_time(self, value: float) -> None: ...

    @property
    @abstractmethod
    def state(self) -> RequestState: ...

    @state.setter
    @abstractmethod
    def state(self, value: RequestState) -> None: ...

    @property
    @abstractmethod
    def prefilled_tokens(self) -> int: ...

    @prefilled_tokens.setter
    @abstractmethod
    def prefilled_tokens(self, value: int) -> None: ...

    @property
    @abstractmethod
    def generated_tokens(self) -> int: ...

    @generated_tokens.setter
    @abstractmethod
    def generated_tokens(self, value: int) -> None: ...

    @property
    @abstractmethod
    def start_time(self) -> Optional[float]: ...

    @start_time.setter
    @abstractmethod
    def start_time(self, value: Optional[float]) -> None: ...

    @property
    @abstractmethod
    def first_token_time(self) -> Optional[float]: ...

    @first_token_time.setter
    @abstractmethod
    def first_token_time(self, value: Optional[float]) -> None: ...

    @property
    @abstractmethod
    def finish_time(self) -> Optional[float]: ...

    @finish_time.setter
    @abstractmethod
    def finish_time(self, value: Optional[float]) -> None: ...

    @property
    @abstractmethod
    def is_finished(self) -> bool: ...

    @property
    @abstractmethod
    def remaining_prefill_tokens(self) -> int: ...

    @property
    @abstractmethod
    def prefill_completed(self) -> bool: ...


class RunningRequests(ABC):
    @property
    @abstractmethod
    def prefill_requests(self) -> List[Request]: ...

    @prefill_requests.setter
    @abstractmethod
    def prefill_requests(self, value: List[Request]) -> None: ...

    @property
    @abstractmethod
    def decoding_requests(self) -> List[Request]: ...

    @abstractmethod
    def __len__(self) -> int: ...

    @property
    @abstractmethod
    def all_requests(self) -> List[Request]: ...


class SimulationLogger(ABC):
    @abstractmethod
    def log(self, message: str) -> None: ...

    @abstractmethod
    def flush(self) -> None: ...

    @abstractmethod
    def log_engine_step(
        self,
        current_time: float,
        agents: Dict[AgentId, Agent],
        stepping_engine: LLMEngine,
        next_arrival_time: Optional[float],
    ) -> None: ...

    @abstractmethod
    def log_request_arrival(
        self,
        current_time: float,
        req_id: str,
        target_agent: AgentId,
        assigned_engine: Optional[LLMEngine],
    ) -> None: ...

    @abstractmethod
    def log_reallocation(
        self,
        current_time: float,
        giver_id: AgentId,
        receiver_id: AgentId,
        mig_profile: str,
    ) -> None: ...

    @abstractmethod
    def log_engine_restart_complete(
        self,
        current_time: float,
        engine_id: str,
        giver_id: AgentId,
        receiver_id: AgentId,
    ) -> None: ...


# --- Payloads and Events ---


class EmptyPayload(TypedDict):
    """Payload for REALLOCATION_TRIGGER events (no data)."""


class EnginePayload(TypedDict):
    """Payload for ENGINE_RESTART_COMPLETE events."""

    engine_id: str


class EngineStepPayload(TypedDict):
    """Payload for ENGINE_STEP_COMPLETE events."""

    engine_id: str
    steps_taken: int


class RequestArrivalPayload(TypedDict):
    """Payload for REQUEST_ARRIVAL events."""

    request: Request
    target_agent: AgentId


type PayloadType = Union[
    EmptyPayload,
    EnginePayload,
    EngineStepPayload,
    RequestArrivalPayload,
]


@dataclass(order=True)
class SimulationEvent:
    time: float
    event_type: EventType = field(compare=False)
    payload: PayloadType = field(compare=False, repr=False)


# --- Core Interfaces ---


class Agent(ABC):
    @property
    @abstractmethod
    def agent_id(self) -> AgentId: ...

    @property
    @abstractmethod
    def engines(self) -> List[LLMEngine]: ...

    @property
    @abstractmethod
    def completed_requests(self) -> List[Request]: ...

    @abstractmethod
    def add_engine(self, engine: LLMEngine) -> None: ...

    @abstractmethod
    def dispatch(self, request: Request, current_time: float) -> LLMEngine: ...


class LLMEngine(ABC):
    @property
    @abstractmethod
    def engine_id(self) -> str: ...

    @property
    @abstractmethod
    def owner(self) -> Agent: ...

    @owner.setter
    @abstractmethod
    def owner(self, value: Agent) -> None: ...

    @property
    @abstractmethod
    def model_name(self) -> str: ...

    @property
    @abstractmethod
    def mig_profile(self) -> str: ...

    @property
    @abstractmethod
    def current_time(self) -> float: ...

    @property
    @abstractmethod
    def status(self) -> EngineStatus: ...

    @property
    @abstractmethod
    def waiting_queue(self) -> List[Request]: ...

    @property
    @abstractmethod
    def running_queue(self) -> RunningRequests: ...

    @abstractmethod
    def get_tpot(self, concurrent_requests: int) -> float: ...

    @abstractmethod
    def get_prefill_time(self, num_tokens: int) -> float: ...

    @abstractmethod
    def update_model(
        self,
        model_name: str,
        max_batched_tokens: int,
        prefill_params: ParamDict,
        tpot_params: ParamDict,
        restart_time: float,
    ) -> None: ...

    @abstractmethod
    def add_request(self, request: Request, current_time: float) -> None: ...

    @abstractmethod
    def trigger_reallocation(
        self, current_time: float
    ) -> Optional[SimulationEvent]: ...

    @abstractmethod
    def finish_restart(self, current_time: float) -> None: ...

    @abstractmethod
    def step(
        self, current_time: float, next_arrival_time: Optional[float] = None
    ) -> Optional[SimulationEvent]: ...


class Simulator(ABC):
    @property
    @abstractmethod
    def agents(self) -> Mapping[AgentId, Agent]: ...

    @property
    @abstractmethod
    def engines(self) -> Dict[str, LLMEngine]: ...

    @property
    @abstractmethod
    def events(self) -> List[SimulationEvent]: ...

    @property
    @abstractmethod
    def current_time(self) -> float: ...

    @property
    @abstractmethod
    def logger(self) -> SimulationLogger: ...

    @abstractmethod
    def add_arrival_events(self, requests: List[Request]) -> None: ...

    @abstractmethod
    def run(self) -> None: ...
