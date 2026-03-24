from __future__ import annotations

from enum import Enum
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, List, Optional, Dict, Tuple, Union, TypedDict, Mapping, Literal

from sortedcontainers import SortedList

__all__ = [
    "EventType",
    "Request",
    "RunningRequests",
    "RequestState",
    "Agent",
    "AgentId",
    "MIGProfile",
    "MIGConfigType",
    "MIGProfileRule",
    "SimulationEvent",
    "LLMEngine",
    "Simulator",
    "SimulationLogger",
    "OperationPurpose",
    "EngineStatus",
    "ParamDict",
    "EngineStepPayload",
    "RequestArrivalPayload",
    "ShutdownPayload",
    "ShutdownReallocatePayload",
    "ShutdownMergePayload",
    "ShutdownSplitPayload",
    "BootPayload",
    "TransferDetails",
    "EnvironmentStateData",
    "EnvironmentState",
    "Worker",
    "ResourceManagerAction",
    "VramTransferAction",
    "MigAction",
]

type ParamDict = Dict[Literal["alpha", "beta", "sigma"], float]


@dataclass(frozen=True)
class MIGProfileValue:
    size: int
    vram: int


class MIGProfile(Enum):
    MIG_1G_10GB = MIGProfileValue(1, 10)
    MIG_2G_10GB = MIGProfileValue(2, 10)
    MIG_3G_20GB = MIGProfileValue(3, 20)
    MIG_4G_20GB = MIGProfileValue(4, 20)
    MIG_7G_40GB = MIGProfileValue(7, 40)

    @property
    def string(self) -> str:
        return f"{self.value.size}g.{self.value.vram}gb"

    @property
    def size(self) -> int:
        return self.value.size

    @property
    def vram(self) -> int:
        return self.value.vram

    @property
    def idx(self) -> int:
        return list(MIGProfile).index(self)


type MIGConfigType = Tuple[MIGProfile, ...]


class RequestState(Enum):
    PENDING = "PENDING"
    PREFILLING = "PREFILLING"
    DECODING = "DECODING"
    COMPLETED = "COMPLETED"


class EngineStatus(Enum):
    ACTIVE = "ACTIVE"
    DRAINING = "DRAINING"
    BOOTING = "BOOTING"


class AgentId(Enum):
    CODING = "CodingAgent"
    RAG = "RAGAgent"


class EventType(Enum):
    REQUEST_ARRIVAL = "REQUEST_ARRIVAL"
    RAG_SEARCH_COMPLETE = "RAG_SEARCH_COMPLETE"
    ENGINE_STEP_COMPLETE = "ENGINE_STEP_COMPLETE"
    RESOURCE_MANAGER_TRIGGER = "RESOURCE_MANAGER_TRIGGER"
    ENGINE_SHUTDOWN_COMPLETE = "ENGINE_SHUTDOWN_COMPLETE"
    ENGINE_BOOT_COMPLETE = "ENGINE_BOOT_COMPLETE"


class OperationPurpose(Enum):
    REALLOCATE = "reallocate"
    MERGE = "merge"
    SPLIT = "split"
    PLAIN = "plain"


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
        req: Request,
        eng: Optional[LLMEngine],
    ) -> None: ...

    @abstractmethod
    def log_vram_transfer(
        self,
        current_time: float,
        giver_id: AgentId,
        receiver_id: AgentId,
        amount: int,
        eids: List[str],
    ) -> None: ...

    @abstractmethod
    def log_discard_vram_transfer(
        self, current_time: float, detail: TransferDetails
    ) -> None: ...

    @abstractmethod
    def log_engine_boot_complete(
        self,
        current_time: float,
        engine_id: str,
    ) -> None: ...

    @abstractmethod
    def log_mig_merge_trigger(
        self, current_time: float, eids: List[str], gpu: int
    ) -> None: ...

    @abstractmethod
    def log_mig_split_trigger(
        self, current_time: float, engine_id: str, gpu: int
    ) -> None: ...

    @abstractmethod
    def log_mig_merge_complete(
        self, current_time: float, new_engine_id: str
    ) -> None: ...

    @abstractmethod
    def log_mig_split_complete(self, current_time: float, engine_id: str) -> None: ...

    @abstractmethod
    def log_environment_state(
        self, current_time: float, state: EnvironmentStateData
    ) -> None: ...


# --- Payloads and Events ---


# RESOURCE_MANAGER_TRIGGER
class EmptyPayload(TypedDict):
    """Payload for RESOURCE_MANAGER_TRIGGER events (no data)."""


# ENGINE_STEP_COMPLETE
class EngineStepPayload(TypedDict):
    engine_id: str
    steps_taken: int


# REQUEST_ARRIVAL or RAG_SEARCH_COMPLETE
class RequestArrivalPayload(TypedDict):
    request: Request


# ENGINE_SHUTDOWN_COMPLETE — one variant per purpose


class ShutdownReallocatePayload(TypedDict):
    """ENGINE_SHUTDOWN_COMPLETE when purpose == OperationPurpose.REALLOCATE."""

    engine_id: str
    purpose: OperationPurpose
    receiver_id: AgentId


class ShutdownMergePayload(TypedDict):
    """ENGINE_SHUTDOWN_COMPLETE when purpose == OperationPurpose.MERGE."""

    engine_id: str
    purpose: OperationPurpose
    # Identifiers of both engines participating in the merge
    merge_engine_ids: Tuple[str, ...]
    # IDs that have already drained (grows as each engine shuts down)
    drained_ids: List[str]
    new_profile: MIGProfile
    agent_id: AgentId
    gpu: int
    # If set, the merged engine boots directly on the receiver (VRAM transfer via merge)
    receiver_id: Optional[AgentId]


class ShutdownSplitPayload(TypedDict):
    """ENGINE_SHUTDOWN_COMPLETE when purpose == OperationPurpose.SPLIT."""

    engine_id: str
    purpose: OperationPurpose
    new_profiles: List[MIGProfile]
    agent_id: AgentId
    gpu: int
    # If set, it's VRAM transfer via split
    receiver_id: Optional[AgentId]
    received_profile: Optional[MIGProfile]


ShutdownPayload = Union[
    ShutdownReallocatePayload,
    ShutdownMergePayload,
    ShutdownSplitPayload,
]


# ENGINE_BOOT_COMPLETE — one variant per boot context


class BootPayload(TypedDict):
    """ENGINE_BOOT_COMPLETE with no extra context (e.g. after a merge)."""

    engine_id: str
    purpose: OperationPurpose
    # IDs of all engines spawned by this split (used to detect when all are booted)
    sibling_engine_ids: List[str] | None


type PayloadType = Union[
    EmptyPayload,
    EngineStepPayload,
    RequestArrivalPayload,
    ShutdownPayload,
    BootPayload,
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

    @property
    def dispatch_queue(self) -> List[Request]: ...

    @abstractmethod
    def add_engine(self, engine: LLMEngine) -> None: ...

    @abstractmethod
    def dispatch(
        self, request: Request, current_time: float
    ) -> Optional[LLMEngine]: ...

    @abstractmethod
    def process_waiting_queue(self, current_time: float) -> None: ...


class LLMEngine(ABC):
    @property
    @abstractmethod
    def engine_id(self) -> str: ...

    @property
    @abstractmethod
    def gpu(self) -> int: ...

    @property
    @abstractmethod
    def owner(self) -> Agent: ...

    @owner.setter
    @abstractmethod
    def owner(self, value: Agent) -> None: ...

    @property
    def current_kv_utilization(self) -> float: ...

    @property
    @abstractmethod
    def model_name(self) -> str: ...

    @property
    @abstractmethod
    def mig_profile(self) -> MIGProfile: ...

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
    def update_model(
        self,
        new_owner: Agent,
        model_name: str,
        max_batched_tokens: int,
        prefill_params: ParamDict,
        tpot_params: ParamDict,
        restart_time: float,
    ) -> None: ...

    @abstractmethod
    def add_request(self, request: Request, current_time: float) -> None: ...

    @staticmethod
    @abstractmethod
    def create(
        gpu: int,
        engine_id: str,
        owner: Agent,
        mig_profile: MIGProfile,
        current_time: float,
    ) -> LLMEngine: ...

    @abstractmethod
    def trigger_shutdown(
        self, payload: ShutdownPayload, current_time: float
    ) -> Optional[SimulationEvent]: ...

    @abstractmethod
    def trigger_boot(self, payload: BootPayload) -> SimulationEvent: ...

    @abstractmethod
    def activate(self, current_time: float) -> None: ...

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
    def events(self) -> SortedList[SimulationEvent]: ...

    @property
    @abstractmethod
    def current_time(self) -> float: ...

    @property
    @abstractmethod
    def logger(self) -> SimulationLogger: ...

    @abstractmethod
    def add_arrival_events(self, requests: List[Request]) -> None: ...

    @abstractmethod
    def run(self) -> bool: ...


@dataclass
class VramTransferAction:
    giver: AgentId
    receiver: AgentId
    amount: int


@dataclass
class MigAction:
    action: str  # "split" or "merge"
    victim: AgentId


class ResourceManagerAction(Enum):
    NO_ACTION = None

    # 1-4: VRAM
    TRANSFER_10_CODING_RAG = VramTransferAction(
        giver=AgentId.CODING, receiver=AgentId.RAG, amount=10
    )
    TRANSFER_10_RAG_CODING = VramTransferAction(
        giver=AgentId.RAG, receiver=AgentId.CODING, amount=10
    )
    TRANSFER_20_CODING_RAG = VramTransferAction(
        giver=AgentId.CODING, receiver=AgentId.RAG, amount=20
    )
    TRANSFER_20_RAG_CODING = VramTransferAction(
        giver=AgentId.RAG, receiver=AgentId.CODING, amount=20
    )

    # 5-8: MIG
    SPLIT_CODING = MigAction(action="split", victim=AgentId.CODING)
    SPLIT_RAG = MigAction(action="split", victim=AgentId.RAG)
    MERGE_CODING = MigAction(action="merge", victim=AgentId.CODING)
    MERGE_RAG = MigAction(action="merge", victim=AgentId.RAG)


# --- Management Interfaces ---


@dataclass
class TransferDetails:
    amount: int
    giver_id: AgentId
    receiver_id: AgentId


class EnvironmentStateData(TypedDict):
    arrival_rate: Dict[AgentId, float]
    arrival_trend: Dict[AgentId, float]
    avg_queue_length: Dict[AgentId, float]
    queue_delta: Dict[AgentId, int]
    p99_ttft: Dict[AgentId, float]
    avg_tpot: Dict[AgentId, float]
    kv_cache_utilization: Dict[int, List[float]]
    mig_config_encoding: Dict[int, List[int]]
    recovery_flag: bool
    requests: List[Request]
    avg_running_requests: Dict[AgentId, float]


class Worker(ABC):
    @abstractmethod
    def transfer(
        self,
        current_time: float,
        details: TransferDetails,
        agents: Dict[AgentId, Agent],
    ) -> Tuple[str, Any] | None: ...


class EnvironmentState(ABC):
    @property
    @abstractmethod
    def action_interval(self) -> float: ...

    @abstractmethod
    def reset_for_next_interval(
        self, current_time: float, agents: Dict[AgentId, Agent]
    ) -> None: ...

    @abstractmethod
    def record_queue_length_advance(
        self, current_time: float, agents: Dict[AgentId, Agent]
    ) -> None: ...

    @abstractmethod
    def register_arrival(
        self, agent_id: AgentId, time: float, request: Request
    ) -> None: ...

    @abstractmethod
    def register_reconfig(self) -> None: ...

    @abstractmethod
    def get_state(
        self,
        current_time: float,
        agents: Dict[AgentId, Agent],
        engines: Dict[str, LLMEngine],
    ) -> EnvironmentStateData: ...


class MIGProfileRule(ABC):
    @abstractmethod
    def get_possible_merges(
        self, agent: Agent
    ) -> List[Tuple[List[LLMEngine], MIGProfile]]: ...

    @abstractmethod
    def get_possible_splits(
        self, agent: Agent
    ) -> List[Tuple[LLMEngine, List[MIGProfile]]]: ...
