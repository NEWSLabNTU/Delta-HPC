from __future__ import annotations

from enum import Enum
from typing import (
    Dict,
    List,
    Optional,
    Union,
    Tuple,
    TypedDict,
    Literal,
)
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

# Simulation-specific types only. Shared types are imported from src.share.models directly.
import src.share.models as m

__all__ = [
    "RequestState",
    "RunningRequests",
    "SimulationLogger",
    "EnvironmentState",
    "OperationPurpose",
    "EmptyPayload",
    "EngineStepPayload",
    "RequestArrivalPayload",
    "ShutdownPayload",
    "ShutdownReallocatePayload",
    "ShutdownMergePayload",
    "ShutdownSplitPayload",
    "BootPayload",
    "PayloadType",
    "SimulationEvent",
    "ActionHistoryKey",
    "AgentRatioKeys",
]


class RequestState(Enum):
    PENDING = "PENDING"
    PREFILLING = "PREFILLING"
    DECODING = "DECODING"
    COMPLETED = "COMPLETED"


class RunningRequests(ABC):
    @property
    @abstractmethod
    def prefill_requests(self) -> List[m.Request]: ...

    @prefill_requests.setter
    @abstractmethod
    def prefill_requests(self, value: List[m.Request]) -> None: ...

    @property
    @abstractmethod
    def decoding_requests(self) -> List[m.Request]: ...

    @abstractmethod
    def __len__(self) -> int: ...

    @property
    @abstractmethod
    def all_requests(self) -> List[m.Request]: ...


class SimulationLogger(ABC):
    @abstractmethod
    def log(self, message: str) -> None: ...

    @abstractmethod
    def flush(self) -> None: ...

    @abstractmethod
    def log_engine_step(
        self,
        current_time: float,
        agents: Dict[m.AgentId, m.Agent],
        stepping_engine: m.LLMEngine,
        next_arrival_time: Optional[float],
    ) -> None: ...

    @abstractmethod
    def log_request_arrival(
        self,
        current_time: float,
        req: m.Request,
        eng: Optional[m.LLMEngine],
    ) -> None: ...

    @abstractmethod
    def log_vram_transfer(
        self,
        current_time: float,
        giver_id: m.AgentId,
        receiver_id: m.AgentId,
        amount: int,
        eids: List[str],
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
        self, current_time: float, state: m.EnvironmentStateData
    ) -> None: ...


class EnvironmentState(ABC):
    @property
    @abstractmethod
    def current_budget(self) -> float: ...

    @current_budget.setter
    @abstractmethod
    def current_budget(self, v: float) -> None: ...

    @property
    @abstractmethod
    def reconfig_flag(self) -> bool: ...

    @reconfig_flag.setter
    @abstractmethod
    def reconfig_flag(self, v: bool) -> None: ...

    @property
    @abstractmethod
    def last_action_downtime(self) -> float: ...

    @property
    @abstractmethod
    def interval_requests(self) -> Dict[m.AgentId, List[m.Request]]: ...

    @last_action_downtime.setter
    @abstractmethod
    def last_action_downtime(self, v: float) -> None: ...

    @abstractmethod
    def advance_all_last_action(self) -> None: ...

    @abstractmethod
    def set_last_action(
        self, agent_id: m.AgentId, event_type: ActionHistoryKey, amount: int = 0
    ) -> None: ...

    @abstractmethod
    def reset_last_actions(self) -> None: ...

    @abstractmethod
    def refresh_budget(self) -> None: ...

    @abstractmethod
    def reset_for_next_interval(
        self, current_time: float, agents: Dict[m.AgentId, m.Agent]
    ) -> None: ...

    @abstractmethod
    def record_queue_length_advance(
        self, current_time: float, agents: Dict[m.AgentId, m.Agent]
    ) -> None: ...

    @abstractmethod
    def register_arrival(self, request: m.Request) -> None: ...

    @abstractmethod
    def get_state(
        self,
        current_time: float,
        agents: Dict[m.AgentId, m.Agent],
        gpu_current_state: Dict[int, int],
    ) -> m.EnvironmentStateData: ...

    @abstractmethod
    def get_steps_since(
        self, agent_id: m.AgentId, event_type: ActionHistoryKey
    ) -> int: ...


class OperationPurpose(Enum):
    REALLOCATE = "reallocate"
    MERGE = "merge"
    SPLIT = "split"
    PLAIN = "plain"


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
    request: m.Request


class ShutdownReallocatePayload(TypedDict):
    """ENGINE_SHUTDOWN_COMPLETE when purpose == OperationPurpose.REALLOCATE."""

    engine_id: str
    purpose: OperationPurpose
    receiver_id: m.AgentId


class ShutdownMergePayload(TypedDict):
    """ENGINE_SHUTDOWN_COMPLETE when purpose == OperationPurpose.MERGE."""

    engine_id: str
    purpose: OperationPurpose
    # Identifiers of both engines participating in the merge
    merge_engine_ids: Tuple[str, ...]
    # IDs that have already drained (grows as each engine shuts down)
    drained_ids: List[str]
    target_mig_indices: List[int]
    agent_id: m.AgentId
    gpu: int
    # If set, the merged engine boots directly on the receiver (VRAM transfer via merge)
    receiver_id: Optional[m.AgentId]
    target_state_id: int


class ShutdownSplitPayload(TypedDict):
    """ENGINE_SHUTDOWN_COMPLETE when purpose == OperationPurpose.SPLIT."""

    engine_id: str
    purpose: OperationPurpose
    target_mig_indices: List[int]
    agent_id: m.AgentId
    gpu: int
    # If set, it's VRAM transfer via split
    receiver_id: Optional[m.AgentId]
    target_state_id: int
    received_profile: Optional[m.MIGProfile]


ShutdownPayload = Union[
    ShutdownReallocatePayload,
    ShutdownMergePayload,
    ShutdownSplitPayload,
]


class BootPayload(TypedDict):
    engine_id: str
    purpose: OperationPurpose
    # IDs of all engines spawned by this split (used to detect when all are booted)
    sibling_engine_ids: List[str] | None


PayloadType = Union[
    EmptyPayload,
    EngineStepPayload,
    RequestArrivalPayload,
    ShutdownPayload,
    BootPayload,
]


@dataclass(order=True)
class SimulationEvent:
    time: float
    event_type: m.EventType
    payload: PayloadType = field(compare=False, repr=False)


ActionHistoryKey = Literal["split", "merge", "give", "receive"]
AgentRatioKeys = Literal[
    "agent_arrival_rate_ratio",
    "agent_avg_queue_len_ratio",
    "agent_avg_running_req_ratio",
    "agent_avg_kv_cache_ratio",
    "agent_avg_composite_latency_ratio",
    "agent_vram_ratio",
    "agent_sm_ratio",
]
