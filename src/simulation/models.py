from __future__ import annotations

from collections import deque
from enum import Enum, IntEnum
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple, Union, TypedDict, Literal

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
    "EnvironmentStateData",
    "EnvironmentState",
    "ResourceManagerAction",
    "VramTransferAction",
    "MigAction",
    "InitialMIGCombination",
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

    @classmethod
    def from_string(cls, profile_str: str) -> MIGProfile:
        for p in cls:
            if p.string == profile_str:
                return p
        raise ValueError(f"Invalid MIG profile string: {profile_str}")


type MIGConfigType = Tuple[MIGProfile, ...]


@dataclass(slots=True)
class MIGEncoding:
    p1: int = 0
    p2: int = 0
    p3: int = 0
    p4: int = 0
    p7: int = 0

    def __getitem__(self, index: int):
        match index:
            case 0:
                return self.p1
            case 1:
                return self.p2
            case 2:
                return self.p3
            case 3:
                return self.p4
            case 4:
                return self.p7
            case _:
                raise IndexError("MIGEncoding index out of range")

    def __setitem__(self, index: int, value: int):
        match index:
            case 0:
                self.p1 = value
            case 1:
                self.p2 = value
            case 2:
                self.p3 = value
            case 3:
                self.p4 = value
            case 4:
                self.p7 = value
            case _:
                raise IndexError("MIGEncoding index out of range")


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


class EventType(IntEnum):
    REQUEST_ARRIVAL = 0
    RAG_SEARCH_COMPLETE = 1
    ENGINE_STEP_COMPLETE = 2
    RESOURCE_MANAGER_TRIGGER = 3
    ENGINE_SHUTDOWN_COMPLETE = 4
    ENGINE_BOOT_COMPLETE = 5
    REFRESH_ACTION_BUDGET = 6


class OperationPurpose(Enum):
    REALLOCATE = "reallocate"
    MERGE = "merge"
    SPLIT = "split"
    PLAIN = "plain"


class InitialMIGCombination(Enum):
    C7 = (MIGProfile.MIG_7G_40GB,)
    C4_3 = (MIGProfile.MIG_4G_20GB, MIGProfile.MIG_3G_20GB)
    C4_2_1 = (
        MIGProfile.MIG_4G_20GB,
        MIGProfile.MIG_2G_10GB,
        MIGProfile.MIG_1G_10GB,
    )
    C3_2_2 = (
        MIGProfile.MIG_3G_20GB,
        MIGProfile.MIG_2G_10GB,
        MIGProfile.MIG_2G_10GB,
    )
    C2_2_2_1 = (
        MIGProfile.MIG_2G_10GB,
        MIGProfile.MIG_2G_10GB,
        MIGProfile.MIG_2G_10GB,
        MIGProfile.MIG_1G_10GB,
    )
    RANDOM = ()

    def __repr__(self) -> str:
        if self == InitialMIGCombination.RANDOM:
            return "RANDOM"
        return " | ".join([p.string for p in self.value])


# --- Interfaces ---


class Request(ABC):
    @property
    @abstractmethod
    def id(self) -> str: ...

    @abstractmethod
    def clone(self) -> Request: ...

    @property
    @abstractmethod
    def agent_id(self) -> AgentId: ...

    @property
    @abstractmethod
    def serving_engine(self) -> Optional[LLMEngine]: ...

    @serving_engine.setter
    def serving_engine(self, e: LLMEngine) -> None: ...

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
    event_type: EventType
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
    def completed_requests(self) -> deque[Request]: ...

    @abstractmethod
    def add_engine(self, engine: LLMEngine) -> None: ...

    @abstractmethod
    def dispatch(
        self, request: Request, current_time: float
    ) -> Optional[LLMEngine]: ...


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

    @abstractmethod
    def predict_drain_time(self) -> float: ...

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

    @property
    @abstractmethod
    def is_permanent(self) -> bool: ...

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

    @classmethod
    @abstractmethod
    def create(
        cls,
        gpu: int,
        engine_id: str,
        owner: Agent,
        mig_profile: MIGProfile,
        current_time: float,
        is_permanent: bool = False,
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
    def agents(self) -> Dict[AgentId, Agent]: ...

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

    @property
    @abstractmethod
    def interval_requests(self) -> Dict[AgentId, List[Request]]: ...

    @abstractmethod
    def need_requests_replenish(self) -> List[AgentId]: ...

    @abstractmethod
    def latest_arrival_time(self, agent_id: AgentId) -> float: ...

    @abstractmethod
    def init_simulator(self, requests: List[Request], max_steps: int) -> None: ...

    @abstractmethod
    def add_arrival_events(self, requests: List[Request]) -> None: ...

    @abstractmethod
    def has_active_work(self) -> bool: ...

    @abstractmethod
    def handle_resource_manager_trigger(
        self, action: ResourceManagerAction
    ) -> None: ...

    @abstractmethod
    def run(self) -> bool: ...

    @abstractmethod
    def reset(
        self,
        init_mode: InitialMIGCombination
        | Tuple[
            InitialMIGCombination, InitialMIGCombination
        ] = InitialMIGCombination.RANDOM,
    ) -> None: ...

    @abstractmethod
    def get_action_mask(self) -> List[bool]: ...

    @abstractmethod
    def get_state(self) -> EnvironmentStateData: ...


@dataclass
class VramTransferAction:
    giver: AgentId
    receiver: AgentId
    mig: MIGProfile


MIG_4_3 = (MIGProfile.MIG_4G_20GB, MIGProfile.MIG_3G_20GB)
MIG_4_2_1 = (
    MIGProfile.MIG_4G_20GB,
    MIGProfile.MIG_2G_10GB,
    MIGProfile.MIG_1G_10GB,
)
MIG_3_2_2 = (
    MIGProfile.MIG_3G_20GB,
    MIGProfile.MIG_2G_10GB,
    MIGProfile.MIG_2G_10GB,
)
MIG_2_2_2_1 = (
    MIGProfile.MIG_2G_10GB,
    MIGProfile.MIG_2G_10GB,
    MIGProfile.MIG_2G_10GB,
    MIGProfile.MIG_1G_10GB,
)
MIG_2_2 = (MIGProfile.MIG_2G_10GB, MIGProfile.MIG_2G_10GB)
MIG_2_1 = (MIGProfile.MIG_2G_10GB, MIGProfile.MIG_1G_10GB)


@dataclass
class MigAction:
    action: Literal["split", "merge"]
    victim: AgentId
    profiles: Tuple[MIGProfile, ...]


class ResourceManagerAction(Enum):
    NO_ACTION = None

    # 1-10: VRAM precise
    TRANSFER_7G_CODING_RAG = VramTransferAction(
        giver=AgentId.CODING, receiver=AgentId.RAG, mig=MIGProfile.MIG_7G_40GB
    )
    TRANSFER_4G_CODING_RAG = VramTransferAction(
        giver=AgentId.CODING, receiver=AgentId.RAG, mig=MIGProfile.MIG_4G_20GB
    )
    TRANSFER_3G_CODING_RAG = VramTransferAction(
        giver=AgentId.CODING, receiver=AgentId.RAG, mig=MIGProfile.MIG_3G_20GB
    )
    TRANSFER_2G_CODING_RAG = VramTransferAction(
        giver=AgentId.CODING, receiver=AgentId.RAG, mig=MIGProfile.MIG_2G_10GB
    )
    TRANSFER_1G_CODING_RAG = VramTransferAction(
        giver=AgentId.CODING, receiver=AgentId.RAG, mig=MIGProfile.MIG_1G_10GB
    )

    TRANSFER_7G_RAG_CODING = VramTransferAction(
        giver=AgentId.RAG, receiver=AgentId.CODING, mig=MIGProfile.MIG_7G_40GB
    )
    TRANSFER_4G_RAG_CODING = VramTransferAction(
        giver=AgentId.RAG, receiver=AgentId.CODING, mig=MIGProfile.MIG_4G_20GB
    )
    TRANSFER_3G_RAG_CODING = VramTransferAction(
        giver=AgentId.RAG, receiver=AgentId.CODING, mig=MIGProfile.MIG_3G_20GB
    )
    TRANSFER_2G_RAG_CODING = VramTransferAction(
        giver=AgentId.RAG, receiver=AgentId.CODING, mig=MIGProfile.MIG_2G_10GB
    )
    TRANSFER_1G_RAG_CODING = VramTransferAction(
        giver=AgentId.RAG, receiver=AgentId.CODING, mig=MIGProfile.MIG_1G_10GB
    )

    # 5-24: MIG per agent
    # Splits Coding
    SPLIT_7_TO_4_3_CODING = MigAction(
        "split",
        AgentId.CODING,
        profiles=MIG_4_3,
    )
    SPLIT_7_TO_3_2_2_CODING = MigAction(
        "split",
        AgentId.CODING,
        profiles=MIG_3_2_2,
    )
    SPLIT_7_TO_2_2_2_1_CODING = MigAction(
        "split",
        AgentId.CODING,
        profiles=MIG_2_2_2_1,
    )
    SPLIT_7_TO_4_2_1_CODING = MigAction(
        "split",
        AgentId.CODING,
        profiles=MIG_4_2_1,
    )
    SPLIT_4_TO_2_2_CODING = MigAction(
        "split",
        AgentId.CODING,
        profiles=MIG_2_2,
    )
    SPLIT_3_TO_2_1_CODING = MigAction(
        "split",
        AgentId.CODING,
        profiles=MIG_2_1,
    )

    # Splits RAG
    SPLIT_7_TO_4_3_RAG = MigAction("split", AgentId.RAG, profiles=MIG_4_3)
    SPLIT_7_TO_3_2_2_RAG = MigAction(
        "split",
        AgentId.RAG,
        profiles=MIG_3_2_2,
    )
    SPLIT_7_TO_2_2_2_1_RAG = MigAction(
        "split",
        AgentId.RAG,
        profiles=MIG_2_2_2_1,
    )
    SPLIT_7_TO_4_2_1_RAG = MigAction(
        "split",
        AgentId.RAG,
        profiles=MIG_4_2_1,
    )
    SPLIT_4_TO_2_2_RAG = MigAction("split", AgentId.RAG, profiles=MIG_2_2)
    SPLIT_3_TO_2_1_RAG = MigAction("split", AgentId.RAG, profiles=MIG_2_1)

    # Merges Coding
    MERGE_4_3_TO_7_CODING = MigAction(
        "merge",
        AgentId.CODING,
        profiles=MIG_4_3,
    )
    MERGE_3_2_2_TO_7_CODING = MigAction(
        "merge",
        AgentId.CODING,
        profiles=MIG_3_2_2,
    )
    MERGE_2_2_2_1_TO_7_CODING = MigAction(
        "merge",
        AgentId.CODING,
        profiles=MIG_2_2_2_1,
    )
    MERGE_4_2_1_TO_7_CODING = MigAction(
        "merge",
        AgentId.CODING,
        profiles=MIG_4_2_1,
    )
    MERGE_2_2_TO_4_CODING = MigAction(
        "merge",
        AgentId.CODING,
        profiles=MIG_2_2,
    )
    MERGE_2_1_TO_3_CODING = MigAction(
        "merge",
        AgentId.CODING,
        profiles=MIG_2_1,
    )

    # Merges RAG
    MERGE_4_3_TO_7_RAG = MigAction("merge", AgentId.RAG, profiles=MIG_4_3)
    MERGE_3_2_2_TO_7_RAG = MigAction(
        "merge",
        AgentId.RAG,
        profiles=MIG_3_2_2,
    )
    MERGE_2_2_2_1_TO_7_RAG = MigAction(
        "merge",
        AgentId.RAG,
        profiles=MIG_2_2_2_1,
    )
    MERGE_4_2_1_TO_7_RAG = MigAction(
        "merge",
        AgentId.RAG,
        profiles=MIG_4_2_1,
    )
    MERGE_2_2_TO_4_RAG = MigAction("merge", AgentId.RAG, profiles=MIG_2_2)
    MERGE_2_1_TO_3_RAG = MigAction("merge", AgentId.RAG, profiles=MIG_2_1)


# --- Management Interfaces ---


class EnvironmentStateData(TypedDict):
    arrival_rate: Dict[AgentId, float]
    predicted_arrival_rate: Dict[AgentId, float]
    arrival_rate_history: Dict[AgentId, Tuple[float, ...]]
    avg_queue_length: Dict[AgentId, Tuple[float, ...]]
    avg_queue_length_trend: Dict[AgentId, Tuple[float, ...]]
    kv_cache_utilization: Dict[AgentId, Tuple[float, ...]]
    avg_composite_latency: Dict[AgentId, Tuple[float, ...]]
    n_mig_instance: Dict[AgentId, float]
    agent_owns_mig: Dict[AgentId, Tuple[float, ...]]
    mig_geometry: Dict[int, List[float]]
    current_budget: float
    recovery_flag: bool
    avg_running_requests: Dict[AgentId, Tuple[float, ...]]
    downtime_ratio: float
    total_sm_ratio: Dict[AgentId, float]
    total_vram_ratio: Dict[AgentId, float]
    requests: Dict[AgentId, List[Request]]
    last_split: Dict[AgentId, float]
    last_merge: Dict[AgentId, float]
    last_give: Dict[AgentId, float]
    last_receive: Dict[AgentId, float]
    last_give_amount: Dict[AgentId, float]
    last_receive_amount: Dict[AgentId, float]
    # Agent Ratios (CODING / RAG)
    agent_arrival_rate_ratio: float
    agent_avg_queue_len_ratio: float
    agent_avg_running_req_ratio: float
    agent_avg_kv_cache_ratio: float
    agent_avg_composite_latency_ratio: float
    agent_n_mig_ratio: float
    agent_vram_ratio: float
    agent_sm_ratio: float


type ActionHistoryKey = Literal["split", "merge", "give", "receive"]


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
    def interval_requests(self) -> Dict[AgentId, List[Request]]: ...

    @last_action_downtime.setter
    @abstractmethod
    def last_action_downtime(self, v: float) -> None: ...

    @abstractmethod
    def advance_all_last_action(self) -> None: ...

    @abstractmethod
    def set_last_action(
        self, agent_id: AgentId, event_type: ActionHistoryKey, amount: int = 0
    ) -> None: ...

    @abstractmethod
    def reset_last_actions(self) -> None: ...

    @abstractmethod
    def refresh_budget(self) -> None: ...

    @abstractmethod
    def reset_for_next_interval(
        self, current_time: float, agents: Dict[AgentId, Agent]
    ) -> None: ...

    @abstractmethod
    def record_queue_length_advance(
        self, current_time: float, agents: Dict[AgentId, Agent]
    ) -> None: ...

    @abstractmethod
    def register_arrival(self, request: Request) -> None: ...

    @abstractmethod
    def get_state(
        self,
        current_time: float,
        agents: Dict[AgentId, Agent],
        engines: Dict[str, LLMEngine],
        current_step: int,
    ) -> EnvironmentStateData: ...

    @abstractmethod
    def get_steps_since(
        self, agent_id: AgentId, event_type: ActionHistoryKey
    ) -> int: ...


class MIGProfileRule(ABC):
    @abstractmethod
    def get_possible_merges(
        self, agent: Agent
    ) -> List[Tuple[List[LLMEngine], MIGProfile]]: ...

    @abstractmethod
    def get_possible_splits(
        self, agent: Agent
    ) -> List[Tuple[LLMEngine, List[MIGProfile]]]: ...

    @abstractmethod
    def get_best_specific_split(
        self, agent: Agent, target_profiles: Tuple[MIGProfile, ...]
    ) -> Tuple[LLMEngine, List[MIGProfile]] | None: ...

    @abstractmethod
    def get_best_specific_merge(
        self, agent: Agent, target_profiles: Tuple[MIGProfile, ...]
    ) -> Tuple[List[LLMEngine], MIGProfile] | None: ...

    @abstractmethod
    def has_exact_match(self, agent: Agent, mig: MIGProfile) -> bool: ...

    @abstractmethod
    def get_best_exact_match(
        self,
        giver_aid: AgentId,
        mig: MIGProfile,
        receiver_aid: AgentId,  # Pass the ID of the agent RECEIVING the MIG
        all_engines: List[LLMEngine],
    ) -> LLMEngine | None: ...

    @abstractmethod
    def select_best_split_action(
        self, agent: Agent, mask: List[bool], all_actions: List[ResourceManagerAction]
    ) -> ResourceManagerAction | None: ...

    @abstractmethod
    def select_best_merge_action(
        self, agent: Agent, mask: List[bool], all_actions: List[ResourceManagerAction]
    ) -> ResourceManagerAction | None: ...

    @abstractmethod
    def select_best_transfer_action(
        self,
        giver: Agent,
        receiver_aid: AgentId,
        mask: List[bool],
        all_actions: List[ResourceManagerAction],
        all_engines: List[LLMEngine],
    ) -> ResourceManagerAction | None: ...
