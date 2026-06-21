from __future__ import annotations

from enum import Enum, IntEnum
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import (
    Dict,
    Iterator,
    List,
    Optional,
    Tuple,
    NamedTuple,
    Type,
    TypeVar,
    TypedDict,
    Literal,
    Any,
)
from collections import deque
from sortedcontainers import SortedList

__all__ = [
    "EventType",
    "Request",
    "Agent",
    "AgentId",
    "MIGProfile",
    "MIGProfileBase",
    "ProfileInfo",
    "MIGConfigType",
    "LLMEngine",
    "Simulator",
    "ParamDict",
    "EngineStatus",
    "EnvironmentStateData",
    "ResourceManagerAction",
    "ActionType",
    "Action",
    "Receiver",
    "ResourceManagerActionValue",
]

ParamDict = Dict[Literal["alpha", "beta", "sigma"], float]


class MIGProfile(Enum):
    """Normalized MIG Profile Slots for RL Observation and Action Space."""

    MIG_7G = 0
    MIG_4G = 1
    MIG_3G = 2
    MIG_2G = 3
    MIG_1G_LARGE = 4
    MIG_1G_SMALL = 5

    @property
    def size(self) -> int:
        if self == MIGProfile.MIG_1G_LARGE or self == MIGProfile.MIG_1G_SMALL:
            return 1
        return [7, 4, 3, 2, 1, 1][self.value]

    @property
    def short_name(self) -> str:
        return ["7G", "4G", "3G", "2G", "1L", "1S"][self.value]

    def __eq__(self, other):
        if isinstance(other, MIGProfileBase):
            raise TypeError(
                f"Cannot compare logical MIGProfile with hardware-specific {type(other).__name__}. "
                "Use .profile_type comparison instead."
            )
        return super().__eq__(other)

    def __hash__(self):
        return super().__hash__()


class ProfileInfo(NamedTuple):
    size: int
    vram: int
    profile_type: MIGProfile


T = TypeVar("T", bound="MIGProfileBase")


class MIGProfileBase(Enum):
    """Base class for hardware-specific MIG profiles."""

    @property
    def size(self) -> int:
        return self.value.size

    @property
    def vram(self) -> int:
        return self.value.vram

    @property
    def profile_type(self) -> MIGProfile:
        """The logical normalized profile type."""
        return self.value.profile_type

    @property
    def gpu_model(self) -> str:
        """The GPU model name (e.g., A100_40GB)."""
        raise NotImplementedError()

    @classmethod
    def unsupported_profiles(cls) -> List[MIGProfile]:
        """Profiles that exist on hardware but are not supported by the RL/Simulation."""
        return []

    def __eq__(self, other):
        if isinstance(other, MIGProfile):
            raise TypeError(
                f"Cannot compare hardware-specific {type(self).__name__} with logical MIGProfile. "
                "Use .profile_type comparison instead."
            )
        return super().__eq__(other)

    def __hash__(self):
        return super().__hash__()

    @property
    def string(self) -> str:
        # We use size.vram to ensure unique keys in dictionaries
        return f"{self.size}g.{self.vram}gb"

    @property
    def idx(self) -> int:
        return self.profile_type.value

    @classmethod
    def from_string(cls, profile_str: str) -> "MIGProfileBase":
        """Create a profile instance from its string representation."""
        for member in cls:
            if member.string == profile_str:
                return member
        raise ValueError(f"Unknown {cls.__name__} profile: {profile_str}")

    @classmethod
    def __iter__(cls: Type[T]) -> Iterator[T]:
        """Explicitly define iteration for type checkers."""
        return iter(cls.__members__.values())

    def __str__(self):
        return self.string

    def __repr__(self):
        return f"<{self.__class__.__name__}.{self.name}: {self.string}>"


MIGConfigType = Tuple[MIGProfileBase, ...]


class EngineStatus(Enum):
    ACTIVE = "ACTIVE"
    DRAINING = "DRAINING"
    BOOTING = "BOOTING"


def _get_active_agent_count() -> int:
    import yaml
    from pathlib import Path

    try:
        config_path = Path("configs/simulation_config.yaml")
        if config_path.exists():
            with open(config_path, "r") as f:
                data = yaml.safe_load(f)
            
            agent_conf = data.get("agent")
            if agent_conf is not None:
                return sum(1 for is_active in agent_conf.values() if is_active)
            
            agents = data.get("simulation", {}).get("agents", {})
            if agents:
                return len(agents)
    except Exception:
        pass
    return 2


_active_agent_count = _get_active_agent_count()


class AgentId3Agent(Enum):
    CODING = "CodingAgent"
    RAG = "RAGAgent"
    CHAT = "ChatAgent"

class AgentId2Agent(Enum):
    CODING = "CodingAgent"
    RAG = "RAGAgent"

if _active_agent_count == 3:
    AgentId = AgentId3Agent
elif _active_agent_count == 2:
    AgentId = AgentId2Agent
else:
    raise ValueError("Invalid number of active agents")


class EventType(IntEnum):
    REQUEST_ARRIVAL = 0
    RAG_SEARCH_COMPLETE = 1
    ENGINE_STEP_COMPLETE = 2
    RESOURCE_MANAGER_TRIGGER = 3
    ENGINE_SHUTDOWN_COMPLETE = 4
    ENGINE_BOOT_COMPLETE = 5
    REFRESH_ACTION_BUDGET = 6


# --- Interfaces ---


class Request(ABC):
    @property
    @abstractmethod
    def id(self) -> str: ...

    @property
    @abstractmethod
    def prompt(self) -> str: ...

    @prompt.setter
    @abstractmethod
    def prompt(self, value: str) -> None: ...

    @abstractmethod
    def clone(self) -> Request: ...

    @property
    @abstractmethod
    def agent_id(self) -> AgentId: ...

    @property
    @abstractmethod
    def serving_engine(self) -> Optional[LLMEngine]: ...

    @serving_engine.setter
    @abstractmethod
    def serving_engine(self, e: LLMEngine) -> None: ...

    @property
    @abstractmethod
    def prompt_tokens(self) -> int: ...

    @prompt_tokens.setter
    def prompt_tokens(self, v: int): ...

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
    def state(self) -> Any: ...

    @state.setter
    @abstractmethod
    def state(self, value: Any) -> None: ...

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
    def mig_profile(self) -> MIGProfileBase: ...

    @property
    @abstractmethod
    def mig_index(self) -> int: ...

    @mig_index.setter
    @abstractmethod
    def mig_index(self, value: int): ...

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
    def running_queue(self) -> Any: ...

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
        mig_profile: MIGProfileBase,
        current_time: float,
        mig_index: int,
        is_permanent: bool = False,
    ) -> LLMEngine: ...

    @abstractmethod
    def trigger_shutdown(self, payload: Any, current_time: float) -> Optional[Any]: ...

    @abstractmethod
    def trigger_boot(self, payload: Any) -> Any: ...

    @abstractmethod
    def activate(self, current_time: float) -> None: ...

    @abstractmethod
    def step(
        self, current_time: float, next_arrival_time: Optional[float] = None
    ) -> Optional[Any]: ...


class Simulator(ABC):
    @property
    @abstractmethod
    def agents(self) -> Dict[AgentId, Agent]: ...

    @property
    @abstractmethod
    def engines(self) -> Dict[str, LLMEngine]: ...

    @property
    @abstractmethod
    def events(self) -> SortedList[Any]: ...

    @property
    @abstractmethod
    def current_time(self) -> float: ...

    @property
    @abstractmethod
    def logger(self) -> Any: ...

    @property
    @abstractmethod
    def gpu_engines(self) -> Dict[int, List[LLMEngine]]: ...

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
    def map_to_action(self, res_action: ResourceManagerAction) -> Optional[Action]: ...

    @abstractmethod
    def handle_resource_manager_trigger(self, action: Optional[Action]) -> None: ...

    @property
    @abstractmethod
    def gpu_current_state(self) -> Dict[int, int]: ...

    @abstractmethod
    def run(self) -> bool: ...

    @abstractmethod
    def reset(
        self,
        initial_state_mode: Literal["random", "no_mig", "split_extreme"] = "random",
    ) -> None: ...

    @abstractmethod
    def get_action_mask(self, ignore_cooldowns: bool = False) -> List[bool]: ...

    @abstractmethod
    def get_state(self) -> EnvironmentStateData: ...


class ActionType(Enum):
    SPLIT = "split"
    MERGE = "merge"
    TRANSFER = "transfer"


class Receiver(NamedTuple):
    receiver_id: AgentId
    mig_idx: int


class ResourceManagerActionValue(NamedTuple):
    gpu_id: int
    target_state_id: Optional[int]
    transfer_mig: Optional[MIGProfile]
    receiver_id: Optional[AgentId] = None


@dataclass
class Action:
    action: ActionType
    gpu_id: int
    mig_src: List[int]
    mig_target: List[int]
    target_state_id: Optional[int] = None
    receiver: Optional[Receiver] = None





# fmt: off
class ResourceManagerAction2Agent(Enum):
    NO_ACTION = None

    # GPU 0 State Transitions
    GPU_0_PROFILE_1_7G = ResourceManagerActionValue(0, 1, None)
    GPU_0_PROFILE_2_4G_3G = ResourceManagerActionValue(0, 2, None)
    GPU_0_PROFILE_3_4G_2G_1L = ResourceManagerActionValue(0, 3, None)
    GPU_0_PROFILE_4_4G_1S_1S_1L = ResourceManagerActionValue(0, 4, None)
    GPU_0_PROFILE_8_2G_2G_3G = ResourceManagerActionValue(0, 8, None)
    GPU_0_PROFILE_9_2G_1S_1S_3G = ResourceManagerActionValue(0, 9, None)
    GPU_0_PROFILE_10_1S_1S_2G_3G = ResourceManagerActionValue(0, 10, None)
    GPU_0_PROFILE_11_1S_1S_1S_1S_3G = ResourceManagerActionValue(0, 11, None)
    GPU_0_PROFILE_12_2G_2G_2G_1L = ResourceManagerActionValue(0, 12, None)
    GPU_0_PROFILE_13_2G_1S_1S_2G_1L = ResourceManagerActionValue(0, 13, None)
    GPU_0_PROFILE_14_1S_1S_2G_2G_1L = ResourceManagerActionValue(0, 14, None)
    GPU_0_PROFILE_15_2G_1S_1S_1S_1S_1L = ResourceManagerActionValue(0, 15, None)
    GPU_0_PROFILE_16_1S_1S_2G_1S_1S_1L = ResourceManagerActionValue(0, 16, None)
    GPU_0_PROFILE_17_1S_1S_1S_1S_2G_1L = ResourceManagerActionValue(0, 17, None)
    GPU_0_PROFILE_19_1S_1S_1S_1S_1S_1S_1L = ResourceManagerActionValue(0, 19, None)

    # GPU 1 State Transitions
    GPU_1_PROFILE_1_7G = ResourceManagerActionValue(1, 1, None)
    GPU_1_PROFILE_2_4G_3G = ResourceManagerActionValue(1, 2, None)
    GPU_1_PROFILE_3_4G_2G_1L = ResourceManagerActionValue(1, 3, None)
    GPU_1_PROFILE_4_4G_1S_1S_1L = ResourceManagerActionValue(1, 4, None)
    GPU_1_PROFILE_8_2G_2G_3G = ResourceManagerActionValue(1, 8, None)
    GPU_1_PROFILE_9_2G_1S_1S_3G = ResourceManagerActionValue(1, 9, None)
    GPU_1_PROFILE_10_1S_1S_2G_3G = ResourceManagerActionValue(1, 10, None)
    GPU_1_PROFILE_11_1S_1S_1S_1S_3G = ResourceManagerActionValue(1, 11, None)
    GPU_1_PROFILE_12_2G_2G_2G_1L = ResourceManagerActionValue(1, 12, None)
    GPU_1_PROFILE_13_2G_1S_1S_2G_1L = ResourceManagerActionValue(1, 13, None)
    GPU_1_PROFILE_14_1S_1S_2G_2G_1L = ResourceManagerActionValue(1, 14, None)
    GPU_1_PROFILE_15_2G_1S_1S_1S_1S_1L = ResourceManagerActionValue(1, 15, None)
    GPU_1_PROFILE_16_1S_1S_2G_1S_1S_1L = ResourceManagerActionValue(1, 16, None)
    GPU_1_PROFILE_17_1S_1S_1S_1S_2G_1L = ResourceManagerActionValue(1, 17, None)
    GPU_1_PROFILE_19_1S_1S_1S_1S_1S_1S_1L = ResourceManagerActionValue(1, 19, None)

    # GPU 0 Pure Transfer Explicit
    TRANSFER_GPU_0_7G_TO_CODING = ResourceManagerActionValue(0, None, MIGProfile.MIG_7G, receiver_id=AgentId2Agent.CODING)
    TRANSFER_GPU_0_7G_TO_RAG = ResourceManagerActionValue(0, None, MIGProfile.MIG_7G, receiver_id=AgentId2Agent.RAG)
    TRANSFER_GPU_0_4G_TO_CODING = ResourceManagerActionValue(0, None, MIGProfile.MIG_4G, receiver_id=AgentId2Agent.CODING)
    TRANSFER_GPU_0_4G_TO_RAG = ResourceManagerActionValue(0, None, MIGProfile.MIG_4G, receiver_id=AgentId2Agent.RAG)
    TRANSFER_GPU_0_3G_TO_CODING = ResourceManagerActionValue(0, None, MIGProfile.MIG_3G, receiver_id=AgentId2Agent.CODING)
    TRANSFER_GPU_0_3G_TO_RAG = ResourceManagerActionValue(0, None, MIGProfile.MIG_3G, receiver_id=AgentId2Agent.RAG)
    TRANSFER_GPU_0_2G_TO_CODING = ResourceManagerActionValue(0, None, MIGProfile.MIG_2G, receiver_id=AgentId2Agent.CODING)
    TRANSFER_GPU_0_2G_TO_RAG = ResourceManagerActionValue(0, None, MIGProfile.MIG_2G, receiver_id=AgentId2Agent.RAG)
    TRANSFER_GPU_0_1L_TO_CODING = ResourceManagerActionValue(0, None, MIGProfile.MIG_1G_LARGE, receiver_id=AgentId2Agent.CODING)
    TRANSFER_GPU_0_1L_TO_RAG = ResourceManagerActionValue(0, None, MIGProfile.MIG_1G_LARGE, receiver_id=AgentId2Agent.RAG)
    TRANSFER_GPU_0_1S_TO_CODING = ResourceManagerActionValue(0, None, MIGProfile.MIG_1G_SMALL, receiver_id=AgentId2Agent.CODING)
    TRANSFER_GPU_0_1S_TO_RAG = ResourceManagerActionValue(0, None, MIGProfile.MIG_1G_SMALL, receiver_id=AgentId2Agent.RAG)

    # GPU 1 Pure Transfer Explicit
    TRANSFER_GPU_1_7G_TO_CODING = ResourceManagerActionValue(1, None, MIGProfile.MIG_7G, receiver_id=AgentId2Agent.CODING)
    TRANSFER_GPU_1_7G_TO_RAG = ResourceManagerActionValue(1, None, MIGProfile.MIG_7G, receiver_id=AgentId2Agent.RAG)
    TRANSFER_GPU_1_4G_TO_CODING = ResourceManagerActionValue(1, None, MIGProfile.MIG_4G, receiver_id=AgentId2Agent.CODING)
    TRANSFER_GPU_1_4G_TO_RAG = ResourceManagerActionValue(1, None, MIGProfile.MIG_4G, receiver_id=AgentId2Agent.RAG)
    TRANSFER_GPU_1_3G_TO_CODING = ResourceManagerActionValue(1, None, MIGProfile.MIG_3G, receiver_id=AgentId2Agent.CODING)
    TRANSFER_GPU_1_3G_TO_RAG = ResourceManagerActionValue(1, None, MIGProfile.MIG_3G, receiver_id=AgentId2Agent.RAG)
    TRANSFER_GPU_1_2G_TO_CODING = ResourceManagerActionValue(1, None, MIGProfile.MIG_2G, receiver_id=AgentId2Agent.CODING)
    TRANSFER_GPU_1_2G_TO_RAG = ResourceManagerActionValue(1, None, MIGProfile.MIG_2G, receiver_id=AgentId2Agent.RAG)
    TRANSFER_GPU_1_1L_TO_CODING = ResourceManagerActionValue(1, None, MIGProfile.MIG_1G_LARGE, receiver_id=AgentId2Agent.CODING)
    TRANSFER_GPU_1_1L_TO_RAG = ResourceManagerActionValue(1, None, MIGProfile.MIG_1G_LARGE, receiver_id=AgentId2Agent.RAG)
    TRANSFER_GPU_1_1S_TO_CODING = ResourceManagerActionValue(1, None, MIGProfile.MIG_1G_SMALL, receiver_id=AgentId2Agent.CODING)
    TRANSFER_GPU_1_1S_TO_RAG = ResourceManagerActionValue(1, None, MIGProfile.MIG_1G_SMALL, receiver_id=AgentId2Agent.RAG)

class ResourceManagerAction3Agent(Enum):
    NO_ACTION = None

    # GPU 0 State Transitions
    GPU_0_PROFILE_1_7G = ResourceManagerActionValue(0, 1, None)
    GPU_0_PROFILE_2_4G_3G = ResourceManagerActionValue(0, 2, None)
    GPU_0_PROFILE_3_4G_2G_1L = ResourceManagerActionValue(0, 3, None)
    GPU_0_PROFILE_4_4G_1S_1S_1L = ResourceManagerActionValue(0, 4, None)
    GPU_0_PROFILE_8_2G_2G_3G = ResourceManagerActionValue(0, 8, None)
    GPU_0_PROFILE_9_2G_1S_1S_3G = ResourceManagerActionValue(0, 9, None)
    GPU_0_PROFILE_10_1S_1S_2G_3G = ResourceManagerActionValue(0, 10, None)
    GPU_0_PROFILE_11_1S_1S_1S_1S_3G = ResourceManagerActionValue(0, 11, None)
    GPU_0_PROFILE_12_2G_2G_2G_1L = ResourceManagerActionValue(0, 12, None)
    GPU_0_PROFILE_13_2G_1S_1S_2G_1L = ResourceManagerActionValue(0, 13, None)
    GPU_0_PROFILE_14_1S_1S_2G_2G_1L = ResourceManagerActionValue(0, 14, None)
    GPU_0_PROFILE_15_2G_1S_1S_1S_1S_1L = ResourceManagerActionValue(0, 15, None)
    GPU_0_PROFILE_16_1S_1S_2G_1S_1S_1L = ResourceManagerActionValue(0, 16, None)
    GPU_0_PROFILE_17_1S_1S_1S_1S_2G_1L = ResourceManagerActionValue(0, 17, None)
    GPU_0_PROFILE_19_1S_1S_1S_1S_1S_1S_1L = ResourceManagerActionValue(0, 19, None)

    # GPU 1 State Transitions
    GPU_1_PROFILE_1_7G = ResourceManagerActionValue(1, 1, None)
    GPU_1_PROFILE_2_4G_3G = ResourceManagerActionValue(1, 2, None)
    GPU_1_PROFILE_3_4G_2G_1L = ResourceManagerActionValue(1, 3, None)
    GPU_1_PROFILE_4_4G_1S_1S_1L = ResourceManagerActionValue(1, 4, None)
    GPU_1_PROFILE_8_2G_2G_3G = ResourceManagerActionValue(1, 8, None)
    GPU_1_PROFILE_9_2G_1S_1S_3G = ResourceManagerActionValue(1, 9, None)
    GPU_1_PROFILE_10_1S_1S_2G_3G = ResourceManagerActionValue(1, 10, None)
    GPU_1_PROFILE_11_1S_1S_1S_1S_3G = ResourceManagerActionValue(1, 11, None)
    GPU_1_PROFILE_12_2G_2G_2G_1L = ResourceManagerActionValue(1, 12, None)
    GPU_1_PROFILE_13_2G_1S_1S_2G_1L = ResourceManagerActionValue(1, 13, None)
    GPU_1_PROFILE_14_1S_1S_2G_2G_1L = ResourceManagerActionValue(1, 14, None)
    GPU_1_PROFILE_15_2G_1S_1S_1S_1S_1L = ResourceManagerActionValue(1, 15, None)
    GPU_1_PROFILE_16_1S_1S_2G_1S_1S_1L = ResourceManagerActionValue(1, 16, None)
    GPU_1_PROFILE_17_1S_1S_1S_1S_2G_1L = ResourceManagerActionValue(1, 17, None)
    GPU_1_PROFILE_19_1S_1S_1S_1S_1S_1S_1L = ResourceManagerActionValue(1, 19, None)

    # GPU 2 State Transitions
    GPU_2_PROFILE_1_7G = ResourceManagerActionValue(2, 1, None)
    GPU_2_PROFILE_2_4G_3G = ResourceManagerActionValue(2, 2, None)
    GPU_2_PROFILE_3_4G_2G_1L = ResourceManagerActionValue(2, 3, None)
    GPU_2_PROFILE_4_4G_1S_1S_1L = ResourceManagerActionValue(2, 4, None)
    GPU_2_PROFILE_8_2G_2G_3G = ResourceManagerActionValue(2, 8, None)
    GPU_2_PROFILE_9_2G_1S_1S_3G = ResourceManagerActionValue(2, 9, None)
    GPU_2_PROFILE_10_1S_1S_2G_3G = ResourceManagerActionValue(2, 10, None)
    GPU_2_PROFILE_11_1S_1S_1S_1S_3G = ResourceManagerActionValue(2, 11, None)
    GPU_2_PROFILE_12_2G_2G_2G_1L = ResourceManagerActionValue(2, 12, None)
    GPU_2_PROFILE_13_2G_1S_1S_2G_1L = ResourceManagerActionValue(2, 13, None)
    GPU_2_PROFILE_14_1S_1S_2G_2G_1L = ResourceManagerActionValue(2, 14, None)
    GPU_2_PROFILE_15_2G_1S_1S_1S_1S_1L = ResourceManagerActionValue(2, 15, None)
    GPU_2_PROFILE_16_1S_1S_2G_1S_1S_1L = ResourceManagerActionValue(2, 16, None)
    GPU_2_PROFILE_17_1S_1S_1S_1S_2G_1L = ResourceManagerActionValue(2, 17, None)
    GPU_2_PROFILE_19_1S_1S_1S_1S_1S_1S_1L = ResourceManagerActionValue(2, 19, None)

    # GPU 0 Pure Transfer Explicit
    TRANSFER_GPU_0_7G_TO_CODING = ResourceManagerActionValue(0, None, MIGProfile.MIG_7G, receiver_id=AgentId3Agent.CODING)
    TRANSFER_GPU_0_7G_TO_RAG = ResourceManagerActionValue(0, None, MIGProfile.MIG_7G, receiver_id=AgentId3Agent.RAG)
    TRANSFER_GPU_0_7G_TO_CHAT = ResourceManagerActionValue(0, None, MIGProfile.MIG_7G, receiver_id=AgentId3Agent.CHAT)
    TRANSFER_GPU_0_4G_TO_CODING = ResourceManagerActionValue(0, None, MIGProfile.MIG_4G, receiver_id=AgentId3Agent.CODING)
    TRANSFER_GPU_0_4G_TO_RAG = ResourceManagerActionValue(0, None, MIGProfile.MIG_4G, receiver_id=AgentId3Agent.RAG)
    TRANSFER_GPU_0_4G_TO_CHAT = ResourceManagerActionValue(0, None, MIGProfile.MIG_4G, receiver_id=AgentId3Agent.CHAT)
    TRANSFER_GPU_0_3G_TO_CODING = ResourceManagerActionValue(0, None, MIGProfile.MIG_3G, receiver_id=AgentId3Agent.CODING)
    TRANSFER_GPU_0_3G_TO_RAG = ResourceManagerActionValue(0, None, MIGProfile.MIG_3G, receiver_id=AgentId3Agent.RAG)
    TRANSFER_GPU_0_3G_TO_CHAT = ResourceManagerActionValue(0, None, MIGProfile.MIG_3G, receiver_id=AgentId3Agent.CHAT)
    TRANSFER_GPU_0_2G_TO_CODING = ResourceManagerActionValue(0, None, MIGProfile.MIG_2G, receiver_id=AgentId3Agent.CODING)
    TRANSFER_GPU_0_2G_TO_RAG = ResourceManagerActionValue(0, None, MIGProfile.MIG_2G, receiver_id=AgentId3Agent.RAG)
    TRANSFER_GPU_0_2G_TO_CHAT = ResourceManagerActionValue(0, None, MIGProfile.MIG_2G, receiver_id=AgentId3Agent.CHAT)
    TRANSFER_GPU_0_1L_TO_CODING = ResourceManagerActionValue(0, None, MIGProfile.MIG_1G_LARGE, receiver_id=AgentId3Agent.CODING)
    TRANSFER_GPU_0_1L_TO_RAG = ResourceManagerActionValue(0, None, MIGProfile.MIG_1G_LARGE, receiver_id=AgentId3Agent.RAG)
    TRANSFER_GPU_0_1L_TO_CHAT = ResourceManagerActionValue(0, None, MIGProfile.MIG_1G_LARGE, receiver_id=AgentId3Agent.CHAT)
    TRANSFER_GPU_0_1S_TO_CODING = ResourceManagerActionValue(0, None, MIGProfile.MIG_1G_SMALL, receiver_id=AgentId3Agent.CODING)
    TRANSFER_GPU_0_1S_TO_RAG = ResourceManagerActionValue(0, None, MIGProfile.MIG_1G_SMALL, receiver_id=AgentId3Agent.RAG)
    TRANSFER_GPU_0_1S_TO_CHAT = ResourceManagerActionValue(0, None, MIGProfile.MIG_1G_SMALL, receiver_id=AgentId3Agent.CHAT)

    # GPU 1 Pure Transfer Explicit
    TRANSFER_GPU_1_7G_TO_CODING = ResourceManagerActionValue(1, None, MIGProfile.MIG_7G, receiver_id=AgentId3Agent.CODING)
    TRANSFER_GPU_1_7G_TO_RAG = ResourceManagerActionValue(1, None, MIGProfile.MIG_7G, receiver_id=AgentId3Agent.RAG)
    TRANSFER_GPU_1_7G_TO_CHAT = ResourceManagerActionValue(1, None, MIGProfile.MIG_7G, receiver_id=AgentId3Agent.CHAT)
    TRANSFER_GPU_1_4G_TO_CODING = ResourceManagerActionValue(1, None, MIGProfile.MIG_4G, receiver_id=AgentId3Agent.CODING)
    TRANSFER_GPU_1_4G_TO_RAG = ResourceManagerActionValue(1, None, MIGProfile.MIG_4G, receiver_id=AgentId3Agent.RAG)
    TRANSFER_GPU_1_4G_TO_CHAT = ResourceManagerActionValue(1, None, MIGProfile.MIG_4G, receiver_id=AgentId3Agent.CHAT)
    TRANSFER_GPU_1_3G_TO_CODING = ResourceManagerActionValue(1, None, MIGProfile.MIG_3G, receiver_id=AgentId3Agent.CODING)
    TRANSFER_GPU_1_3G_TO_RAG = ResourceManagerActionValue(1, None, MIGProfile.MIG_3G, receiver_id=AgentId3Agent.RAG)
    TRANSFER_GPU_1_3G_TO_CHAT = ResourceManagerActionValue(1, None, MIGProfile.MIG_3G, receiver_id=AgentId3Agent.CHAT)
    TRANSFER_GPU_1_2G_TO_CODING = ResourceManagerActionValue(1, None, MIGProfile.MIG_2G, receiver_id=AgentId3Agent.CODING)
    TRANSFER_GPU_1_2G_TO_RAG = ResourceManagerActionValue(1, None, MIGProfile.MIG_2G, receiver_id=AgentId3Agent.RAG)
    TRANSFER_GPU_1_2G_TO_CHAT = ResourceManagerActionValue(1, None, MIGProfile.MIG_2G, receiver_id=AgentId3Agent.CHAT)
    TRANSFER_GPU_1_1L_TO_CODING = ResourceManagerActionValue(1, None, MIGProfile.MIG_1G_LARGE, receiver_id=AgentId3Agent.CODING)
    TRANSFER_GPU_1_1L_TO_RAG = ResourceManagerActionValue(1, None, MIGProfile.MIG_1G_LARGE, receiver_id=AgentId3Agent.RAG)
    TRANSFER_GPU_1_1L_TO_CHAT = ResourceManagerActionValue(1, None, MIGProfile.MIG_1G_LARGE, receiver_id=AgentId3Agent.CHAT)
    TRANSFER_GPU_1_1S_TO_CODING = ResourceManagerActionValue(1, None, MIGProfile.MIG_1G_SMALL, receiver_id=AgentId3Agent.CODING)
    TRANSFER_GPU_1_1S_TO_RAG = ResourceManagerActionValue(1, None, MIGProfile.MIG_1G_SMALL, receiver_id=AgentId3Agent.RAG)
    TRANSFER_GPU_1_1S_TO_CHAT = ResourceManagerActionValue(1, None, MIGProfile.MIG_1G_SMALL, receiver_id=AgentId3Agent.CHAT)

    # GPU 2 Pure Transfer Explicit
    TRANSFER_GPU_2_7G_TO_CODING = ResourceManagerActionValue(2, None, MIGProfile.MIG_7G, receiver_id=AgentId3Agent.CODING)
    TRANSFER_GPU_2_7G_TO_RAG = ResourceManagerActionValue(2, None, MIGProfile.MIG_7G, receiver_id=AgentId3Agent.RAG)
    TRANSFER_GPU_2_7G_TO_CHAT = ResourceManagerActionValue(2, None, MIGProfile.MIG_7G, receiver_id=AgentId3Agent.CHAT)
    TRANSFER_GPU_2_4G_TO_CODING = ResourceManagerActionValue(2, None, MIGProfile.MIG_4G, receiver_id=AgentId3Agent.CODING)
    TRANSFER_GPU_2_4G_TO_RAG = ResourceManagerActionValue(2, None, MIGProfile.MIG_4G, receiver_id=AgentId3Agent.RAG)
    TRANSFER_GPU_2_4G_TO_CHAT = ResourceManagerActionValue(2, None, MIGProfile.MIG_4G, receiver_id=AgentId3Agent.CHAT)
    TRANSFER_GPU_2_3G_TO_CODING = ResourceManagerActionValue(2, None, MIGProfile.MIG_3G, receiver_id=AgentId3Agent.CODING)
    TRANSFER_GPU_2_3G_TO_RAG = ResourceManagerActionValue(2, None, MIGProfile.MIG_3G, receiver_id=AgentId3Agent.RAG)
    TRANSFER_GPU_2_3G_TO_CHAT = ResourceManagerActionValue(2, None, MIGProfile.MIG_3G, receiver_id=AgentId3Agent.CHAT)
    TRANSFER_GPU_2_2G_TO_CODING = ResourceManagerActionValue(2, None, MIGProfile.MIG_2G, receiver_id=AgentId3Agent.CODING)
    TRANSFER_GPU_2_2G_TO_RAG = ResourceManagerActionValue(2, None, MIGProfile.MIG_2G, receiver_id=AgentId3Agent.RAG)
    TRANSFER_GPU_2_2G_TO_CHAT = ResourceManagerActionValue(2, None, MIGProfile.MIG_2G, receiver_id=AgentId3Agent.CHAT)
    TRANSFER_GPU_2_1L_TO_CODING = ResourceManagerActionValue(2, None, MIGProfile.MIG_1G_LARGE, receiver_id=AgentId3Agent.CODING)
    TRANSFER_GPU_2_1L_TO_RAG = ResourceManagerActionValue(2, None, MIGProfile.MIG_1G_LARGE, receiver_id=AgentId3Agent.RAG)
    TRANSFER_GPU_2_1L_TO_CHAT = ResourceManagerActionValue(2, None, MIGProfile.MIG_1G_LARGE, receiver_id=AgentId3Agent.CHAT)
    TRANSFER_GPU_2_1S_TO_CODING = ResourceManagerActionValue(2, None, MIGProfile.MIG_1G_SMALL, receiver_id=AgentId3Agent.CODING)
    TRANSFER_GPU_2_1S_TO_RAG = ResourceManagerActionValue(2, None, MIGProfile.MIG_1G_SMALL, receiver_id=AgentId3Agent.RAG)
    TRANSFER_GPU_2_1S_TO_CHAT = ResourceManagerActionValue(2, None, MIGProfile.MIG_1G_SMALL, receiver_id=AgentId3Agent.CHAT)
# fmt: on

if _active_agent_count == 3:
    ResourceManagerAction = ResourceManagerAction3Agent
elif _active_agent_count == 2:
    ResourceManagerAction = ResourceManagerAction2Agent
else:
    raise ValueError("Invalid number of active agents")


class EnvironmentStateData2Agent(TypedDict):
    arrival_rate: Dict[AgentId, float]
    predicted_arrival_rate: Dict[AgentId, float]
    arrival_rate_history: Dict[AgentId, Tuple[float, ...]]
    avg_queue_length: Dict[AgentId, Tuple[float, ...]]
    avg_queue_length_trend: Dict[AgentId, Tuple[float, ...]]
    kv_cache_utilization: Dict[AgentId, Tuple[float, ...]]
    avg_composite_latency: Dict[AgentId, Tuple[float, ...]]
    mig_profile_id_onehot: Dict[int, List[float]]
    ownership_grid: Dict[int, List[int]]
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
    agent_vram_ratio: float
    agent_sm_ratio: float


class EnvironmentStateData3Agent(TypedDict):
    arrival_rate: Dict[AgentId, float]
    predicted_arrival_rate: Dict[AgentId, float]
    arrival_rate_history: Dict[AgentId, Tuple[float, ...]]
    avg_queue_length: Dict[AgentId, Tuple[float, ...]]
    avg_queue_length_trend: Dict[AgentId, Tuple[float, ...]]
    kv_cache_utilization: Dict[AgentId, Tuple[float, ...]]
    avg_composite_latency: Dict[AgentId, Tuple[float, ...]]
    mig_profile_id_onehot: Dict[int, List[float]]
    ownership_grid: Dict[int, List[int]]
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
    give_to_steps: Dict[AgentId, Tuple[float, float, float]]
    # Agent Ratios (CODING/RAG, CODING/CHAT, RAG/CHAT)
    agent_arrival_rate_ratio: Tuple[float, float, float]
    agent_avg_queue_len_ratio: Tuple[float, float, float]
    agent_avg_running_req_ratio: Tuple[float, float, float]
    agent_avg_kv_cache_ratio: Tuple[float, float, float]
    agent_avg_composite_latency_ratio: Tuple[float, float, float]
    agent_vram_ratio: Tuple[float, float, float]
    agent_sm_ratio: Tuple[float, float, float]


if _active_agent_count == 3:
    EnvironmentStateData = EnvironmentStateData3Agent
else:
    EnvironmentStateData = EnvironmentStateData2Agent
