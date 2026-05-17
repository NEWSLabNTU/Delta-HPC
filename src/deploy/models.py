"""
src/deploy/models.py

Shared data types for the deploy package.

All dataclasses and type aliases that cross module boundaries within
``src/deploy/`` are defined here so that other files can import them from
a single, stable location without creating circular dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, TypedDict

from src.share.models import AgentId, MIGProfileBase


class SimulatedEngineConfig(TypedDict):
    """Configuration for a single simulated permanent engine."""

    agent: str
    mig: str


class SimulatedGPUConfig(TypedDict):
    """Configuration for a simulated GPU and its permanent engines."""

    model: str
    state_id: int
    permanent_engines: List[SimulatedEngineConfig]


# Mapping of gpu_idx → ordered list of ProfilePlacement, covering all managed GPUs.
AllGpuMIGConfigs = Dict[int, List["ProfilePlacement"]]


# ---------------------------------------------------------------------------
# MIGController data containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProfilePlacement:
    """Describes a single MIG GPU instance that should be created.

    Attributes
    ----------
    profile:
        Hardware profile enum member (e.g. from ``MIGProfileA100``).
    start_slice:
        Index of the first memory slice to occupy (0-indexed).  Together with
        the profile's slice count this fully determines placement.
    """

    profile: MIGProfileBase
    start_slice: int


@dataclass
class GpuInstanceInfo:
    """Metadata returned for an existing GPU instance.

    Attributes
    ----------
    instance_id:
        NVML GPU-instance ID.
    profile:
        Hardware profile enum member.
    start_slice:
        Starting memory-slice index.
    slice_count:
        Number of memory slices occupied.
    """

    instance_id: int
    profile: MIGProfileBase
    start_slice: int
    slice_count: int

    def __repr__(self) -> str:
        return (
            f"GpuInstanceInfo(id={self.instance_id}, "
            f"profile={self.profile.string}, "
            f"slices=[{self.start_slice}, {self.start_slice + self.slice_count - 1}])"
        )


# ---------------------------------------------------------------------------
# DeployGPUSetup data containers
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Live system state
# ---------------------------------------------------------------------------


@dataclass
class MIGSlotState:
    """Live state of a single physical MIG GPU-instance slot.

    Attributes
    ----------
    gpu_idx:
        Physical GPU index.
    profile_placement:
        Hardware profile enum member and starting slice (from :class:`ProfilePlacement`).
    mig_uuid:
        NVML MIG compute-instance UUID (e.g. ``"MIG-GPU-xxxx.../1/0"``).  Used
        as the ``device_ids`` entry in docker-compose.
    model_id:
        vLLM model config ID currently running on this slot
        (e.g. ``"qwen2.5-7b-instruct"``), or ``None`` when no container is up.
    port:
        Host TCP port the vLLM container listens on, or ``None``.
    container_name:
        Docker compose project name for this slot, or ``None``.
    agent_id:
        The RL agent (:class:`~src.share.models.AgentId`) assigned to this slot,
        or ``None``.
    """

    gpu_idx: int
    profile_placement: ProfilePlacement
    mig_uuid: str
    model_id: Optional[str] = None
    port: Optional[int] = None
    container_name: Optional[str] = None
    agent_id: Optional[AgentId] = None
    is_draining: bool = False
    is_ready: bool = False


@dataclass
class GPUState:
    """Live state of one physical GPU.

    Attributes
    ----------
    gpu_idx:
        Physical GPU index.
    model_name:
        Hardware model name, e.g. ``"A100_40GB"``.
    mig_profile_cls:
        Hardware-specific :class:`~src.share.models.MIGProfileBase` enum class.
    slots:
        Ordered list of :class:`MIGSlotState` objects, one per active MIG slot,
        sorted by :attr:`ProfilePlacement.start_slice`.
    is_simulated:
        ``True`` for GPUs that are emulated in software (permanent-engine only)
        and therefore not subject to MIG reconfiguration by the RL agent.
    """

    gpu_idx: int
    model_name: str
    mig_profile_cls: type[MIGProfileBase]
    slots: List[MIGSlotState] = field(default_factory=list)
    is_simulated: bool = False


@dataclass
class ClusterState:
    """Full live state of the deployed cluster.

    Attributes
    ----------
    gpus:
        Mapping from physical GPU index to its current :class:`GPUState`.
        Only GPUs that have been configured by :class:`~src.deploy.cluster.DeployGPUSetup`
        appear here.
    """

    gpus: Dict[int, GPUState] = field(default_factory=dict)
