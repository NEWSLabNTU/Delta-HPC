"""
src/deploy/models.py

Shared data types for the deploy package.

All dataclasses and type aliases that cross module boundaries within
``src/deploy/`` are defined here so that other files can import them from
a single, stable location without creating circular dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from src.simulation.models import MIGProfile

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
    profile_string:
        Hardware profile string in the form ``"{size}g.{vram}gb"``,
        e.g. ``"4g.20gb"``, ``"1g.10gb"``.
    start_slice:
        Index of the first memory slice to occupy (0-indexed).  Together with
        the profile's slice count this fully determines placement.
    """

    profile_string: str
    start_slice: int


@dataclass
class GpuInstanceInfo:
    """Metadata returned for an existing GPU instance.

    Attributes
    ----------
    instance_id:
        NVML GPU-instance ID.
    profile_string:
        Human-readable profile string, e.g. ``"4g.20gb"``.
    start_slice:
        Starting memory-slice index.
    slice_count:
        Number of memory slices occupied.
    """

    instance_id: int
    profile_string: str
    start_slice: int
    slice_count: int

    def __repr__(self) -> str:
        return (
            f"GpuInstanceInfo(id={self.instance_id}, "
            f"profile={self.profile_string}, "
            f"slices=[{self.start_slice}, {self.start_slice + self.slice_count - 1}])"
        )


# ---------------------------------------------------------------------------
# DeployGPUSetup data containers
# ---------------------------------------------------------------------------


@dataclass
class DetectedGPU:
    """All information about a single detected MIG-capable GPU.

    Attributes
    ----------
    gpu_idx:
        Physical GPU index (matches ``nvidia-smi`` index).
    model_name:
        Config-file model name, e.g. ``"A100_40GB"``.
    nvml_name:
        Raw NVML device name, e.g. ``"NVIDIA A100-SXM4-40GB"``.
    mig_profile_cls:
        Hardware-specific :class:`~src.simulation.models.MIGProfileBase`
        enum class loaded from ``configs/gpus/<model_name>.py``.
    valid_combos:
        All valid logical MIG profile tuples for this GPU (mirrors
        ``GPU_VALID_COMBINATIONS`` in the simulation config).
    combo_to_state_id:
        Reverse map from each valid combo tuple to its ``STATE_DEFINITIONS``
        key (state ID), used to look up the corresponding ``SLICE_MAPPING``.
    """

    gpu_idx: int
    model_name: str
    nvml_name: str
    mig_profile_cls: type
    valid_combos: List[Tuple[MIGProfile, ...]] = field(default_factory=list)
    combo_to_state_id: Dict[Tuple[MIGProfile, ...], int] = field(default_factory=dict)
