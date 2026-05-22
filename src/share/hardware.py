"""
src/share/hardware.py

Shared hardware detection utilities for the Delta-HPC infrastructure.
"""

from __future__ import annotations

import re
import logging
import importlib.util
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from src.share.mig_matrix import STATE_DEFINITIONS
from src.share.models import MIGProfile, MIGProfileBase

logger = logging.getLogger(__name__)


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
        Hardware-specific :class:`~src.share.models.MIGProfileBase`
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
    mig_profile_cls: MIGProfileBase
    valid_combos: List[Tuple[MIGProfile, ...]] = field(default_factory=list)
    combo_to_state_id: Dict[Tuple[MIGProfile, ...], int] = field(default_factory=dict)


def load_mig_profile_class(gpu_model: str) -> MIGProfileBase:
    """Import and return the ``MIG_PROFILE`` enum class for *gpu_model*.

    Parameters
    ----------
    gpu_model:
        GPU model name matching a file in ``configs/gpus/``, e.g. ``"A100_40GB"``.

    Returns
    -------
    type
        The ``MIGProfileBase`` subclass (enum) defined in that file.
    """
    module_path = Path(f"configs/gpus/{gpu_model}.py")
    if not module_path.exists():
        raise FileNotFoundError(
            f"GPU config file not found: {module_path}.  "
            f"Available: {list(Path('configs/gpus').glob('*.py'))}"
        )
    spec = importlib.util.spec_from_file_location(f"gpu_mod_{gpu_model}", module_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod.MIG_PROFILE  # type: ignore[return-value]


def derive_valid_combinations(
    mig_profile_cls: MIGProfileBase,
) -> List[Tuple[MIGProfile, ...]]:
    """Return valid logical profile tuples supported by *mig_profile_cls*.

    Mirrors ``GPU_VALID_COMBINATIONS`` construction in
    ``src/simulation/config.py``.
    """
    unsupported = mig_profile_cls.unsupported_profiles()
    supported = {
        p.profile_type for p in mig_profile_cls if p.profile_type not in unsupported
    }
    result = []
    for profiles in STATE_DEFINITIONS.values():
        if all(p in supported for p in profiles):
            result.append(profiles)
    return result


def match_gpu_model(nvml_name: str, config_dir: Path) -> Optional[str]:
    """Find the best-matching config filename stem for *nvml_name*."""

    def tokenise(s: str) -> List[str]:
        return re.findall(r"[A-Z0-9]+", s.upper())

    nvml_tokens = set(tokenise(nvml_name))

    best_match: Optional[str] = None
    best_score = -1

    for cfg_file in config_dir.glob("*.py"):
        if cfg_file.stem.startswith("__"):
            continue
        cfg_tokens = tokenise(cfg_file.stem)
        # All tokens from the config name must be present in the NVML name.
        if all(t in nvml_tokens for t in cfg_tokens):
            score = len(cfg_tokens)
            if score > best_score:
                best_score = score
                best_match = cfg_file.stem

    return best_match


# Deleted detect_mig_gpus from here
