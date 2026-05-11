"""
src/deploy/cluster.py

Configures every MIG-capable GPU found on the machine to a randomly selected
valid MIG combination before a deployment run.

The set of managed GPUs and their hardware profile classes are **detected
automatically** at construction time via pynvml: only GPUs that have MIG mode
currently enabled are included.  NVML device names are matched against the
Python files in ``configs/gpus/`` using a flexible token-matching heuristic
so the class works correctly for any GPU model that has a corresponding config.

The valid combinations are derived from ``STATE_DEFINITIONS`` in exactly the
same way as the simulation (see ``src/simulation/config.py``), but instead of
just picking logical profile types we translate them to:

  1. Hardware profile strings understood by pynvml  (e.g. ``"4g.20gb"``)
  2. Concrete slice placements taken from ``SLICE_MAPPING``

Example usage::

    from src.deploy.cluster import DeployGPUSetup

    setup = DeployGPUSetup()
    # Inspect what was detected
    for gpu_idx, info in setup.gpu_info.items():
        print(f"GPU {gpu_idx}: {info.model_name}  ({len(info.valid_combos)} combos)")

    combos = setup.pick_random_combinations()
    # combos: {0: [("4g.20gb", 0), ("3g.20gb", 4)], 1: [...]}
    setup.apply(combos)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import random
import logging
import importlib.util
from pathlib import Path

import pynvml

import src.deploy.system as system
from src.deploy.mig_controller import MIGController

from src.deploy.models import (
    AllGpuMIGConfigs,
    DetectedGPU,
    GPUState,
    MIGSlotState,
    ProfilePlacement,
)
from src.share.mig_matrix import SLICE_MAPPING, STATE_DEFINITIONS
from src.share.models import MIGProfile, MIGProfileBase

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper: load hardware MIG profile class from configs/gpus/<model>.py
# ---------------------------------------------------------------------------


def _load_mig_profile_class(gpu_model: str) -> MIGProfileBase:
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


# ---------------------------------------------------------------------------
# Helper: derive valid combinations (mirrors logic in SimulationConfig.load)
# ---------------------------------------------------------------------------


def _derive_valid_combinations(
    mig_profile_cls: MIGProfileBase,
) -> List[Tuple[MIGProfile, ...]]:
    """Return valid logical profile tuples supported by *mig_profile_cls*.

    Mirrors ``GPU_VALID_COMBINATIONS`` construction in
    ``src/simulation/config.py``.
    """
    supported = {p.profile_type for p in mig_profile_cls}
    return [
        profiles
        for profiles in STATE_DEFINITIONS.values()
        if all(p in supported for p in profiles)
    ]


# ---------------------------------------------------------------------------
# Helper: translate a logical combo + state_id → hardware placement list
# ---------------------------------------------------------------------------


def _combo_to_placement(
    logical_combo: Tuple[MIGProfile, ...],
    state_id: int,
    mig_profile_cls: MIGProfileBase,
) -> List[ProfilePlacement]:
    """Translate a logical MIG profile tuple into a concrete hardware placement.

    Parameters
    ----------
    logical_combo:
        Tuple of :class:`~src.share.models.MIGProfile` values that form
        the chosen MIG state (in the same order as ``STATE_DEFINITIONS[state_id]``).
    state_id:
        The integer key in :data:`~src.share.mig_matrix.STATE_DEFINITIONS`
        that corresponds to *logical_combo*.
    mig_profile_cls:
        Hardware-specific :class:`~src.share.models.MIGProfileBase` enum
        class (e.g. ``MIGProfileA100``).

    Returns
    -------
    list of :class:`~src.deploy.models.ProfilePlacement`
        Ordered list ready to pass to
        :meth:`~src.deploy.mig_controller.MIGController.create_gi_with_placement`.
    """
    slice_groups: List[List[int]] = SLICE_MAPPING[state_id]
    assert len(logical_combo) == len(slice_groups), (
        f"Mismatch: state {state_id} has {len(slice_groups)} slice groups "
        f"but combo has {len(logical_combo)} profiles."
    )

    placements: List[ProfilePlacement] = []
    for logical_prof, slice_group in zip(logical_combo, slice_groups):
        hw_prof: MIGProfileBase = next(
            p for p in mig_profile_cls if p.profile_type == logical_prof
        )
        placements.append(ProfilePlacement(hw_prof, slice_group[0]))

    return placements


# ---------------------------------------------------------------------------
# Helper: match an NVML device name to a configs/gpus/<model>.py file
# ---------------------------------------------------------------------------


def _match_gpu_model(nvml_name: str, config_dir: Path) -> Optional[str]:
    """Find the best-matching config filename stem for *nvml_name*.

    The matching strategy normalises both strings to uppercase alphanumeric
    tokens and selects the config file whose tokens are all present in the
    NVML name tokens.  When multiple files match, the one with more matching
    tokens wins.

    Parameters
    ----------
    nvml_name:
        Raw string returned by ``nvmlDeviceGetName``,
        e.g. ``"NVIDIA A100-SXM4-40GB"``.
    config_dir:
        Directory containing ``*.py`` GPU config files.

    Returns
    -------
    str or None
        The matched filename stem (e.g. ``"A100_40GB"``), or ``None`` if no
        config file matches.
    """
    import re

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


# ---------------------------------------------------------------------------
# Hardware detection
# ---------------------------------------------------------------------------


def detect_mig_gpus(config_dir: Path = Path("configs/gpus")) -> List[DetectedGPU]:
    """Detect all GPUs with MIG mode currently enabled on this machine.

    Uses pynvml to enumerate GPUs, checks MIG mode, maps NVML device names to
    hardware config files, and builds :class:`DetectedGPU` records.

    Parameters
    ----------
    config_dir:
        Directory containing ``*.py`` GPU config files.
        Defaults to ``configs/gpus`` relative to the working directory.

    Returns
    -------
    list of :class:`DetectedGPU`
        One entry per MIG-enabled GPU, sorted by physical index.

    Raises
    ------
    RuntimeError
        If no MIG-enabled GPU is found on the machine.
    """
    pynvml.nvmlInit()
    try:
        count = pynvml.nvmlDeviceGetCount()
        detected: List[DetectedGPU] = []

        for idx in range(count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(idx)

            # Check MIG mode
            try:
                current_mode, _ = pynvml.nvmlDeviceGetMigMode(handle)
            except pynvml.NVMLError:
                # GPU does not support MIG at all
                continue

            if current_mode != pynvml.NVML_DEVICE_MIG_ENABLE:
                logger.debug("GPU %d: MIG mode not enabled — skipping.", idx)
                continue

            nvml_name = pynvml.nvmlDeviceGetName(handle)
            logger.info("GPU %d: MIG enabled  name=%r", idx, nvml_name)

            model_name = _match_gpu_model(nvml_name, config_dir)
            if model_name is None:
                logger.warning(
                    "GPU %d (%r): no matching config file in %s — skipping.",
                    idx,
                    nvml_name,
                    config_dir,
                )
                continue

            mig_profile_cls = _load_mig_profile_class(model_name)
            valid_combos = _derive_valid_combinations(mig_profile_cls)
            if not valid_combos:
                logger.warning(
                    "GPU %d (%s): no valid MIG combinations found — skipping.",
                    idx,
                    model_name,
                )
                continue

            combo_to_state_id: Dict[Tuple[MIGProfile, ...], int] = {
                profiles: sid
                for sid, profiles in STATE_DEFINITIONS.items()
                if profiles in valid_combos
            }

            detected.append(
                DetectedGPU(
                    gpu_idx=idx,
                    model_name=model_name,
                    nvml_name=nvml_name,
                    mig_profile_cls=mig_profile_cls,
                    valid_combos=valid_combos,
                    combo_to_state_id=combo_to_state_id,
                )
            )
            logger.info(
                "GPU %d: matched model=%s  valid_combos=%d",
                idx,
                model_name,
                len(valid_combos),
            )
    finally:
        pynvml.nvmlShutdown()

    if not detected:
        raise RuntimeError(
            "No MIG-enabled GPUs found on this machine.  "
            "Enable MIG mode with: nvidia-smi -i <gpu_idx> -mig 1"
        )

    return sorted(detected, key=lambda d: d.gpu_idx)


# ---------------------------------------------------------------------------
# DeployGPUSetup
# ---------------------------------------------------------------------------


class DeployGPUSetup:
    """Selects and applies valid MIG configurations for all MIG-capable GPUs.

    On construction the class automatically detects every GPU that has MIG
    mode enabled, matches it to a hardware config in ``configs/gpus/``, and
    derives the set of valid MIG combinations — exactly the same logic used
    by :class:`~src.simulation.config.SimulationConfig`.

    Parameters
    ----------
    seed:
        Optional random seed for reproducibility.
    config_dir:
        Directory containing ``*.py`` GPU config files.
        Defaults to ``configs/gpus`` relative to the working directory.

    Attributes
    ----------
    gpu_info : dict[int, DetectedGPU]
        Maps each detected GPU index to its :class:`DetectedGPU` record.

    Examples
    --------
    ::

        setup = DeployGPUSetup()
        combos = setup.pick_random_combinations()
        # combos: {0: [("4g.20gb", 0), ("3g.20gb", 4)], 1: [...]}
        setup.apply(combos)
    """

    def __init__(
        self,
        seed: Optional[int] = None,
        config_dir: Path = Path("configs/gpus"),
    ) -> None:
        if seed is not None:
            random.seed(seed)

        detected = detect_mig_gpus(config_dir)
        self.gpu_info: Dict[int, DetectedGPU] = {d.gpu_idx: d for d in detected}

        logger.info(
            "DeployGPUSetup: detected %d MIG GPU(s): %s",
            len(self.gpu_info),
            {idx: d.model_name for idx, d in self.gpu_info.items()},
        )

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def gpu_indices(self) -> List[int]:
        """Physical indices of all detected MIG-capable GPUs."""
        return sorted(self.gpu_info.keys())

    def valid_combinations(self, gpu_idx: int) -> List[Tuple[MIGProfile, ...]]:
        """All valid logical MIG profile combinations for *gpu_idx*."""
        return list(self.gpu_info[gpu_idx].valid_combos)

    # ------------------------------------------------------------------
    # Selecting combinations
    # ------------------------------------------------------------------

    def pick_random_combination(
        self, gpu_idx: int
    ) -> Tuple[Tuple[MIGProfile, ...], int]:
        """Randomly pick one valid logical combination for *gpu_idx*.

        Returns
        -------
        tuple
            ``(logical_combo, state_id)`` where *logical_combo* is a tuple of
            :class:`~src.share.models.MIGProfile` values and *state_id*
            is the corresponding key in ``STATE_DEFINITIONS``.
        """
        info = self.gpu_info[gpu_idx]
        combo = random.choice(info.valid_combos)
        state_id = info.combo_to_state_id[combo]
        return combo, state_id

    def pick_random_combinations(self) -> AllGpuMIGConfigs:
        """Independently pick a random valid configuration for each detected GPU.

        Returns
        -------
        dict
            ``{gpu_idx: [(profile_string, start_slice), ...]}`` for every
            detected MIG GPU.  Suitable for passing directly to :meth:`apply`.
        """
        result: AllGpuMIGConfigs = {}
        for gpu_idx, info in self.gpu_info.items():
            combo, state_id = self.pick_random_combination(gpu_idx)
            placements = _combo_to_placement(combo, state_id, info.mig_profile_cls)
            result[gpu_idx] = placements
            logger.info(
                "GPU %d (%s): selected state_id=%d  combo=%s  placements=%s",
                gpu_idx,
                info.model_name,
                state_id,
                tuple(p.name for p in combo),
                placements,
            )
        return result

    # ------------------------------------------------------------------
    # Applying configurations
    # ------------------------------------------------------------------

    def apply(
        self,
        configs: AllGpuMIGConfigs,
        *,
        dry_run: bool = False,
    ) -> None:
        """Apply MIG configurations to the physical GPUs and register state.

        After all GPU instances are created, MIG device UUIDs are resolved via
        NVML and the resulting :class:`~src.deploy.models.GPUState` is stored
        in :data:`~src.deploy.system.SYSTEM_STATE` so that
        :class:`~src.deploy.vllm.VLLMManager` can find the UUIDs for docker.

        Parameters
        ----------
        configs:
            Mapping from gpu_idx to a list of :class:`~src.deploy.models.ProfilePlacement`
            objects, as returned by :meth:`pick_random_combinations`.
        dry_run:
            If ``True``, log what *would* be done but do not call any NVML
            functions.  Useful for testing without root privileges.
        """
        if dry_run:
            logger.info("[DRY RUN] Would apply the following MIG configurations:")
            for gpu_idx, placements in configs.items():
                logger.info("  GPU %d: %s", gpu_idx, placements)
            return

        with MIGController(gpu_indices=self.gpu_indices) as ctrl:
            # 1. Apply hardware MIG configurations.
            for gpu_idx, placements in configs.items():
                ctrl.apply_full_configuration(gpu_idx, placements)

            # 2. Resolve MIG UUIDs while NVML is still active, then register
            #    the resulting GPUState into the global SYSTEM_STATE.
            for gpu_idx, placements in configs.items():
                info = self.gpu_info[gpu_idx]
                
                # Register the GPU first so MIGController can look up the profile class
                system.register_gpu(
                    GPUState(gpu_idx, info.model_name, info.mig_profile_cls)
                )
                
                uuid_map = dict(ctrl.list_mig_device_uuids(gpu_idx))
                
                slots = [
                    MIGSlotState(
                        gpu_idx=gpu_idx,
                        profile_placement=p,
                        mig_uuid=uuid_map[p.start_slice],
                    )
                    for p in placements
                ]
                gpu_state = GPUState(
                    gpu_idx=gpu_idx,
                    model_name=info.model_name,
                    mig_profile_cls=info.mig_profile_cls,
                    slots=sorted(slots, key=lambda s: s.profile_placement.start_slice),
                )
                system.register_gpu(gpu_state)

        logger.info("All GPU MIG configurations applied and SYSTEM_STATE updated.")

    def apply_random(self, *, dry_run: bool = False) -> AllGpuMIGConfigs:
        """Pick random valid combinations and immediately apply them.

        This is the primary one-shot entry point for the setup script.

        Parameters
        ----------
        dry_run:
            Passed through to :meth:`apply`.

        Returns
        -------
        AllGpuMIGConfigs
            The configurations that were (or would be) applied.
        """
        configs = self.pick_random_combinations()
        self.apply(configs, dry_run=dry_run)
        return configs
