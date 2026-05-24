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
from pathlib import Path

import src.deploy.system as system
from src.share.hardware import DetectedGPU, load_mig_profile_class
from src.deploy.mig_controller import MIGController
from src.deploy.config import DEPLOY_CONFIG
from src.deploy.models import (
    AllGpuMIGConfigs,
    GPUState,
    MIGSlotState,
    ProfilePlacement,
)
from src.share.mig_matrix import SLICE_MAPPING, STATE_DEFINITIONS
from src.share.models import MIGProfile, MIGProfileBase
import src.share.models as m

logger = logging.getLogger(__name__)


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

        detected = MIGController.detect_mig_gpus(config_dir)
        self.gpu_info: Dict[int, DetectedGPU] = {d.gpu_idx: d for d in detected}
        self.mig_ctrl = MIGController(gpu_indices=self.gpu_indices)

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
                "GPU %d (%s): selected state_id=%d  placements=%s",
                gpu_idx,
                info.model_name,
                state_id,
                placements,
            )
        return result

    def pick_fixed_combinations(self, state_id: int) -> AllGpuMIGConfigs:
        """Pick a specific valid configuration for each detected GPU by state_id.

        Parameters
        ----------
        state_id:
            The state_id corresponding to the desired MIG configuration.

        Returns
        -------
        dict
            ``{gpu_idx: [(profile_string, start_slice), ...]}`` for every
            detected MIG GPU. Suitable for passing directly to :meth:`apply`.
        """
        result: AllGpuMIGConfigs = {}
        for gpu_idx, info in self.gpu_info.items():
            combo = STATE_DEFINITIONS[state_id]
            placements = _combo_to_placement(combo, state_id, info.mig_profile_cls)
            result[gpu_idx] = placements
            logger.info(
                "GPU %d (%s): selected fixed state_id=%d  combo=%s  placements=%s",
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

        # 1. Apply hardware MIG configurations.
        for gpu_idx, placements in configs.items():
            self.mig_ctrl.apply_full_configuration(gpu_idx, placements)

        # 2. Resolve MIG UUIDs, then register
        #    the resulting GPUState into the global SYSTEM_STATE.
        for gpu_idx, placements in configs.items():
            info = self.gpu_info[gpu_idx]

            # Register the GPU first so MIGController can look up the profile class
            system.register_gpu(
                GPUState(gpu_idx, info.model_name, info.mig_profile_cls)
            )

            uuid_map = dict(self.mig_ctrl.list_mig_device_uuids(gpu_idx))

            slots = [
                MIGSlotState(
                    gpu_idx=gpu_idx,
                    profile_placement=p,
                    mig_uuid=uuid_map[p.start_slice],
                    agent_id=m.AgentId(DEPLOY_CONFIG.gpu_assignment[gpu_idx]),
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

    def cleanup(self) -> None:
        """Destroy all MIG instances on all managed GPUs."""
        logger.info("Cleaning up all MIG instances on managed GPUs...")
        for gpu_idx in self.gpu_indices:
            self.mig_ctrl.disable_all_instances(gpu_idx)

    def register_simulated_gpus(self) -> None:
        """Register simulated GPUs and permanent engines into SYSTEM_STATE.

        This method mirrors the permanent engine configuration defined in
        DEPLOY_CONFIG.simulated_gpus for deployment environments that lack the extra hardware.
        """

        for gpu_id, cfg in DEPLOY_CONFIG.simulated_gpus.items():
            # Only register if this GPU is NOT a physical MIG-enabled GPU
            if gpu_id in self.gpu_info:
                logger.warning(
                    f"GPU {gpu_id} is detected as physical hardware. "
                    "Skipping simulation registration for this GPU."
                )
                continue

            model_name = cfg["model"]
            state_id = cfg.get("state_id")
            mig_profile_cls = load_mig_profile_class(model_name)

            slots = []
            target_slices = SLICE_MAPPING[state_id]
            target_profiles = STATE_DEFINITIONS[state_id]

            # Match permanent engines to slots in the defined state
            for i, engine in enumerate(cfg["permanent_engines"]):
                mig_str = engine["mig"]
                agent_id = m.AgentId(engine["agent"])

                # Convert mig_str to a logical profile type for comparison
                hw_prof = next(p for p in mig_profile_cls if p.string == mig_str)
                target_prof_type = hw_prof.profile_type

                # Find a slot in the state that matches the requested MIG profile
                found = False
                for slot_idx, prof in enumerate(target_profiles):
                    if prof == target_prof_type:
                        # Check if this slot is already taken
                        start_slice = target_slices[slot_idx][0]
                        if any(
                            s.profile_placement.start_slice == start_slice
                            for s in slots
                        ):
                            continue

                        slots.append(
                            MIGSlotState(
                                gpu_idx=gpu_id,
                                profile_placement=ProfilePlacement(
                                    hw_prof, start_slice
                                ),
                                mig_uuid=f"SIM-MIG-GPU-{gpu_id}-{slot_idx}",
                                model_id=None,
                                agent_id=agent_id,
                            )
                        )
                        found = True
                        break

                if not found:
                    logger.error(
                        f"Simulated GPU {gpu_id}: Could not find a '{mig_str}' slot in state {state_id}."
                    )

            gpu_state = GPUState(
                gpu_idx=gpu_id,
                model_name=model_name,
                mig_profile_cls=mig_profile_cls,
                slots=slots,
                is_simulated=True,
            )
            system.register_gpu(gpu_state)
            logger.info(
                f"Registered simulated GPU {gpu_id} ({model_name}) with {len(slots)} permanent engines."
            )

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

    def apply_fixed(self, state_id: int, *, dry_run: bool = False) -> AllGpuMIGConfigs:
        """Pick fixed combinations based on state_id and immediately apply them.

        Parameters
        ----------
        state_id:
            The state_id corresponding to the desired MIG configuration.
        dry_run:
            Passed through to :meth:`apply`.

        Returns
        -------
        AllGpuMIGConfigs
            The configurations that were (or would be) applied.
        """
        configs = self.pick_fixed_combinations(state_id)
        self.apply(configs, dry_run=dry_run)
        return configs
