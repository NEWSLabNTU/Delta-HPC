"""
src/deploy/mig_controller.py

Low-level MIG management via pynvml.

This module exposes a ``MIGController`` class that wraps every nvidia-smi
MIG operation (enable/disable MIG mode, destroy instances, create instances
with explicit placement) through the pynvml Python bindings.

Usage example
-------------
    from src.deploy.mig_controller import MIGController

    ctrl = MIGController(gpu_indices=[0, 1])
    with ctrl:
        ctrl.disable_all_instances(0)
        ctrl.create_gi_with_placement(
            gpu_idx=0,
            profile_placements=[
                ("4g.20gb", 0),   # 4G slice starting at position 0
                ("3g.20gb", 4),   # 3G slice starting at position 4
            ],
        )
        print(ctrl.list_gis(0))
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple
import pynvml

from src.deploy.models import GpuInstanceInfo, ProfilePlacement  # noqa: F401
from src.share.models import MIGProfileBase
from src.deploy.system import SYSTEM_STATE


logger = logging.getLogger(__name__)

# Upper bound for MIG GPU-instance profile IDs as defined by the NVML enum
# ``nvmlGpuInstanceProfile_t``.  Older pynvml builds may not expose the
# constant, so we fall back to 16 which exceeds the current maximum (0xC = 12).
_MAX_PROFILE_ID: int = getattr(pynvml, "NVML_GPU_INSTANCE_PROFILE_COUNT", 16)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _profile_string(size: int, vram: int) -> str:
    """Construct the canonical ``"{size}g.{vram}gb"`` key used across the repo."""
    return f"{size}g.{vram}gb"


def _build_profile_lookup(
    handle: pynvml.c_nvmlDevice_t,  # type: ignore[valid-type]
) -> Dict[str, Tuple[pynvml.c_nvmlGpuInstanceProfileInfo_t, int]]:  # type: ignore[valid-type]
    """Return a mapping ``profile_string → (ProfileInfo, profile_id)``.

    Iterates over all NVML-defined profile IDs (``0 … NVML_GPU_INSTANCE_PROFILE_COUNT-1``)
    and collects the ones that are actually supported by *handle*.
    """
    lookup: Dict[str, Tuple[pynvml.c_nvmlGpuInstanceProfileInfo_t, int]] = {}  # type: ignore[valid-type]
    for profile_id in range(_MAX_PROFILE_ID):
        try:
            info = pynvml.nvmlDeviceGetGpuInstanceProfileInfo(handle, profile_id)
        except pynvml.NVMLError:
            continue  # profile_id not supported on this GPU
        key = _profile_string(info.sliceCount, info.memorySizeMB // 1024)  # type: ignore[arg-type]
        lookup[key] = (info, profile_id)  # type: ignore
        logger.debug(
            "GPU profile discovered: id=%d  key=%s  multiprocessorCount=%d",
            profile_id,
            key,
            info.multiprocessorCount,  # type: ignore
        )
    return lookup


# ---------------------------------------------------------------------------
# MIGController
# ---------------------------------------------------------------------------


class MIGController:
    """Wrapper around pynvml MIG management functions.

    Parameters
    ----------
    gpu_indices:
        List of physical GPU indices to manage (matches ``nvidia-smi`` index).

    Examples
    --------
    Use as a context manager so that NVML is always properly shut down::

        with MIGController(gpu_indices=[0, 1]) as ctrl:
            ctrl.create_gi_with_placement(
                gpu_idx=0,
                profile_placements=[ProfilePlacement("4g.20gb", 0), ProfilePlacement("3g.20gb", 4)],
            )
    """

    def __init__(self, gpu_indices: List[int]) -> None:
        self._gpu_indices = list(gpu_indices)
        self._initialized = False

    # ------------------------------------------------------------------
    # Context manager / lifecycle
    # ------------------------------------------------------------------

    def __enter__(self) -> "MIGController":
        self.init()
        return self

    def __exit__(self, *_) -> None:
        self.shutdown()

    def init(self) -> None:
        """Initialize the NVML library.  Must be called before any other method
        (unless using the context manager interface)."""
        if not self._initialized:
            pynvml.nvmlInit()
            self._initialized = True
            logger.info("NVML initialized.")

    def shutdown(self) -> None:
        """Release the NVML library handle."""
        if self._initialized:
            pynvml.nvmlShutdown()
            self._initialized = False
            logger.info("NVML shut down.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _require_init(self) -> None:
        if not self._initialized:
            raise RuntimeError(
                "MIGController has not been initialized.  "
                "Call .init() or use it as a context manager."
            )

    def _handle(self, gpu_idx: int) -> pynvml.c_nvmlDevice_t:  # type: ignore[valid-type]
        return pynvml.nvmlDeviceGetHandleByIndex(gpu_idx)

    def _profile_lookup(
        self, gpu_idx: int
    ) -> Dict[str, Tuple[pynvml.c_nvmlGpuInstanceProfileInfo_t, int]]:  # type: ignore[valid-type]
        return _build_profile_lookup(self._handle(gpu_idx))

    # ------------------------------------------------------------------
    # MIG mode
    # ------------------------------------------------------------------

    def get_mig_mode(self, gpu_idx: int) -> bool:
        """Return ``True`` if MIG mode is currently *enabled* on *gpu_idx*."""
        self._require_init()
        handle = self._handle(gpu_idx)
        current_mode, _ = pynvml.nvmlDeviceGetMigMode(handle)
        return current_mode == pynvml.NVML_DEVICE_MIG_ENABLE

    def set_mig_mode(self, gpu_idx: int, *, enable: bool) -> None:
        """Enable or disable MIG mode on *gpu_idx*.

        .. warning::
            Changing MIG mode requires a GPU reset / reboot to take effect on
            some driver versions.  This method calls the NVML setter but does
            **not** trigger a reset.

        Parameters
        ----------
        gpu_idx:
            Physical GPU index.
        enable:
            ``True`` to enable MIG mode, ``False`` to disable.
        """
        self._require_init()
        handle = self._handle(gpu_idx)
        mode = (
            pynvml.NVML_DEVICE_MIG_ENABLE if enable else pynvml.NVML_DEVICE_MIG_DISABLE
        )
        pynvml.nvmlDeviceSetMigMode(handle, mode)
        logger.info(
            "GPU %d: MIG mode set to %s.", gpu_idx, "ENABLED" if enable else "DISABLED"
        )

    # ------------------------------------------------------------------
    # Querying existing instances
    # ------------------------------------------------------------------

    def list_gis(self, gpu_idx: int) -> List[GpuInstanceInfo]:
        """Return a list of all GPU instances currently configured on *gpu_idx*.

        Parameters
        ----------
        gpu_idx:
            Physical GPU index.

        Returns
        -------
        list of :class:`GpuInstanceInfo`
            One entry per existing GPU instance, sorted by starting slice.
        """
        self._require_init()
        handle = self._handle(gpu_idx)
        profile_lookup = self._profile_lookup(gpu_idx)

        gpu = SYSTEM_STATE.gpus.get(gpu_idx)
        if gpu is None:
            raise ValueError(f"GPU {gpu_idx} not found in SYSTEM_STATE")
        mig_profile_cls = gpu.mig_profile_cls

        infos: List[GpuInstanceInfo] = []
        # Iterate over all supported profiles and collect existing instances.
        for prof_str, (prof_info, _) in profile_lookup.items():
            try:
                instances = pynvml.nvmlDeviceGetGpuInstances(handle, prof_info.id)  # type: ignore
            except pynvml.NVMLError:
                continue
            for inst in instances:
                inst_info = pynvml.nvmlGpuInstanceGetInfo(inst)  # type: ignore
                placement = inst_info.placement  # type: ignore
                infos.append(
                    GpuInstanceInfo(
                        instance_id=inst_info.id,
                        profile=mig_profile_cls.from_string(prof_str),
                        start_slice=placement.start,
                        slice_count=placement.size,
                    )
                )

        infos.sort(key=lambda x: x.start_slice)
        return infos

    def get_gi_handles(self, gpu_idx: int) -> List[pynvml.c_nvmlGpuInstance_t]:  # type: ignore[valid-type]
        """Return raw pynvml GPU-instance handles for *gpu_idx* (all profiles)."""
        self._require_init()
        handle = self._handle(gpu_idx)
        profile_lookup = self._profile_lookup(gpu_idx)

        handles: List[pynvml.c_nvmlGpuInstance_t] = []  # type: ignore[valid-type]
        for _, (prof_info, _) in profile_lookup.items():
            try:
                handles.extend(pynvml.nvmlDeviceGetGpuInstances(handle, prof_info.id))  # type: ignore
            except pynvml.NVMLError:
                continue
        return handles

    # ------------------------------------------------------------------
    # Destroying instances
    # ------------------------------------------------------------------

    def destroy_gi(
        self,
        gpu_idx: int,
        instance_handle: pynvml.c_nvmlGpuInstance_t,  # type: ignore[valid-type]
    ) -> None:
        """Destroy a single GPU instance identified by its NVML handle.

        All compute instances inside the GPU instance must be destroyed first.
        This method destroys them automatically before destroying the GPU
        instance itself.

        Parameters
        ----------
        gpu_idx:
            Physical GPU index (used only for logging).
        instance_handle:
            Raw pynvml GPU-instance handle, as returned by
            :meth:`get_gi_handles`.
        """
        self._require_init()
        # Destroy every compute instance inside first.
        inst_info = pynvml.nvmlGpuInstanceGetInfo(instance_handle)
        try:
            ci_profiles = pynvml.nvmlGpuInstanceGetComputeInstanceProfileInfo(  # type: ignore
                instance_handle, 0, pynvml.NVML_COMPUTE_INSTANCE_ENGINE_PROFILE_SHARED
            )
            compute_instances = pynvml.nvmlGpuInstanceGetComputeInstances(  # type: ignore
                instance_handle, ci_profiles.id
            )
            for ci in compute_instances:
                pynvml.nvmlComputeInstanceDestroy(ci)
                logger.debug("GPU %d: destroyed compute instance.", gpu_idx)
        except pynvml.NVMLError as exc:
            logger.debug(
                "GPU %d: skipping compute instance teardown (%s).", gpu_idx, exc
            )

        pynvml.nvmlGpuInstanceDestroy(instance_handle)
        logger.info(
            "GPU %d: destroyed GPU instance id=%d profile=%s start=%d.",
            gpu_idx,
            inst_info.id,
            inst_info.profileId,
            inst_info.placement.start,
        )

    def disable_all_instances(self, gpu_idx: int) -> None:
        """Destroy **all** GPU instances on *gpu_idx* (full reset of MIG state).

        Parameters
        ----------
        gpu_idx:
            Physical GPU index.
        """
        self._require_init()
        handles = self.get_gi_handles(gpu_idx)
        if not handles:
            logger.info("GPU %d: no existing GPU instances to destroy.", gpu_idx)
            return
        for inst_handle in handles:
            self.destroy_gi(gpu_idx, inst_handle)
        logger.info("GPU %d: all GPU instances destroyed.", gpu_idx)

    def disable_all_instances_all_gpus(self) -> None:
        """Destroy all GPU instances on every GPU in :attr:`gpu_indices`."""
        for gpu_idx in self._gpu_indices:
            self.disable_all_instances(gpu_idx)

    # ------------------------------------------------------------------
    # Creating instances with explicit placement
    # ------------------------------------------------------------------

    def create_gi(
        self,
        gpu_idx: int,
        profile: MIGProfileBase,
        start_slice: int,
    ) -> pynvml.c_nvmlGpuInstance_t:  # type: ignore[valid-type]
        """Create a single GPU instance at an explicit slice placement.

        Parameters
        ----------
        gpu_idx:
            Physical GPU index.
        profile:
            Hardware profile enum member (e.g. from ``MIGProfileA100``).
        start_slice:
            The memory-slice index at which the instance should begin.

        Returns
        -------
        pynvml.c_nvmlGpuInstance_t
            Handle to the newly created GPU instance.

        Raises
        ------
        ValueError
            If *profile_string* is not supported on *gpu_idx*.
        pynvml.NVMLError
            If NVML rejects the placement (e.g. overlap, insufficient room).
        """
        self._require_init()
        handle = self._handle(gpu_idx)
        profile_lookup = self._profile_lookup(gpu_idx)

        if profile.string not in profile_lookup:
            available = sorted(profile_lookup.keys())
            raise ValueError(
                f"Profile '{profile.string}' not supported on GPU {gpu_idx}.  "
                f"Available profiles: {available}"
            )

        prof_info, _ = profile_lookup[profile.string]

        # Build the placement struct.
        placement = pynvml.c_nvmlGpuInstancePlacement_t()
        placement.start = start_slice
        placement.size = prof_info.sliceCount

        instance = pynvml.nvmlDeviceCreateGpuInstanceWithPlacement(  # type: ignore
            handle, prof_info.id, placement
        )
        logger.info(
            "GPU %d: created GPU instance  profile=%s  start_slice=%d  slice_count=%d.",
            gpu_idx,
            profile.string,
            start_slice,
            prof_info.sliceCount,
        )
        return instance

    def create_gi_with_placement(
        self,
        gpu_idx: int,
        profile_placements: List[ProfilePlacement],
    ) -> List[pynvml.c_nvmlGpuInstance_t]:  # type: ignore[valid-type]
        """Create multiple GPU instances on *gpu_idx* at the given placements.

        The instances are created in the order supplied.  If any creation
        fails the error propagates immediately (already-created instances
        are **not** automatically cleaned up — call
        :meth:`disable_all_instances` if you need a clean state).

        Parameters
        ----------
        gpu_idx:
            Physical GPU index.
        profile_placements:
            Ordered sequence of :class:`~src.deploy.models.ProfilePlacement`
            objects.  Example::

                [ProfilePlacement("4g.20gb", 0), ProfilePlacement("3g.20gb", 4)]

        Returns
        -------
        list of pynvml.c_nvmlGpuInstance_t
            One handle per created instance, in the same order as
            *profile_placements*.
        """
        self._require_init()
        handles: List[pynvml.c_nvmlGpuInstance_t] = []  # type: ignore[valid-type]
        for p in profile_placements:
            h = self.create_gi(gpu_idx, p.profile, p.start_slice)
            handles.append(h)
        return handles

    def apply_full_configuration(
        self,
        gpu_idx: int,
        profile_placements: List[ProfilePlacement],
    ) -> List[pynvml.c_nvmlGpuInstance_t]:  # type: ignore[valid-type]
        """Wipe all existing instances on *gpu_idx* then apply *profile_placements*.

        This is the primary high-level API for switching a GPU to a completely
        new MIG configuration in one call.

        Parameters
        ----------
        gpu_idx:
            Physical GPU index.
        profile_placements:
            Ordered sequence of :class:`~src.deploy.models.ProfilePlacement`
            objects representing the desired MIG configuration.

        Returns
        -------
        list of pynvml.c_nvmlGpuInstance_t
            Handles for the newly created instances.
        """
        self._require_init()
        logger.info(
            "GPU %d: applying new MIG configuration: %s", gpu_idx, profile_placements
        )
        self.disable_all_instances(gpu_idx)
        return self.create_gi_with_placement(gpu_idx, profile_placements)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def gpu_indices(self) -> List[int]:
        """The physical GPU indices managed by this controller."""
        return list(self._gpu_indices)

    # ------------------------------------------------------------------
    # MIG device UUID resolution
    # ------------------------------------------------------------------

    def list_mig_device_uuids(self, gpu_idx: int) -> List[Tuple[int, str]]:
        """Return ``[(start_slice, mig_uuid), …]`` for each active MIG device on *gpu_idx*.

        The UUID (e.g. ``"MIG-GPU-xxxx.../1/0"``) is the string needed by
        docker-compose's ``device_ids`` field to pin a container to a specific
        MIG compute instance.

        The method assumes **one compute instance per GPU instance**, which is
        the standard single-process setup created by
        :meth:`apply_full_configuration`.  Entries are sorted by
        *start_slice* (lowest first), matching the order returned by
        :meth:`list_gis`.

        Parameters
        ----------
        gpu_idx:
            Physical GPU index.

        Returns
        -------
        list of (int, str)
            ``(start_slice, uuid)`` pairs, sorted by *start_slice*.
        """
        self._require_init()
        handle = self._handle(gpu_idx)

        # GPU instances sorted by start_slice — our placement reference.
        gi_infos = self.list_gis(gpu_idx)

        # MIG devices are also ordered by placement (ascending start_slice).
        try:
            max_count = pynvml.nvmlDeviceGetMaxMigDeviceCount(handle)  # type: ignore
        except pynvml.NVMLError as exc:
            logger.warning("GPU %d: cannot query MIG device count: %s", gpu_idx, exc)
            return []

        mig_uuids: List[str] = []
        for i in range(max_count):
            try:
                mig_handle = pynvml.nvmlDeviceGetMigDeviceHandleByIndex(handle, i)  # type: ignore
                uuid: str = pynvml.nvmlDeviceGetUUID(mig_handle)  # type: ignore
                mig_uuids.append(uuid)
            except pynvml.NVMLError:
                pass  # No MIG device at index i

        if len(mig_uuids) != len(gi_infos):
            logger.warning(
                "GPU %d: %d GPU instances but %d MIG device UUIDs — "
                "UUID list will be truncated to the shorter of the two.",
                gpu_idx,
                len(gi_infos),
                len(mig_uuids),
            )

        result: List[Tuple[int, str]] = [
            (gi.start_slice, uuid) for gi, uuid in zip(gi_infos, mig_uuids)
        ]
        return result
