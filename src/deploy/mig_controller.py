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
import time
from typing import Dict, List, Tuple
import ctypes
import pynvml

from src.deploy.models import GpuInstanceInfo, ProfilePlacement  # noqa: F401
from src.share.models import MIGProfileBase
from src.deploy.system import SYSTEM_STATE


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

# Upper bound for MIG GPU-instance profile IDs as defined by the NVML enum
# ``nvmlGpuInstanceProfile_t``.  Older pynvml builds may not expose the
# constant, so we fall back to 16 which exceeds the current maximum (0xC = 12).
_MAX_PROFILE_ID: int = getattr(pynvml, "NVML_GPU_INSTANCE_PROFILE_COUNT", 16)


def _profile_string(size: int, vram: int) -> str:
    """Construct the canonical ``"{size}g.{vram}gb"`` key used across the repo."""
    return f"{size}g.{vram}gb"


def _safe_nvml_device_get_gpu_instances(handle, profile_id):
    """Retrieve GPU instances while handling raw ctypes signatures."""
    # Some pynvml/driver versions return InvalidArgument if instances is None.
    # We use a fixed-size array (32 is plenty for any current GPU) to get instances.
    count = ctypes.c_uint(32)
    array_type = pynvml.c_nvmlGpuInstance_t * 32
    instances = array_type()
    try:
        pynvml.nvmlDeviceGetGpuInstances(
            handle, profile_id, instances, ctypes.byref(count)
        )
    except pynvml.NVMLError:
        return []
    return [instances[i] for i in range(count.value)]


def _safe_nvml_gpu_instance_get_compute_instances(handle, profile_id):
    """Retrieve compute instances while handling raw ctypes signatures."""
    count = ctypes.c_uint(32)
    array_type = pynvml.c_nvmlComputeInstance_t * 32
    instances = array_type()
    try:
        pynvml.nvmlGpuInstanceGetComputeInstances(
            handle, profile_id, instances, ctypes.byref(count)
        )
    except pynvml.NVMLError:
        return []
    return [instances[i] for i in range(count.value)]


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
        # Ceiling division for VRAM to match config naming (e.g. 40320 MiB -> 40 GB)
        vram_gb = (info.memorySizeMB + 1023) // 1024
        key = _profile_string(info.sliceCount, vram_gb)  # type: ignore[arg-type]
        lookup[key] = (info, profile_id)  # type: ignore
        logger.debug(
            "GPU profile discovered: index=%d  id=%d  key=%s  multiprocessorCount=%d",
            profile_id,
            info.id,
            key,
            info.multiprocessorCount,
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

        gpu = SYSTEM_STATE.gpus.get(gpu_idx)
        if gpu is None:
            raise ValueError(f"GPU {gpu_idx} not found in SYSTEM_STATE")
        mig_profile_cls = gpu.mig_profile_cls

        infos: List[GpuInstanceInfo] = []
        # Get all possible profiles to ensure we find all instances
        seen_ids = set()
        for i in range(32):
            try:
                prof_info = pynvml.nvmlDeviceGetGpuInstanceProfileInfo(handle, i)
                if prof_info.id in seen_ids:
                    continue
                seen_ids.add(prof_info.id)

                instances = _safe_nvml_device_get_gpu_instances(handle, prof_info.id)
                for inst in instances:
                    inst_info = pynvml.nvmlGpuInstanceGetInfo(inst)
                    placement = inst_info.placement

                    # Heuristic to find a matching profile name
                    vram_gb = (prof_info.memorySizeMB + 1023) // 1024
                    prof_str = _profile_string(prof_info.sliceCount, vram_gb)

                    infos.append(
                        GpuInstanceInfo(
                            instance_id=inst_info.id,
                            profile=mig_profile_cls.from_string(prof_str),
                            start_slice=placement.start,
                            slice_count=placement.size,
                        )
                    )
            except (pynvml.NVMLError, ValueError) as exc:
                if isinstance(exc, pynvml.NVMLError_NoPermission):
                    logger.error(
                        "GPU %d: Insufficient permissions to list GPU instances. Try running with sudo.",
                        gpu_idx,
                    )
                    raise
                continue

        infos.sort(key=lambda x: x.start_slice)
        return infos

    def get_gi_handles(self, gpu_idx: int) -> List[pynvml.c_nvmlGpuInstance_t]:  # type: ignore[valid-type]
        """Return raw pynvml GPU-instance handles for *gpu_idx* (all profiles)."""
        self._require_init()
        handle = self._handle(gpu_idx)

        handles: List[pynvml.c_nvmlGpuInstance_t] = []  # type: ignore[valid-type]
        seen_ids = set()
        for i in range(32):
            try:
                prof_info = pynvml.nvmlDeviceGetGpuInstanceProfileInfo(handle, i)
                if prof_info.id not in seen_ids:
                    handles.extend(
                        _safe_nvml_device_get_gpu_instances(handle, prof_info.id)
                    )
                    seen_ids.add(prof_info.id)
            except pynvml.NVMLError as exc:
                if isinstance(exc, pynvml.NVMLError_NoPermission):
                    raise
                continue
        return handles

    def get_gi_handles_map(self, gpu_idx: int) -> Dict[int, pynvml.c_nvmlGpuInstance_t]:  # type: ignore[valid-type]
        """Return a map of {start_slice: handle} for all existing GPU instances on *gpu_idx*."""
        self._require_init()
        handle = self._handle(gpu_idx)

        res: Dict[int, pynvml.c_nvmlGpuInstance_t] = {}
        for i in range(32):
            try:
                prof_info = pynvml.nvmlDeviceGetGpuInstanceProfileInfo(handle, i)
                instances = _safe_nvml_device_get_gpu_instances(handle, prof_info.id)
                for inst in instances:
                    inst_info = pynvml.nvmlGpuInstanceGetInfo(inst)
                    res[inst_info.placement.start] = inst
            except pynvml.NVMLError:
                continue
        return res

    # ------------------------------------------------------------------
    # Destroying instances
    # ------------------------------------------------------------------

    def destroy_gis_at_slices(self, gpu_idx: int, start_slices: List[int]) -> None:
        """Destroy specific GPU instances on *gpu_idx* identified by their starting slices."""
        self._require_init()
        gi_map = self.get_gi_handles_map(gpu_idx)
        for s in start_slices:
            if s in gi_map:
                logger.info(f"GPU {gpu_idx}: destroying instance at slice {s}")
                self.destroy_gi(gpu_idx, gi_map[s])
            else:
                logger.warning(
                    f"GPU {gpu_idx}: no instance found at slice {s} to destroy"
                )

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
        inst_info = pynvml.nvmlGpuInstanceGetInfo(instance_handle)

        # Some driver/OS versions take a moment to release the device after the
        # container stops. We retry the entire GI teardown (both CI and GI).
        for attempt in range(5):
            try:
                # 1. Destroy compute instances (CI)
                try:
                    ci_profiles = pynvml.nvmlGpuInstanceGetComputeInstanceProfileInfo(  # type: ignore
                        instance_handle,
                        0,
                        pynvml.NVML_COMPUTE_INSTANCE_ENGINE_PROFILE_SHARED,
                    )
                    compute_instances = _safe_nvml_gpu_instance_get_compute_instances(
                        instance_handle, ci_profiles.id
                    )
                    for ci in compute_instances:
                        pynvml.nvmlComputeInstanceDestroy(ci)
                        logger.debug("GPU %d: destroyed compute instance.", gpu_idx)
                except pynvml.NVMLError as exc:
                    logger.debug(
                        "GPU %d: compute instance teardown skipped/failed (%s).",
                        gpu_idx,
                        exc,
                    )

                # 2. Destroy GPU instance (GI)
                pynvml.nvmlGpuInstanceDestroy(instance_handle)
                break

            except pynvml.NVMLError as exc:
                if isinstance(exc, pynvml.NVMLError_InUse) and attempt < 4:
                    logger.debug(
                        "GPU %d: instance id=%d still in use, retrying in 1.5s...",
                        gpu_idx,
                        inst_info.id,
                    )
                    time.sleep(1.5)
                    continue
                raise

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

        # Determine correct physical placement size from hardware.
        # Profiles like 7G and 3G on A100 have physical footprints (8 and 4)
        # larger than their logical slice count (7 and 3).
        placement_size = self._get_placement_size(handle, prof_info.id, start_slice)
        if placement_size != prof_info.sliceCount:
            logger.debug(
                "GPU %d: profile %s at slice %d requires physical size %d (logical slices %d)",
                gpu_idx,
                profile.string,
                start_slice,
                placement_size,
                prof_info.sliceCount,
            )

        # Build the placement struct.
        placement = pynvml.c_nvmlGpuInstancePlacement_t()
        placement.start = start_slice
        placement.size = placement_size

        logger.debug(
            "GPU %d: calling nvmlDeviceCreateGpuInstanceWithPlacement(profile_id=%d, start=%d, size=%d)",
            gpu_idx,
            prof_info.id,
            placement.start,
            placement.size,
        )

        instance = pynvml.nvmlDeviceCreateGpuInstanceWithPlacement(  # type: ignore
            handle, prof_info.id, ctypes.byref(placement)
        )
        logger.info(
            "GPU %d: created GPU instance  profile=%s  start_slice=%d  phys_size=%d  log_slices=%d.",
            gpu_idx,
            profile.string,
            start_slice,
            placement_size,
            prof_info.sliceCount,
        )

        # Create a default compute instance for this GPU instance.
        # This is required for the OS to expose a MIG device handle/UUID.
        try:
            # profileId 0 is the default "full" compute profile for the GI
            ci_prof = pynvml.nvmlGpuInstanceGetComputeInstanceProfileInfo(
                instance, 0, pynvml.NVML_COMPUTE_INSTANCE_ENGINE_PROFILE_SHARED
            )
            pynvml.nvmlGpuInstanceCreateComputeInstance(instance, ci_prof.id)
            logger.info("GPU %d: created default compute instance.", gpu_idx)
        except pynvml.NVMLError as exc:
            logger.warning(
                "GPU %d: failed to create compute instance: %s", gpu_idx, exc
            )

        return instance

    def _get_placement_size(
        self, handle: pynvml.c_nvmlDevice_t, profile_id: int, start_slice: int
    ) -> int:  # type: ignore[valid-type]
        """Query NVML for the valid physical placement size at the given start slice."""
        try:
            count = ctypes.c_uint(32)
            placements = (pynvml.c_nvmlGpuInstancePlacement_t * 32)()
            pynvml.nvmlDeviceGetGpuInstancePossiblePlacements(
                handle, profile_id, placements, ctypes.byref(count)
            )
            for i in range(count.value):
                p = placements[i]
                if p.start == start_slice:
                    return p.size
        except pynvml.NVMLError:
            pass

        # Fallback to logical slice count if query fails
        # We need the profile info to get the slice count.
        # Since we have the hardware ID, we should use nvmlDeviceGetGpuInstanceProfileInfoById.
        try:
            info = pynvml.nvmlDeviceGetGpuInstanceProfileInfoById(handle, profile_id)
            return info.sliceCount
        except pynvml.NVMLError:
            return 1  # absolute fallback

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

        This method uses explicit hardware mapping via ``nvmlDeviceGetGpuInstanceId``
        to ensure that each UUID is matched to the correct physical placement
        (start_slice), avoiding errors caused by non-deterministic discovery order.

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

        # 1. Get current GPU instances and their IDs
        gi_infos = self.list_gis(gpu_idx)
        gi_id_to_slice = {info.instance_id: info.start_slice for info in gi_infos}

        if not gi_infos:
            return []

        try:
            max_count = pynvml.nvmlDeviceGetMaxMigDeviceCount(handle)  # type: ignore
        except pynvml.NVMLError as exc:
            logger.warning("GPU %d: cannot query MIG device count: %s", gpu_idx, exc)
            return []

        # 2. Resolve MIG UUIDs with explicit GI mapping.
        res: List[Tuple[int, str]] = []
        res = []
        for i in range(max_count):
            try:
                mig_handle = pynvml.nvmlDeviceGetMigDeviceHandleByIndex(handle, i)  # type: ignore
                uuid: str = pynvml.nvmlDeviceGetUUID(mig_handle)  # type: ignore

                # Explicitly get the GPU Instance ID for this MIG device handle
                gi_id = pynvml.nvmlDeviceGetGpuInstanceId(mig_handle)  # type: ignore

                if gi_id in gi_id_to_slice:
                    res.append((gi_id_to_slice[gi_id], uuid))
            except pynvml.NVMLError:
                continue

        if len(res) != len(gi_infos):
            logger.warning(
                "GPU %d: Expected %d MIG UUIDs but only resolved %d. "
                "The system state may be inconsistent.",
                gpu_idx,
                len(gi_infos),
                len(res),
            )

        return sorted(res, key=lambda x: x[0])
