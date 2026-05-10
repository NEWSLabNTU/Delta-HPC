"""
src/deploy/system.py

Global live state of the deployed cluster.

Both ``cluster.py`` and ``vllm.py`` write to ``SYSTEM_STATE`` as they complete
their respective operations:

* :class:`~src.deploy.cluster.DeployGPUSetup` populates GPU topology and MIG
  UUIDs after :meth:`~src.deploy.cluster.DeployGPUSetup.apply` succeeds.
* :class:`~src.deploy.vllm.VLLMManager` fills in ``model_id``, ``port``, and
  ``container_name`` for each slot as containers start or stop.

Import the singleton directly::

    from src.deploy.system import SYSTEM_STATE
    from src.deploy.models import MIGSlotState, GPUState

Example
-------
::

    from src.deploy.system import SYSTEM_STATE, get_slot, update_slot

    slot = get_slot(gpu_idx=0, start_slice=0)
    update_slot(gpu_idx=0, start_slice=0, port=8100, model_id="qwen2.5-7b-instruct")
"""

from __future__ import annotations

import logging
from typing import Optional

from src.deploy.models import ClusterState, GPUState, MIGSlotState

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global singleton
# ---------------------------------------------------------------------------

#: Live state of the entire cluster.  Mutated by cluster.py and vllm.py.
SYSTEM_STATE: ClusterState = ClusterState()


# ---------------------------------------------------------------------------
# Accessors
# ---------------------------------------------------------------------------


def get_gpu(gpu_idx: int) -> Optional[GPUState]:
    """Return the :class:`GPUState` for *gpu_idx*, or ``None`` if not yet set."""
    return SYSTEM_STATE.gpus.get(gpu_idx)


def get_slot(gpu_idx: int, start_slice: int) -> Optional[MIGSlotState]:
    """Return the :class:`MIGSlotState` at (*gpu_idx*, *start_slice*), or ``None``."""
    gpu = SYSTEM_STATE.gpus.get(gpu_idx)
    if gpu is None:
        return None
    for slot in gpu.slots:
        if slot.profile_placement.start_slice == start_slice:
            return slot
    return None


# ---------------------------------------------------------------------------
# Mutators
# ---------------------------------------------------------------------------


def register_gpu(gpu_state: GPUState) -> None:
    """Replace (or insert) the :class:`GPUState` entry for *gpu_state.gpu_idx*.

    Called by :meth:`~src.deploy.cluster.DeployGPUSetup.apply` after MIG
    reconfiguration completes and UUIDs have been resolved.
    """
    SYSTEM_STATE.gpus[gpu_state.gpu_idx] = gpu_state
    logger.info(
        "system: registered GPU %d (%s) with %d slot(s).",
        gpu_state.gpu_idx,
        gpu_state.model_name,
        len(gpu_state.slots),
    )


def update_slot(gpu_idx: int, start_slice: int, **kwargs: object) -> None:
    """Update fields on the slot at (*gpu_idx*, *start_slice*).

    Parameters
    ----------
    gpu_idx:
        Physical GPU index.
    start_slice:
        Starting memory-slice index of the target slot.
    **kwargs:
        Field names and new values to set on the :class:`MIGSlotState`.
        Commonly used fields: ``model_id``, ``port``, ``container_name``.

    Raises
    ------
    KeyError
        If no slot exists at the given coordinates.
    """
    slot = get_slot(gpu_idx, start_slice)
    if slot is None:
        raise KeyError(
            f"No MIG slot registered at GPU {gpu_idx} start_slice={start_slice}.  "
            "Call cluster.apply() before vllm.start()."
        )
    for key, value in kwargs.items():
        if not hasattr(slot, key):
            raise AttributeError(f"MIGSlotState has no field {key!r}.")
        setattr(slot, key, value)
    logger.debug(
        "system: GPU %d start_slice=%d updated: %s",
        gpu_idx,
        start_slice,
        kwargs,
    )
