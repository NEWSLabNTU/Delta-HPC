"""
src/deploy/mig_controller.py

MIG management via nvidia-smi CLI.

This module exposes a ``MIGController`` class that wraps every nvidia-smi
MIG operation (enable/disable MIG mode, destroy instances, create instances
with explicit placement) cleanly using standard python subprocess execution.

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

import re
import logging
import subprocess
import time
import glob
from pathlib import Path
import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple, Any

from src.deploy.system import SYSTEM_STATE
from src.deploy.models import GpuInstanceInfo, ProfilePlacement
from src.share.models import MIGProfile, MIGProfileBase
from src.share.hardware import (
    DetectedGPU,
    match_gpu_model,
    load_mig_profile_class,
    derive_valid_combinations,
)
from src.share.mig_matrix import STATE_DEFINITIONS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MIGController
# ---------------------------------------------------------------------------


class MIGController:
    """Wrapper around nvidia-smi MIG management functions.

    Parameters
    ----------
    gpu_indices:
        List of physical GPU indices to manage (matches ``nvidia-smi`` index).

    Examples
    --------
    Instantiate and call directly::

        ctrl = MIGController(gpu_indices=[0, 1])
        ctrl.create_gi_with_placement(
            gpu_idx=0,
            profile_placements=[ProfilePlacement("4g.20gb", 0), ProfilePlacement("3g.20gb", 4)],
        )
    """

    def __init__(self, gpu_indices: List[int]) -> None:
        self._gpu_indices = list(gpu_indices)

    @classmethod
    def _run_nvidia_smi(cls, args: List[str]) -> subprocess.CompletedProcess:
        """Run nvidia-smi command with specified arguments using sudo, ignoring infoROM exit code 127."""
        cmd = ["sudo", "nvidia-smi"] + args
        logger.info("Executing: %s", " ".join(cmd))
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        stderr_str = res.stderr.decode().strip()
        stdout_str = res.stdout.decode().strip()

        if stderr_str:
            logger.warning("nvidia-smi stderr: %s", stderr_str)

        if res.returncode not in (0, 127):
            logger.error(
                "nvidia-smi command failed with exit code %d.\nCommand: %s\nstdout: %s\nstderr: %s",
                res.returncode,
                " ".join(cmd),
                stdout_str,
                stderr_str,
            )
            raise subprocess.CalledProcessError(
                returncode=res.returncode, cmd=cmd, output=res.stdout, stderr=res.stderr
            )
        return res

    @classmethod
    def get_active_gpu_processes(cls) -> List[Dict[str, Any]]:
        """Query nvidia-smi for all active compute processes across all GPUs and MIGs."""
        try:
            res = cls._run_nvidia_smi(
                [
                    "--query-compute-apps=pid,process_name,gpu_uuid",
                    "--format=csv,noheader",
                ]
            )
            out = res.stdout.decode().strip()
            processes = []
            for line in out.split("\n"):
                if not line.strip():
                    continue
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 3:
                    processes.append(
                        {
                            "pid": int(parts[0]),
                            "name": parts[1],
                            "gpu_uuid": parts[2],
                        }
                    )
            return processes
        except Exception as e:
            logger.warning(f"Failed to query active GPU processes: {e}")
            return []

    @classmethod
    def get_nvidia_smi_overall_status(cls) -> str:
        """Run standalone 'nvidia-smi' and return its stdout."""
        try:
            res = cls._run_nvidia_smi([])
            return res.stdout.decode().strip()
        except Exception as e:
            return f"Failed to run nvidia-smi: {e}"

    @classmethod
    def get_nvidia_smi_lgi_status(cls, gpu_idx: int) -> str:
        """Run 'nvidia-smi mig -lgi -i {gpu_idx}' and return its stdout."""
        try:
            res = cls._run_nvidia_smi(["mig", "-lgi", "-i", str(gpu_idx)])
            return res.stdout.decode().strip()
        except Exception as e:
            return f"Failed to run nvidia-smi mig -lgi: {e}"

    @classmethod
    def get_nvidia_fuser_output(cls) -> str:
        """Run 'sudo fuser -v /dev/nvidia*' and return its stdout/stderr."""
        try:
            device_paths = glob.glob("/dev/nvidia*")
            if not device_paths:
                return "No /dev/nvidia* device files found."
            cmd = ["sudo", "fuser", "-v"] + device_paths
            res = subprocess.run(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
            )
            return res.stdout.strip()
        except Exception as e:
            return f"Failed to run fuser: {e}"

    @classmethod
    def get_nvidia_lsof_output(cls) -> str:
        """Run 'sudo lsof /dev/nvidia*' and return its stdout/stderr."""
        try:
            device_paths = glob.glob("/dev/nvidia*")
            if not device_paths:
                return "No /dev/nvidia* device files found."
            cmd = ["sudo", "lsof"] + device_paths
            res = subprocess.run(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
            )
            return res.stdout.strip()
        except Exception as e:
            return f"Failed to run lsof: {e}"

    def query_gpu_instances(self, gpu_idx: int) -> List[Dict[str, Any]]:
        """Use nvidia-smi to query active GPU instances on gpu_idx."""
        try:
            res = self.__class__._run_nvidia_smi(["mig", "-lgi", "-i", str(gpu_idx)])
        except subprocess.CalledProcessError as exc:
            if exc.returncode == 6:
                return []
            raise exc
        lines = res.stdout.decode().splitlines()
        gpu_instances = []
        pattern = r"^\|\s*(\d+)\s+MIG\s+([^\s|]+)\s+(\d+)\s+(\d+)\s+[^|]*?\s+(\d+):(\d+)\s*\|$"
        for line in lines:
            m = re.match(pattern, line.strip())
            if m:
                gi_id = int(m.group(4))
                profile_str = m.group(2)
                start_slice = int(m.group(5))
                gpu_instances.append(
                    {
                        "gpu_idx": gpu_idx,
                        "id": gi_id,
                        "profile_str": profile_str,
                        "start_slice": start_slice,
                    }
                )
        return gpu_instances

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
        gpu = SYSTEM_STATE.gpus.get(gpu_idx)
        if gpu is None:
            raise ValueError(f"GPU {gpu_idx} not found in SYSTEM_STATE")
        mig_profile_cls = gpu.mig_profile_cls

        g_insts = self.query_gpu_instances(gpu_idx)
        infos: List[GpuInstanceInfo] = []
        for gi in g_insts:
            prof_str = gi["profile_str"]
            prof_enum = next((p for p in mig_profile_cls if p.string == prof_str), None)
            assert prof_enum is not None, (
                f"Profile {prof_str} not found in mig_profile_cls"
            )
            infos.append(
                GpuInstanceInfo(
                    instance_id=gi["id"],
                    profile=prof_enum,
                    start_slice=gi["start_slice"],
                    slice_count=prof_enum.size,
                )
            )

        infos.sort(key=lambda x: x.start_slice)
        return infos

    # ------------------------------------------------------------------
    # Destroying instances
    # ------------------------------------------------------------------

    def destroy_gis_at_slices(self, gpu_idx: int, start_slices: List[int]) -> None:
        """Destroy specific GPU instances on *gpu_idx* identified by their starting slices."""
        g_insts = self.query_gpu_instances(gpu_idx)
        for s in start_slices:
            # Find the instance at start_slice s
            gi = next((g for g in g_insts if g["start_slice"] == s), None)
            if gi is not None:
                logger.info(f"GPU {gpu_idx}: destroying instance at slice {s}")
                self.destroy_gi_by_id(gpu_idx, gi["id"])
            else:
                logger.warning(
                    f"GPU {gpu_idx}: no instance found at slice {s} to destroy"
                )

    def destroy_gi_by_id(self, gpu_idx: int, gi_id: int) -> None:
        """Destroy a single GPU instance identified by its ID.

        All compute instances inside the GPU instance must be destroyed first.
        This method destroys them automatically using the nvidia-smi CLI.
        """
        logger.info(
            "GPU %d: destroying GPU instance id=%d using nvidia-smi CLI",
            gpu_idx,
            gi_id,
        )

        # 1. Destroy compute instance ID 0 (ignore if already gone)
        cmd_dci = [
            "mig",
            "-dci",
            "-ci",
            "0",
            "-gi",
            str(gi_id),
            "-i",
            str(gpu_idx),
        ]
        try:
            self.__class__._run_nvidia_smi(cmd_dci)
            logger.info("GPU %d: successfully destroyed compute instance.", gpu_idx)
            time.sleep(3.0)  # give a brief pause before tearing down GI
        except subprocess.CalledProcessError as ci_exc:
            # Ignore "Not Found" error if it was already destroyed
            ci_err_msg = ci_exc.stderr.decode().strip() if ci_exc.stderr else ""
            if "Not Found" in ci_err_msg or "not found" in ci_err_msg.lower():
                logger.info(
                    "GPU %d: Compute instance on GI %d already gone (continuing to destroy GI)",
                    gpu_idx,
                    gi_id,
                )
            else:
                logger.warning(
                    "GPU %d: CI deletion on GI %d failed: %s (continuing to destroy GI)",
                    gpu_idx,
                    gi_id,
                    ci_err_msg,
                )

        # 2. Destroy GPU instance (GI)
        cmd_dgi = ["mig", "-dgi", "-gi", str(gi_id), "-i", str(gpu_idx)]

        max_attempts = 3
        attempt_delay = 5.0  # seconds
        for attempt in range(max_attempts):
            try:
                self.__class__._run_nvidia_smi(cmd_dgi)
                logger.info(
                    "GPU %d: destroyed GPU instance id=%d.",
                    gpu_idx,
                    gi_id,
                )
                break
            except subprocess.CalledProcessError as dgi_exc:
                dgi_err_msg = dgi_exc.stderr.decode().strip() if dgi_exc.stderr else ""
                # Check for "In use by another client"
                if (
                    "In use by another client" in dgi_err_msg
                    or dgi_exc.returncode == 19
                ):
                    if attempt < max_attempts - 1:
                        # Standalone overall nvidia-smi status
                        smi_overall = self.__class__.get_nvidia_smi_overall_status()
                        smi_overall_info = (
                            f"\n  nvidia-smi overall status:\n{smi_overall}"
                        )

                        # Standalone nvidia-smi mig -lgip status
                        smi_lgi = self.__class__.get_nvidia_smi_lgi_status(gpu_idx)
                        smi_lgi_info = f"\n  nvidia-smi mig -lgi status:\n{smi_lgi}"

                        # Query active processes to see who is using the GPU
                        active_procs = self.__class__.get_active_gpu_processes()
                        proc_info = ""
                        if active_procs:
                            proc_info = " Active processes: " + ", ".join(
                                f"{p['name']} (PID {p['pid']} on UUID {p['gpu_uuid']})"
                                for p in active_procs
                            )
                        else:
                            proc_info = (
                                " No active compute processes listed by nvidia-smi."
                            )

                        # Run fuser and lsof to get raw device usage
                        fuser_output = self.__class__.get_nvidia_fuser_output()
                        fuser_info = (
                            f"\n  fuser output:\n{fuser_output}"
                            if fuser_output
                            else "\n  fuser output: None"
                        )

                        lsof_output = self.__class__.get_nvidia_lsof_output()
                        lsof_info = (
                            f"\n  lsof output:\n{lsof_output}"
                            if lsof_output
                            else "\n  lsof output: None"
                        )

                        logger.warning(
                            f"GPU {gpu_idx}: GI id={gi_id} deletion failed because device is still in use by another client.{smi_overall_info}{smi_lgi_info}{proc_info}{fuser_info}{lsof_info}\n"
                            f"Retrying in {attempt_delay}s (attempt {attempt + 1}/{max_attempts})..."
                        )
                        time.sleep(attempt_delay)
                        continue
                raise dgi_exc

    def disable_all_instances(self, gpu_idx: int) -> None:
        """Destroy **all** GPU instances on *gpu_idx* (full reset of MIG state).

        Parameters
        ----------
        gpu_idx:
            Physical GPU index.
        """
        g_insts = self.query_gpu_instances(gpu_idx)
        if not g_insts:
            logger.info("GPU %d: no existing GPU instances to destroy.", gpu_idx)
            return
        for gi in g_insts:
            self.destroy_gi_by_id(gpu_idx, gi["id"])
        logger.info("GPU %d: all GPU instances destroyed.", gpu_idx)

    # ------------------------------------------------------------------
    # Creating instances with explicit placement
    # ------------------------------------------------------------------

    def create_gi(
        self,
        gpu_idx: int,
        profile: MIGProfileBase,
        start_slice: int,
    ) -> int:
        """Create a single GPU instance at an explicit slice placement.

        This method uses the nvidia-smi CLI to execute the reconfiguration
        reliably, verifies creation, and returns the newly formed GPU Instance ID.
        """
        logger.info(
            "GPU %d: creating GPU instance profile=%s start_slice=%d using nvidia-smi CLI",
            gpu_idx,
            profile.string,
            start_slice,
        )

        # Call nvidia-smi mig -cgi to create the GPU instance and -C for default compute instance
        cmd = [
            "mig",
            "-cgi",
            f"{profile.string}:{start_slice}",
            "-C",
            "-i",
            str(gpu_idx),
        ]
        self.__class__._run_nvidia_smi(cmd)

        # Verify creation via nvidia-smi query and find GI ID
        gi_infos = self.query_gpu_instances(gpu_idx)
        new_gi = next((gi for gi in gi_infos if gi["start_slice"] == start_slice), None)
        if new_gi is None:
            raise RuntimeError(
                f"GPU {gpu_idx}: failed to find newly created GPU instance at slice {start_slice}."
            )
        return new_gi["id"]

    def create_gi_with_placement(
        self,
        gpu_idx: int,
        profile_placements: List[ProfilePlacement],
    ) -> List[int]:
        """Create multiple GPU instances on *gpu_idx* at the given placements.

        Parameters
        ----------
        gpu_idx:
            Physical GPU index.
        profile_placements:
            Ordered sequence of :class:`~src.deploy.models.ProfilePlacement`
            objects.

        Returns
        -------
        list of int
            Newly created GPU instance IDs.
        """
        ids: List[int] = []
        for p in profile_placements:
            h = self.create_gi(gpu_idx, p.profile, p.start_slice)
            ids.append(h)
        return ids

    def apply_full_configuration(
        self,
        gpu_idx: int,
        profile_placements: List[ProfilePlacement],
    ) -> List[int]:
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
        list of int
            IDs of the newly created instances.
        """
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

        Parameters
        ----------
        gpu_idx:
            Physical GPU index.

        Returns
        -------
        list of (int, str)
            ``(start_slice, uuid)`` pairs, sorted by *start_slice*.
        """
        # 1. Parse MIG device index to UUID from nvidia-smi -L
        res_l = self.__class__._run_nvidia_smi(["-L"])
        gpu_migs: Dict[int, Dict[int, str]] = {}
        current_gpu = None
        for line in res_l.stdout.decode().splitlines():
            line_str = line.strip()
            if line_str.startswith("GPU"):
                match = re.search(r"GPU (\d+):", line_str)
                if match:
                    current_gpu = int(match.group(1))
                    gpu_migs[current_gpu] = {}
            elif "MIG" in line_str and current_gpu is not None:
                match = re.search(
                    r"Device\s+(\d+):\s+\(UUID:\s+(MIG-[^)]+)\)", line_str
                )
                if match:
                    dev_idx = int(match.group(1))
                    mig_uuid = match.group(2)
                    gpu_migs[current_gpu][dev_idx] = mig_uuid

        dev_to_uuid = gpu_migs.get(gpu_idx, {})

        # 2. Parse device index to gpu_instance_id from nvidia-smi -q -x
        res_q = self.__class__._run_nvidia_smi(["-q", "-x"])

        root = ET.fromstring(res_q.stdout)
        dev_to_gi: Dict[int, int] = {}
        for gpu in root.findall("gpu"):
            try:
                g_idx = int(gpu.find("minor_number").text)  # type: ignore
            except (ValueError, AttributeError):
                continue
            if g_idx == gpu_idx:
                mig_devices = gpu.find("mig_devices")
                if mig_devices is not None:
                    for dev in mig_devices.findall("mig_device"):
                        try:
                            idx = int(dev.find("index").text)  # type: ignore
                            gi_id = int(dev.find("gpu_instance_id").text)  # type: ignore
                            dev_to_gi[idx] = gi_id
                        except (ValueError, AttributeError):
                            continue
                break

        # 3. Query start slices of GI IDs on gpu_idx
        gi_infos = self.query_gpu_instances(gpu_idx)
        gi_id_to_slice = {gi["id"]: gi["start_slice"] for gi in gi_infos}

        # 4. Map everything together: start_slice -> UUID
        res: List[Tuple[int, str]] = []
        for dev_idx, uuid in dev_to_uuid.items():
            if dev_idx in dev_to_gi:
                gi_id = dev_to_gi[dev_idx]
                if gi_id in gi_id_to_slice:
                    res.append((gi_id_to_slice[gi_id], uuid))

        return sorted(res, key=lambda x: x[0])

    @classmethod
    def detect_mig_gpus(
        cls, config_dir: Path = Path("configs/gpus")
    ) -> List[DetectedGPU]:
        """Detect all GPUs with MIG mode currently enabled on this machine."""
        try:
            res = cls._run_nvidia_smi(["-q", "-x"])
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"Failed to query GPUs via nvidia-smi: {exc}")

        root = ET.fromstring(res.stdout)
        detected: List[DetectedGPU] = []

        for gpu in root.findall("gpu"):
            try:
                idx = int(gpu.find("minor_number").text.strip())
            except (ValueError, AttributeError):
                continue

            nvml_name = gpu.find("product_name").text.strip()

            # Check MIG mode
            mig_mode_el = gpu.find("mig_mode")
            if mig_mode_el is None:
                logger.debug("GPU %d: no mig_mode element — skipping.", idx)
                continue

            current_mig_el = mig_mode_el.find("current_mig")
            if current_mig_el is None or "Enabled" not in current_mig_el.text:
                logger.debug("GPU %d: MIG mode not enabled — skipping.", idx)
                continue

            logger.info("GPU %d: MIG enabled  name=%r", idx, nvml_name)

            model_name = match_gpu_model(nvml_name, config_dir)
            if model_name is None:
                logger.warning(
                    "GPU %d (%r): no matching config file in %s — skipping.",
                    idx,
                    nvml_name,
                    config_dir,
                )
                continue

            mig_profile_cls = load_mig_profile_class(model_name)
            valid_combos = derive_valid_combinations(mig_profile_cls)
            if not valid_combos:
                logger.warning(
                    "GPU %d (%s): no valid MIG combinations found — skipping.",
                    idx,
                    model_name,
                    config_dir,
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

        if not detected:
            raise RuntimeError(
                "No MIG-enabled GPUs found on this machine.  "
                "Enable MIG mode with: nvidia-smi -i <gpu_idx> -mig 1"
            )

        return sorted(detected, key=lambda d: d.gpu_idx)
