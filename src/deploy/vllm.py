"""
src/deploy/vllm.py

Manages vLLM server containers for real deployment.

Each MIG slot runs one vLLM server inside a docker-compose container, launched
via ``scripts/launch_vllm.sh``.  The model assigned to each slot is looked up
from ``configs/simulation_config.yaml`` using the agent name defined for that
GPU in ``configs/deployment.yaml``.

The class writes live state (port, model_id, container_name) back to
:data:`~src.deploy.system.SYSTEM_STATE` so callers can query the current
deployment topology at any time.

Prerequisites
-------------
* :meth:`~src.deploy.cluster.DeployGPUSetup.apply` must have been called first
  so that :data:`~src.deploy.system.SYSTEM_STATE` contains valid
  :class:`~src.deploy.models.MIGSlotState` entries with ``mig_uuid`` filled in.
* The docker daemon must be reachable and ``scripts/launch_vllm.sh`` must be
  executable.
* ``pip install requests`` (or ``httpx``) must be available.

Example
-------
::

    from src.deploy.cluster import DeployGPUSetup
    from src.deploy.vllm import VLLMManager
    from src.deploy.system import SYSTEM_STATE

    setup = DeployGPUSetup()
    setup.apply_random()                       # populates SYSTEM_STATE

    mgr = VLLMManager()
    for gpu_state in SYSTEM_STATE.gpus.values():
        mgr.start_all(gpu_state)              # launches one container per slot
        for slot in gpu_state.slots:
            mgr.wait_until_ready(slot)

    # later …
    metrics = mgr.collect_metrics(slot)
    print(metrics)  # {"ttft_mean_s": 0.12, "queue_length": 3, "running_requests": 2}
"""

from __future__ import annotations


import yaml
import time
import random
import asyncio
import logging
import requests
import subprocess
import concurrent.futures
from pathlib import Path
from typing import Any, Dict, List

from openai import AsyncOpenAI

import src.deploy.metrics as metrics
import src.deploy.system as system
from src.deploy.config import DEPLOY_CONFIG
from src.deploy.models import GPUState, MIGSlotState
from src.share.models import AgentId
import src.simulation.utils as sim_utils

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Simulated-slot state
# ---------------------------------------------------------------------------


class SimSlotState:
    """Per-MIG-UUID state for simulated permanent engine slots.

    Bundles the three counters previously stored in separate dicts:

    * ``kv_limit``   — KV-cache capacity ceiling in tokens (from SIM_CONFIG).
    * ``kv_tokens``  — KV tokens currently reserved by admitted requests.
    * ``waiting``    — Number of requests in the admission queue.
    """

    __slots__ = ("kv_limit", "kv_tokens", "waiting")

    def __init__(self, kv_limit: int) -> None:
        self.kv_limit: int = kv_limit
        self.kv_tokens: int = 0
        self.waiting: int = 0

    @property
    def kv_util(self) -> float:
        """KV-cache utilisation fraction in [0, 1]."""
        return min(1.0, self.kv_tokens / self.kv_limit) if self.kv_limit else 0.0


# ---------------------------------------------------------------------------
# VLLMManager
# ---------------------------------------------------------------------------


class VLLMManager:
    """Manages vLLM containers on MIG slots.

    The manager loads the model-to-MIG-profile mapping from
    ``configs/simulation_config.yaml`` for the agent assigned to each GPU
    (per ``configs/deployment.yaml``) and drives ``scripts/launch_vllm.sh``
    to bring containers up and down.

    Parameters
    ----------
    sim_config_path:
        Path to ``configs/simulation_config.yaml``.
    """

    def __init__(
        self,
        sim_config_path: Path = Path("configs/simulation_config.yaml"),
    ) -> None:
        self._cfg = DEPLOY_CONFIG

        # _model_map[gpu_idx][agent_id][profile_string] = model_id
        self._model_map: Dict[int, Dict[AgentId, Dict[str, str]]] = (
            self._build_model_map(sim_config_path)
        )

        # Pre-allocate the full port pool: 2 × (7 slots/GPU × num_GPUs).
        # Ports are returned to this set when a slot stops so they are reused
        # on the next reconfiguration.  Using a single set avoids maintaining a
        # separate counter and a separate free list.
        _num_gpus = max(len(system.SYSTEM_STATE.gpus), 1)
        _pool_size = 7 * _num_gpus * 2
        self._port_pool: set[int] = set(
            range(self._cfg.vllm.base_port, self._cfg.vllm.base_port + _pool_size)
        )

        # {mig_uuid: {request_id: current_tokens}}
        # Keyed by MIG UUID so that state is always tied to a specific hardware
        # instance.  Stopping a slot removes its entry; a new slot starts clean.
        self._active_reqs: Dict[str, Dict[str, int]] = {}

        # Simulated-engine KV-cache accounting, one entry per active mig_uuid.
        # Initialised lazily on the first request to a given simulated slot.
        self._sim_slots: Dict[str, SimSlotState] = {}

    # ------------------------------------------------------------------
    # Model-to-MIG mapping
    # ------------------------------------------------------------------

    def _build_model_map(
        self, sim_config_path: Path
    ) -> Dict[int, Dict[AgentId, Dict[str, str]]]:
        """Build ``{gpu_idx: {agent_id: {profile_string: model_id}}}`` map.

        Uses the actual hardware detected in :data:`~src.deploy.system.SYSTEM_STATE`
        and cross-references it with the agent definitions in the simulation config.
        """
        if not system.SYSTEM_STATE.gpus:
            logger.warning(
                "VLLMManager: SYSTEM_STATE.gpus is empty."
                "Ensure cluster.DeployGPUSetup.apply() was called before init."
            )

        with open(sim_config_path, "r") as f:
            raw: Dict[str, Any] = yaml.safe_load(f)

        sim = raw["simulation"]
        agents_cfg = sim["agents"]

        result: Dict[int, Dict[AgentId, Dict[str, str]]] = {}
        for gpu_idx, gpu_state in system.SYSTEM_STATE.gpus.items():
            hw_model = gpu_state.model_name
            result[gpu_idx] = {}

            # Build mapping for all agents, not just the default one
            for agent_id in list(AgentId):
                agent_name = agent_id.value
                if agent_name in agents_cfg and hw_model in agents_cfg[agent_name]:
                    agent_cfg = agents_cfg[agent_name][hw_model]
                    mig_cfg = agent_cfg.get("mig", {})
                    result[gpu_idx][agent_id] = {
                        p_str: p_data["model"] for p_str, p_data in mig_cfg.items()
                    }
                else:
                    logger.debug(
                        "GPU %d (%s): No model mapping for agent %s.",
                        gpu_idx,
                        hw_model,
                        agent_name,
                    )

        return result

    def model_for_slot(self, slot: MIGSlotState) -> str:
        """Return the model ID for *slot* based on the deployment configuration.

        Parameters
        ----------
        slot:
            Target :class:`~src.deploy.models.MIGSlotState`.

        Returns
        -------
        str
            Model config ID (e.g. ``"qwen2.5-7b-instruct"``).

        Raises
        ------
        KeyError
            If no model is configured for this GPU / profile combination.
        """
        profile = slot.profile_placement.profile.string
        gpu_id = slot.gpu_idx
        # Respect existing ownership if set, otherwise fallback to GPU default
        agent_id = slot.agent_id or AgentId(self._cfg.gpu_assignment[gpu_id])

        if gpu_id not in self._model_map:
            raise KeyError(f"GPU {gpu_id} not found in model map.")

        agent_map = self._model_map[gpu_id]
        if agent_id not in agent_map:
            raise KeyError(
                f"No models configured for Agent {agent_id.value} on GPU {gpu_id}."
            )

        profile_map = agent_map[agent_id]
        if profile not in profile_map:
            raise KeyError(
                f"No model configured for GPU {gpu_id} / Agent {agent_id.value} / Profile '{profile}'. "
                f"Available: {list(profile_map)}"
            )
        return profile_map[profile]

    # ------------------------------------------------------------------
    # Container lifecycle
    # ------------------------------------------------------------------

    def _alloc_port(self) -> int:
        """Remove and return the lowest port number from the pool."""
        if not self._port_pool:
            raise RuntimeError("Port pool exhausted — too many concurrent MIG slots.")
        port = min(self._port_pool)
        self._port_pool.discard(port)
        return port

    def _release_port(self, port: int) -> None:
        """Return *port* to the pool so it can be reused."""
        self._port_pool.add(port)

    def _run_script(
        self,
        mig_uuid: str,
        model_id: str,
        port: int,
        action: str,
    ) -> None:
        """Invoke ``launch_vllm.sh <mig_uuid> <model_id> <port> <action>``."""
        script = str(self._cfg.vllm.script)
        cmd = [script, mig_uuid, model_id, str(port), action]
        logger.info("vllm: %s  (cmd=%s)", action.upper(), " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True)

        # Log output for visibility as requested by user
        if result.stdout.strip():
            logger.info("vllm %s stdout:\n%s", action, result.stdout.strip())
        if result.stderr.strip():
            logger.info("vllm %s stderr:\n%s", action, result.stderr.strip())

        if result.returncode != 0:
            raise RuntimeError(
                f"launch_vllm.sh {action} failed (exit {result.returncode}):\n"
                f"{result.stderr.strip()}"
            )

    @staticmethod
    def _project_name(model_id: str, mig_uuid: str) -> str:
        """Docker compose project name derived from the model ID and MIG UUID."""
        short = (mig_uuid or "unknown")[4:8]
        # Use only the base name if it's a path, and replace dots with underscores
        model_base = (model_id or "none").split("/")[-1]
        model = model_base.replace(".", "_")
        return f"vllm-{model}-{short}"

    def start(self, slot: MIGSlotState) -> None:
        """Start a vLLM container for *slot*.

        Allocates a port, resolves the model, calls ``launch_vllm.sh up``, and
        updates :data:`~src.deploy.system.SYSTEM_STATE`.

        Parameters
        ----------
        slot:
            The :class:`~src.deploy.models.MIGSlotState` to serve.  Must have
            a non-empty ``mig_uuid`` (set by
            :meth:`~src.deploy.cluster.DeployGPUSetup.apply`).

        Raises
        ------
        RuntimeError
            If the slot has no MIG UUID or the script exits non-zero.
        KeyError
            If no model is configured for this slot.
        """
        if not slot.mig_uuid:
            raise RuntimeError(
                f"GPU {slot.gpu_idx} start_slice={slot.profile_placement.start_slice}: "
                "mig_uuid is empty.  Call cluster.apply() first."
            )

        model_id = self.model_for_slot(slot)
        is_simulated = system.SYSTEM_STATE.gpus[slot.gpu_idx].is_simulated

        port = None
        container_name = None
        if not is_simulated:
            port = self._alloc_port()
            container_name = self._project_name(model_id, slot.mig_uuid)
            self._run_script(slot.mig_uuid, model_id, port, "up")

        agent_id = slot.agent_id or AgentId(self._cfg.gpu_assignment[slot.gpu_idx])

        system.update_slot(
            slot.gpu_idx,
            slot.profile_placement.start_slice,
            model_id=model_id,
            port=port,
            container_name=container_name,
            agent_id=agent_id,
            is_ready=False,
        )
        logger.info(
            "vllm: GPU %d [%s] → model=%s%s%s",
            slot.gpu_idx,
            slot.profile_placement.profile.string,
            model_id,
            f"  port={port}" if port else " (simulated)",
            f"  container={container_name}" if container_name else "",
        )

    def _wait_for_drain(self, slot: MIGSlotState, poll_interval_s: float = 2.0) -> None:
        """Block until the vLLM server on *slot* has no in-flight requests.

        Polls ``GET /metrics`` via :mod:`src.deploy.metrics` and waits until
        both ``running_requests`` and ``queue_length`` are zero.  Returns
        immediately if the endpoint becomes unreachable (server shutting down).

        Parameters
        ----------
        slot:
            The slot to drain.  Must have a valid ``port``.
        poll_interval_s:
            Seconds between metric polls.
        """
        assert slot.port is not None
        client = metrics.VLLMMetricsClient(slot.port, timeout=5.0)

        logger.info(
            "vllm: GPU %d [%s] waiting for in-flight requests to drain …",
            slot.gpu_idx,
            slot.profile_placement.profile.string,
        )
        while True:
            try:
                # We can fetch metrics once using collect() to save an HTTP request
                stats = client.collect()
                running = stats["running_requests"]
                waiting = stats["queue_length"]
                if running == 0 and waiting == 0:
                    logger.info(
                        "vllm: GPU %d [%s] drained (running=0 waiting=0).",
                        slot.gpu_idx,
                        slot.profile_placement.profile.string,
                    )
                    return
                logger.debug(
                    "vllm: GPU %d [%s] still busy — running=%.0f waiting=%.0f, retrying …",
                    slot.gpu_idx,
                    slot.profile_placement.profile.string,
                    running,
                    waiting,
                )
            except requests.exceptions.RequestException as exc:
                logger.warning(
                    "vllm: GPU %d [%s] /metrics unreachable (%s) — assuming drained.",
                    slot.gpu_idx,
                    slot.profile_placement.profile.string,
                    exc,
                )
                return
            time.sleep(poll_interval_s)

    def stop(self, slot: MIGSlotState, graceful: bool = True) -> None:
        """Gracefully stop the vLLM container for *slot*.

        Waits for all in-flight requests to finish (by polling ``/metrics``)
        before sending the ``down`` command to docker-compose.

        Parameters
        ----------
        slot:
            The slot to stop.  ``model_id``, ``port``, and ``mig_uuid`` must
            all be set (i.e. :meth:`start` must have been called first).
        graceful:
            If False, skip the drain phase and stop immediately.
        """
        if not slot.mig_uuid or slot.model_id is None or slot.port is None:
            logger.warning(
                "vllm: GPU %d start_slice=%d: container not running — skip stop.",
                slot.gpu_idx,
                slot.profile_placement.start_slice,
            )
            return

        is_simulated = system.SYSTEM_STATE.gpus[slot.gpu_idx].is_simulated
        if not is_simulated:
            if graceful:
                self._wait_for_drain(slot)
            self._run_script(slot.mig_uuid, slot.model_id, slot.port, "down")
            # Give the driver a moment to release the device after container shutdown
            time.sleep(2.5)

        # Capture before update_slot clears the field.
        released_port = slot.port

        # Return the port to the free pool so it can be reused.
        if released_port is not None:
            self._release_port(released_port)

        # Drop the UUID's request-tracking state so a future slot starting on
        # the same MIG position begins with a completely clean slate.
        self._active_reqs.pop(slot.mig_uuid, None)
        self._sim_slots.pop(slot.mig_uuid, None)

        system.update_slot(
            slot.gpu_idx,
            slot.profile_placement.start_slice,
            model_id=None,
            port=None,
            container_name=None,
            agent_id=None,
            is_ready=False,
            is_draining=False,
        )
        logger.info(
            "vllm: GPU %d [%s] stopped%s.",
            slot.gpu_idx,
            slot.profile_placement.profile.string,
            f" (port {released_port} released)" if released_port is not None else "",
        )

    def start_all(self, gpu_state: GPUState) -> None:
        """Start vLLM containers for every slot on *gpu_state*.

        Parameters
        ----------
        gpu_state:
            Target :class:`~src.deploy.models.GPUState` from
            :data:`~src.deploy.system.SYSTEM_STATE`.
        """
        for slot in gpu_state.slots:
            self.start(slot)

    def stop_all(self, gpu_state: GPUState, graceful: bool = True) -> None:
        """Stop vLLM containers for every slot on *gpu_state*.

        Parameters
        ----------
        gpu_state:
            Target :class:`~src.deploy.models.GPUState`.
        graceful:
            If False, skip the drain phase for all slots.
        """
        # We use a thread pool to stop slots concurrently, which is critical
        # for fast teardown when multiple slots are in a "stuck" or slow-to-poll state.
        with concurrent.futures.ThreadPoolExecutor() as executor:
            list(
                executor.map(lambda s: self.stop(s, graceful=graceful), gpu_state.slots)
            )

    # ------------------------------------------------------------------
    # Readiness probing
    # ------------------------------------------------------------------

    def wait_until_ready(
        self,
        slot: MIGSlotState,
        poll_interval_s: float = 1.0,
    ) -> None:
        """Block until the vLLM server on *slot* passes its ``/health`` check.

        Once healthy, this method also queries ``/v1/models`` to detect the
        actual model ID being served and updates the system state accordingly.

        Parameters
        ----------
        slot:
            The slot to probe.  Must have a valid ``port``.
        poll_interval_s:
            Seconds between poll attempts.

        Raises
        ------
        TimeoutError
            If the server does not become ready within the configured timeout.
        """
        if system.SYSTEM_STATE.gpus[slot.gpu_idx].is_simulated:
            logger.info("vllm: GPU %d is simulated — assuming ready.", slot.gpu_idx)
            system.update_slot(
                slot.gpu_idx,
                slot.profile_placement.start_slice,
                is_ready=True,
            )
            return

        if slot.port is None:
            raise RuntimeError(
                f"GPU {slot.gpu_idx} start_slice={slot.profile_placement.start_slice}: "
                "port is None — call start() first."
            )

        deadline = time.monotonic() + self._cfg.vllm.health_timeout_s
        health_url = f"http://localhost:{slot.port}/health"
        models_url = f"http://localhost:{slot.port}/v1/models"
        logger.info("vllm: waiting for %s …", health_url)

        while time.monotonic() < deadline:
            try:
                r = requests.get(health_url, timeout=5)
                if r.status_code == 200:
                    logger.info(
                        "vllm: GPU %d port %d is ready.", slot.gpu_idx, slot.port
                    )
                    system.update_slot(
                        slot.gpu_idx,
                        slot.profile_placement.start_slice,
                        is_ready=True,
                    )
                    break
            except requests.exceptions.RequestException:
                pass
            time.sleep(poll_interval_s)
        else:
            raise TimeoutError(
                f"vLLM on GPU {slot.gpu_idx} port {slot.port} did not become ready "
                f"within {self._cfg.vllm.health_timeout_s:.0f}s."
            )

        # Robust model name detection - Runs only AFTER the server is healthy.
        logger.info("vllm: Querying model name from port %d...", slot.port)
        mr = requests.get(models_url, timeout=5)
        mr.raise_for_status()
        data = mr.json()
        if "data" not in data or len(data["data"]) == 0:
            raise RuntimeError(f"vllm: No models found at {models_url}")

        actual_model_id = data["data"][0]["id"]
        if actual_model_id != slot.model_id:
            logger.info(
                "vllm: detected server model ID '%s' (requested '%s'). Updating state.",
                actual_model_id,
                slot.model_id,
            )
            system.update_slot(
                slot.gpu_idx,
                slot.profile_placement.start_slice,
                model_id=actual_model_id,
            )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    async def send_request(
        self,
        slot: MIGSlotState,
        messages: List[Dict[str, str]],
        *,
        max_tokens: int = 2048,
        data_id: str,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Send a chat-completion request to the vLLM server on *slot*.

        Parameters
        ----------
        slot:
            Target slot (must have ``port`` and ``model_id`` set, unless simulated).
        messages:
            OpenAI-style message list, e.g.
            ``[{"role": "user", "content": "Hello"}]``.
        max_tokens:
            Maximum tokens to generate.
        data_id:
            Lookup ID for TOKENS_MAP (usually original_id from dataset).
        **kwargs:
            Extra fields forwarded to the request body.

        Returns
        -------
        dict
            Raw JSON response from the ``/v1/chat/completions`` endpoint,
            augmented with ``ttft`` and ``total_time`` metrics.
        """
        is_simulated = system.SYSTEM_STATE.gpus[slot.gpu_idx].is_simulated

        if not is_simulated and (slot.port is None or slot.model_id is None):
            raise RuntimeError(
                "Slot is not running — call start() and wait_until_ready() first."
            )

        if is_simulated:
            return await self._handle_simulated_request(
                slot, messages, max_tokens, data_id=data_id
            )

        client = AsyncOpenAI(
            base_url=f"http://localhost:{slot.port}/v1", api_key="EMPTY"
        )

        start_time = time.time()
        ttft = None

        response_stream = await client.chat.completions.create(
            model=slot.model_id,
            messages=messages,  # type: ignore
            max_tokens=max_tokens,
            temperature=0,
            timeout=self._cfg.vllm.request_timeout_s,
            stream=True,
            stream_options={"include_usage": True},
            extra_body=kwargs,
        )

        full_content = ""
        usage = None
        chunk_id = None

        if slot.mig_uuid not in self._active_reqs:
            self._active_reqs[slot.mig_uuid] = {}

        try:
            async for chunk in response_stream:
                if chunk.id is not None:
                    chunk_id = chunk.id

                if ttft is None:
                    ttft = time.time() - start_time

                if (
                    chunk.choices
                    and len(chunk.choices) > 0
                    and chunk.choices[0].delta.content
                ):
                    full_content += chunk.choices[0].delta.content

                if chunk.usage is not None:
                    usage = chunk.usage.model_dump()
                    # Track current progress
                    self._active_reqs[slot.mig_uuid][chunk.id] = usage.get(
                        "completion_tokens", 0
                    )

            total_time = time.time() - start_time
        finally:
            # Always clean up tracking even on failure/cancellation
            if chunk_id is not None:
                self._active_reqs[slot.mig_uuid].pop(chunk_id, None)

        return {
            "choices": [{"message": {"role": "assistant", "content": full_content}}],
            "usage": usage
            or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "ttft": ttft if ttft is not None else total_time,
            "total_time": total_time,
        }

    async def _handle_simulated_request(
        self,
        slot: MIGSlotState,
        messages: List[Dict[str, str]],
        max_tokens: int,
        data_id: str,
    ) -> Dict[str, Any]:
        """Mock request handler for simulated permanent engines.

        Mirrors the KV-cache admission gate from ``engine.py``:
        a request waits in a polling loop until the current KV token
        occupancy plus its own token demand fits within the slot's capacity.
        Once admitted it runs for the simulated TTFT + generation time.
        """
        agent_id = slot.agent_id
        if agent_id is None:
            # Fallback to the static assignment if slot ownership isn't explicitly set
            agent_id = AgentId(self._cfg.gpu_assignment[slot.gpu_idx])

        # Find the model being used for this agent on this simulated hardware
        hw_prof = slot.profile_placement.profile
        model_name = sim_utils.SIM_CONFIG.get_model(
            agent_id, hw_prof, gpu_id=slot.gpu_idx
        )

        # Strict lookup in TOKENS_MAP - Will raise KeyError if data_id is missing
        token_info = sim_utils.TOKENS_MAP[agent_id][model_name][data_id]
        prompt_tokens, completion_tokens = token_info

        # Get performance params
        prefill_params = sim_utils.SIM_CONFIG.get_prefill_params(
            agent_id, hw_prof, gpu_id=slot.gpu_idx
        )
        tpot_params = sim_utils.SIM_CONFIG.get_tpot_params(
            agent_id, hw_prof, gpu_id=slot.gpu_idx
        )

        # ----------------------------------------------------------------
        # KV-cache admission gate  (mirrors engine.py lines 308-315)
        # ----------------------------------------------------------------
        mig_uuid = slot.mig_uuid

        # Initialise per-slot accounting on first request
        if mig_uuid not in self._sim_slots:
            self._sim_slots[mig_uuid] = SimSlotState(
                kv_limit=sim_utils.SIM_CONFIG.get_max_kv_cache_tokens(
                    agent_id, hw_prof, gpu_id=slot.gpu_idx
                )
            )
        sim = self._sim_slots[mig_uuid]

        req_tokens = prompt_tokens + completion_tokens

        # Spin until there is room in the KV cache
        sim.waiting += 1
        try:
            while sim.kv_tokens + req_tokens > sim.kv_limit:
                await asyncio.sleep(0.5)
        finally:
            sim.waiting = max(0, sim.waiting - 1)

        # Admitted — reserve KV tokens
        sim.kv_tokens += req_tokens

        # Concurrency tracking — keyed by MIG UUID so a reconfigured slot
        # never inherits stale counters from a previous layout.
        if mig_uuid not in self._active_reqs:
            self._active_reqs[mig_uuid] = {}
        self._active_reqs[mig_uuid][data_id] = 0

        try:
            concurrency = len(self._active_reqs[mig_uuid])

            # Calculate TTFT and TPOT (correctly matching engine.py simulation logic)
            ttft = prefill_params["alpha"] + prefill_params["beta"] * prompt_tokens
            tpot = tpot_params["alpha"] + tpot_params["beta"] * concurrency

            # Add some noise
            ttft = max(0.01, random.normalvariate(ttft, prefill_params["sigma"]))
            tpot = max(0.01, random.normalvariate(tpot, tpot_params["sigma"]))

            total_gen_time = tpot * completion_tokens
            total_time = ttft + total_gen_time

            start_time = time.time()
            await asyncio.sleep(total_time)
        finally:
            self._active_reqs[mig_uuid].pop(data_id, None)
            # Release KV tokens so waiting requests can be admitted
            sim.kv_tokens = max(0, sim.kv_tokens - req_tokens)

        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Simulated response content.",
                    }
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
            "ttft": ttft,
            "total_time": time.time() - start_time,
        }

    def get_running_requests_tokens(self, mig_uuid: str) -> List[int]:
        """Return the current token counts for all active requests on *mig_uuid*."""
        return list(self._active_reqs.get(mig_uuid, {}).values())

    def get_sim_waiting(self, mig_uuid: str) -> int:
        """Return the number of requests currently waiting for KV-cache admission
        on the simulated slot identified by *mig_uuid*.

        Returns 0 for unknown / not-yet-initialised slots.
        """
        sim = self._sim_slots.get(mig_uuid)
        return sim.waiting if sim is not None else 0

    def get_sim_kv_util(self, mig_uuid: str) -> float:
        """Return the current KV-cache utilisation fraction [0, 1] for *mig_uuid*.

        Returns 0.0 if the slot has not been initialised yet.
        """
        sim = self._sim_slots.get(mig_uuid)
        return sim.kv_util if sim is not None else 0.0
