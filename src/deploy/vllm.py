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
import logging
import requests
import subprocess
from pathlib import Path
from typing import Any, Dict, List

from openai import AsyncOpenAI

import src.deploy.metrics as metrics
import src.deploy.system as system
from src.deploy.config import DEPLOY_CONFIG
from src.deploy.models import GPUState, MIGSlotState
from src.share.models import AgentId

logger = logging.getLogger(__name__)


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

        # _model_map[gpu_idx][profile_string] = model_id
        self._model_map: Dict[int, Dict[str, str]] = self._build_model_map(
            sim_config_path
        )

        # Port counter — assigned once per slot, incremented permanently.
        self._next_port: int = self._cfg.vllm.base_port

    # ------------------------------------------------------------------
    # Model-to-MIG mapping
    # ------------------------------------------------------------------

    def _build_model_map(self, sim_config_path: Path) -> Dict[int, Dict[str, str]]:
        """Build ``{gpu_idx: {profile_string: model_id}}`` from the sim config.

        For each GPU, the agent name is taken from
        :attr:`~src.deploy.config.DeploymentConfig.gpu_assignment`.
        """
        with open(sim_config_path, "r") as f:
            raw: Dict[str, Any] = yaml.safe_load(f)

        sim = raw["simulation"]
        agents_cfg = sim["agents"]
        cluster_cfg = sim["cluster"]

        result: Dict[int, Dict[str, str]] = {}
        for gpu_idx_str, hw_model in cluster_cfg.items():
            gpu_idx = int(gpu_idx_str)
            agent_name = self._cfg.gpu_assignment[gpu_idx]
            agent_cfg = agents_cfg[agent_name][hw_model]
            mig_cfg: Dict[str, Any] = agent_cfg["mig"]

            result[gpu_idx] = {
                profile_str: profile_data["model"]
                for profile_str, profile_data in mig_cfg.items()
            }
            logger.debug(
                "GPU %d (%s / %s): model map = %s",
                gpu_idx,
                agent_name,
                hw_model,
                result[gpu_idx],
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
        gpu_map = self._model_map[slot.gpu_idx]
        if profile not in gpu_map:
            raise KeyError(
                f"No model configured for GPU {slot.gpu_idx} profile '{profile}'. "
                f"Available: {list(gpu_map)}"
            )
        return gpu_map[profile]

    # ------------------------------------------------------------------
    # Container lifecycle
    # ------------------------------------------------------------------

    def _alloc_port(self) -> int:
        port = self._next_port
        self._next_port += 1
        return port

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
        if result.returncode != 0:
            raise RuntimeError(
                f"launch_vllm.sh {action} failed (exit {result.returncode}):\n"
                f"{result.stderr.strip()}"
            )

    @staticmethod
    def _project_name(model_id: str, mig_uuid: str) -> str:
        """Docker compose project name derived from the model ID and MIG UUID."""
        short = (mig_uuid or "unknown")[4:8]
        model = (model_id or "none").replace(".", "_")
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
        port = self._alloc_port()
        container_name = self._project_name(model_id, slot.mig_uuid)

        self._run_script(slot.mig_uuid, model_id, port, "up")

        agent_name = self._cfg.gpu_assignment[slot.gpu_idx]

        system.update_slot(
            slot.gpu_idx,
            slot.profile_placement.start_slice,
            model_id=model_id,
            port=port,
            container_name=container_name,
            agent_id=AgentId(agent_name),
        )
        logger.info(
            "vllm: GPU %d [%s] → model=%s  port=%d  container=%s",
            slot.gpu_idx,
            slot.profile_placement.profile.string,
            model_id,
            port,
            container_name,
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

    def stop(self, slot: MIGSlotState) -> None:
        """Gracefully stop the vLLM container for *slot*.

        Waits for all in-flight requests to finish (by polling ``/metrics``)
        before sending the ``down`` command to docker-compose.

        Parameters
        ----------
        slot:
            The slot to stop.  ``model_id``, ``port``, and ``mig_uuid`` must
            all be set (i.e. :meth:`start` must have been called first).
        """
        if not slot.mig_uuid or slot.model_id is None or slot.port is None:
            logger.warning(
                "vllm: GPU %d start_slice=%d: container not running — skip stop.",
                slot.gpu_idx,
                slot.profile_placement.start_slice,
            )
            return

        self._wait_for_drain(slot)
        self._run_script(slot.mig_uuid, slot.model_id, slot.port, "down")

        system.update_slot(
            slot.gpu_idx,
            slot.profile_placement.start_slice,
            model_id=None,
            port=None,
            container_name=None,
            agent_id=None,
        )
        logger.info(
            "vllm: GPU %d [%s] stopped.",
            slot.gpu_idx,
            slot.profile_placement.profile.string,
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

    def stop_all(self, gpu_state: GPUState) -> None:
        """Stop vLLM containers for every slot on *gpu_state*.

        Parameters
        ----------
        gpu_state:
            Target :class:`~src.deploy.models.GPUState`.
        """
        for slot in gpu_state.slots:
            self.stop(slot)

    # ------------------------------------------------------------------
    # Readiness probing
    # ------------------------------------------------------------------

    def wait_until_ready(
        self,
        slot: MIGSlotState,
        poll_interval_s: float = 1.0,
    ) -> None:
        """Block until the vLLM server on *slot* passes its ``/health`` check.

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
        if slot.port is None:
            raise RuntimeError(
                f"GPU {slot.gpu_idx} start_slice={slot.profile_placement.start_slice}: "
                "port is None — call start() first."
            )

        deadline = time.monotonic() + self._cfg.vllm.health_timeout_s
        url = f"http://localhost:{slot.port}/health"
        logger.info("vllm: waiting for %s …", url)

        while time.monotonic() < deadline:
            try:
                r = requests.get(url, timeout=5)
                if r.status_code == 200:
                    logger.info(
                        "vllm: GPU %d port %d is ready.", slot.gpu_idx, slot.port
                    )
                    return
            except requests.exceptions.RequestException:
                pass
            time.sleep(poll_interval_s)

        raise TimeoutError(
            f"vLLM on GPU {slot.gpu_idx} port {slot.port} did not become ready "
            f"within {self._cfg.vllm.health_timeout_s:.0f}s."
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
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Send a chat-completion request to the vLLM server on *slot*.

        Parameters
        ----------
        slot:
            Target slot (must have ``port`` and ``model_id`` set).
        messages:
            OpenAI-style message list, e.g.
            ``[{"role": "user", "content": "Hello"}]``.
        max_tokens:
            Maximum tokens to generate.
        temperature:
            Sampling temperature.
        **kwargs:
            Extra fields forwarded to the request body.

        Returns
        -------
        dict
            Raw JSON response from the ``/v1/chat/completions`` endpoint,
            augmented with ``ttft`` and ``total_time`` metrics.
        """
        if slot.port is None or slot.model_id is None:
            raise RuntimeError(
                "Slot is not running — call start() and wait_until_ready() first."
            )

        client = AsyncOpenAI(base_url=f"http://localhost:{slot.port}/v1")

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

        async for chunk in response_stream:
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

        total_time = time.time() - start_time

        return {
            "choices": [{"message": {"role": "assistant", "content": full_content}}],
            "usage": usage
            or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "ttft": ttft if ttft is not None else total_time,
            "total_time": total_time,
        }
