import asyncio
import time
import logging
from typing import Dict, List, Set

import src.share.models as m
from src.deploy.vllm import VLLMManager
from src.deploy.models import MIGSlotState
from src.deploy.system import SYSTEM_STATE
from src.deploy.metrics import VLLMMetricsClient
from src.share.request_loader import RequestLoader
from src.deploy.obs import OBS_COLLECTOR
from src.deploy.config import DEPLOY_CONFIG

from src.deploy.report import AgentMetrics, print_benchmark_report, MetricsCollector

logger = logging.getLogger(__name__)


class ReqPublisher:
    def __init__(self, vllm_manager: VLLMManager, request_loader: RequestLoader):
        self.vllm_manager = vllm_manager
        self.request_loader = request_loader
        self.agent_metrics: Dict[m.AgentId, AgentMetrics] = {}
        self.metrics_collector = MetricsCollector(self.agent_metrics, self.vllm_manager)
        self._request_tasks: Set[asyncio.Task] = set()
        self._loop_tasks: List[asyncio.Task] = []
        self.completed_requests: int = 0
        self.total_requests: int = 0

    @property
    def dashboard(self):
        return self.metrics_collector.dashboard

    @dashboard.setter
    def dashboard(self, value):
        self.metrics_collector.dashboard = value

    def start_sending(self, duration_s: float) -> asyncio.Future:
        """Set up and launch all dispatch/metric tasks, returning a Future that
        resolves when all loops have finished.  The caller should ``await`` this
        Future and then call :meth:`cleanup` in its own ``finally`` block."""
        logger.info(f"Starting dispatcher for {duration_s} seconds")

        # Initialize metrics based on initial SYSTEM_STATE
        for gpu_idx, gpu_state in SYSTEM_STATE.gpus.items():
            for slot in gpu_state.slots:
                if slot.agent_id:
                    if slot.agent_id not in self.agent_metrics:
                        self.agent_metrics[slot.agent_id] = AgentMetrics()

        # Pre-generate requests to know the total count for the progress bar
        total_requests = 0
        agent_to_requests = {}
        for agent_id in self.agent_metrics.keys():
            all_reqs = self.request_loader.generate_requests(agent_id)
            # Filter requests that fall within the duration
            filtered = [r for r in all_reqs if r.arrival_time <= duration_s]
            agent_to_requests[agent_id] = filtered
            total_requests += len(filtered)

        self.completed_requests = 0
        self.total_requests = total_requests

        self._loop_tasks = []
        for agent_id, requests in agent_to_requests.items():
            self._loop_tasks.append(
                asyncio.create_task(self._dispatch_loop(agent_id, requests, duration_s))
            )
            self._loop_tasks.append(
                self.metrics_collector.start_collection(agent_id, duration_s)
            )

        return asyncio.gather(*self._loop_tasks, return_exceptions=False)

    async def cleanup(self):
        """Cancel any still-running loop tasks, drain all in-flight requests,
        close the progress bar, and print the final metrics report.
        Call this from the caller's ``finally`` block after awaiting
        :meth:`start_sending`."""
        # 1. Stop the dispatch and metric loops
        for t in self._loop_tasks:
            if not t.done():
                t.cancel()
        if self._loop_tasks:
            await asyncio.gather(*self._loop_tasks, return_exceptions=True)

        # 2. Wait for all in-flight requests that were already dispatched with a grace period
        if self._request_tasks:
            logger.info(
                f"Waiting for {len(self._request_tasks)} in-flight requests to complete..."
            )
            try:
                await asyncio.wait_for(
                    asyncio.gather(*self._request_tasks, return_exceptions=True),
                    timeout=DEPLOY_CONFIG.vllm.request_timeout_s,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "Some in-flight requests did not finish within the request timeout. Cancelling them..."
                )
                for t in self._request_tasks:
                    if not t.done():
                        t.cancel()
                await asyncio.gather(*self._request_tasks, return_exceptions=True)
            self._request_tasks.clear()

        # 3. Print final benchmark summary
        print_benchmark_report(self.agent_metrics)

    def _get_active_slots(self, agent_id: m.AgentId) -> List[MIGSlotState]:
        """Return slots ready to receive NEW requests."""
        active_slots = []
        for gpu_idx, gpu_state in SYSTEM_STATE.gpus.items():
            for slot in gpu_state.slots:
                is_simulated = gpu_state.is_simulated
                if (
                    slot.agent_id == agent_id
                    and (slot.port is not None or is_simulated)
                    and not slot.is_draining
                    and slot.is_ready
                ):
                    active_slots.append(slot)
        return active_slots

    def _get_active_pattern(self, agent_id: m.AgentId, t: float) -> str:
        assert (
            self.request_loader.phase_history
            and agent_id in self.request_loader.phase_history
        ), f"Phase history not initialized or missing for agent {agent_id.value}!"
        for ph in self.request_loader.phase_history[agent_id]:
            if ph["start_time"] <= t <= ph["start_time"] + ph["duration"] + 1e-5:
                return ph["pattern"]
        last_ph = self.request_loader.phase_history[agent_id][-1]
        if t > last_ph["start_time"] + last_ph["duration"]:
            return last_ph["pattern"]
        raise AssertionError(f"No pattern found for {agent_id} at t={t:.2f}s")

    async def _get_slot_waiting(self, slot: MIGSlotState) -> float:
        """Query the waiting queue length of slot directly from metrics.py or VLLMManager."""
        if SYSTEM_STATE.gpus[slot.gpu_idx].is_simulated:
            return float(self.vllm_manager.get_sim_waiting(slot.mig_uuid))
        assert slot.port is not None
        client = VLLMMetricsClient(slot.port, timeout=1)
        return await asyncio.to_thread(client.waiting_requests)

    async def _dispatch_loop(
        self, agent_id: m.AgentId, requests: List[m.Request], duration_s: float
    ):
        start_time = time.time()
        dispatch_tasks = []

        for req in sorted(requests, key=lambda x: x.arrival_time):
            now = time.time()
            elapsed = now - start_time
            wait_time = req.arrival_time - elapsed
            if wait_time > 0:
                await asyncio.sleep(wait_time)

            active_slots = self._get_active_slots(agent_id)
            if not active_slots:
                for gpu_idx, gpu_state in SYSTEM_STATE.gpus.items():
                    for slot in gpu_state.slots:
                        logger.error(
                            f"  GPU {gpu_idx} start_slice={slot.profile_placement.start_slice} "
                            f"agent_id={slot.agent_id} port={slot.port} is_draining={slot.is_draining} is_ready={slot.is_ready}"
                        )
            assert active_slots, f"No active slots for agent {agent_id}"

            # Query waiting queue size for all active slots concurrently
            slot_waitings = {}
            if len(active_slots) == 1:
                # Bypass gather if there is only one slot
                best_slot = active_slots[0]
            else:
                waiting_results = await asyncio.gather(
                    *(self._get_slot_waiting(slot) for slot in active_slots),
                    return_exceptions=False,
                )
                for slot, res in zip(active_slots, waiting_results):
                    slot_waitings[slot.mig_uuid] = (
                        float(res) if isinstance(res, (int, float)) else 0.0
                    )

                def selection_key(slot: MIGSlotState):
                    waiting = slot_waitings[slot.mig_uuid]
                    has_waiting = waiting > 0
                    size = slot.profile_placement.profile.size
                    return (has_waiting, waiting, -size)

                best_slot = min(active_slots, key=selection_key)

            task = asyncio.create_task(self._handle_request(agent_id, req, best_slot))
            self._request_tasks.add(task)
            task.add_done_callback(self._request_tasks.discard)
            dispatch_tasks.append(task)

        if dispatch_tasks:
            await asyncio.gather(*dispatch_tasks, return_exceptions=True)

    async def _handle_request(
        self, agent_id: m.AgentId, req: m.Request, slot: MIGSlotState
    ):
        OBS_COLLECTOR.record_arrival(agent_id)
        messages = [{"role": "user", "content": req.prompt}]

        slot_key = (slot.gpu_idx, slot.profile_placement.start_slice)
        try:
            lookup_id = req.original_id if req.original_id else req.id
            timeout = DEPLOY_CONFIG.vllm.request_timeout_s
            is_simulated = SYSTEM_STATE.gpus[slot.gpu_idx].is_simulated

            if is_simulated:
                response = await self.vllm_manager.send_request(
                    slot=slot,
                    messages=messages,
                    max_tokens=2048,
                    data_id=lookup_id,
                )
            else:
                response = await asyncio.wait_for(
                    self.vllm_manager.send_request(
                        slot=slot,
                        messages=messages,
                        max_tokens=2048,
                        data_id=lookup_id,
                    ),
                    timeout=timeout,
                )

            metrics = self.agent_metrics[agent_id]
            tokens_generated = response["usage"]["completion_tokens"]

            # Track token migration attribution by active workload patterns
            r_pat_c = self._get_active_pattern(m.AgentId.CODING, req.arrival_time)
            r_pat_r = self._get_active_pattern(m.AgentId.RAG, req.arrival_time)
            mig_profile = slot.profile_placement.profile
            if mig_profile.profile_type in metrics.tokens_by_mig[r_pat_c][r_pat_r]:
                metrics.tokens_by_mig[r_pat_c][r_pat_r][mig_profile.profile_type] += (
                    tokens_generated
                )

            if not is_simulated:
                metrics.ttfts.append(response["ttft"])
                metrics.completion_times.append(response["total_time"])

            # Record completion to OBS_COLLECTOR (for both physical and simulated/permanent slots)
            ttft = response["ttft"]
            total_time = response["total_time"]
            tpot = (
                (total_time - ttft) / tokens_generated if tokens_generated > 0 else 0.0
            )
            is_permanent = is_simulated
            mig_idx = (
                6 if is_permanent else slot.profile_placement.profile.profile_type.value
            )
            OBS_COLLECTOR.record_completion(
                agent_id, ttft, tpot, is_permanent, mig_idx, tokens_generated
            )

        except asyncio.TimeoutError:
            logger.warning(
                f"Request {req.id} timed out after {timeout} seconds on slot {slot_key}!"
            )
        except Exception as e:
            logger.error(f"Request {req.id} failed on slot {slot_key}: {e}")
            if not is_simulated:
                import requests

                def probe():
                    try:
                        r = requests.get(
                            f"http://localhost:{slot.port}/health", timeout=2.0
                        )
                        return r.status_code == 200
                    except Exception:
                        return False

                is_healthy = await asyncio.to_thread(probe)
                if not is_healthy and slot.is_ready:
                    slot.is_ready = False
                    logger.warning(
                        f"Slot {slot_key} failed health probe. Restarting engine in background."
                    )
                    asyncio.create_task(self._restart_engine(slot))
        finally:
            self.completed_requests += 1

    async def _restart_engine(self, slot: MIGSlotState):
        slot_key = (slot.gpu_idx, slot.profile_placement.start_slice)
        logger.info(f"Background restart initiated for slot {slot_key}")
        try:
            await asyncio.to_thread(self.vllm_manager.stop, slot, graceful=False)
            await asyncio.to_thread(self.vllm_manager.start, slot)
            await asyncio.to_thread(self.vllm_manager.wait_until_ready, slot)
            logger.info(f"Background restart complete for slot {slot_key}")
        except Exception as e:
            logger.error(f"Failed to background restart engine for slot {slot_key}: {e}")

