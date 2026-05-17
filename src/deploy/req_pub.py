import asyncio
import time
import logging
from typing import Dict, List, Tuple, Set, Optional

import numpy as np
from tqdm import tqdm

import src.share.models as m
from src.deploy.vllm import VLLMManager
from src.deploy.models import MIGSlotState
from src.deploy.system import SYSTEM_STATE
from src.deploy.metrics import VLLMMetricsClient
from src.share.request_loader import RequestLoader
from src.deploy.obs import OBS_COLLECTOR
from src.deploy.config import DEPLOY_CONFIG

logger = logging.getLogger(__name__)


class AgentMetrics:
    def __init__(self):
        self.ttfts: List[float] = []
        self.completion_times: List[float] = []
        self.queue_length_samples: List[float] = []

        self.profile_existence_time: Dict[str, float] = {
            p.short_name: 0.0 for p in m.MIGProfile
        }
        self.total_observation_time: float = 0.0

        self.tokens_by_profile: Dict[str, int] = {p.short_name: 0 for p in m.MIGProfile}


class ReqPublisher:
    def __init__(self, vllm_manager: VLLMManager, request_loader: RequestLoader):
        self.vllm_manager = vllm_manager
        self.request_loader = request_loader
        self.agent_metrics: Dict[m.AgentId, AgentMetrics] = {}
        self.slot_queues: Dict[Tuple[int, int], int] = {}
        self._request_tasks: Set[asyncio.Task] = set()
        self._loop_tasks: List[asyncio.Task] = []
        self.pbar: Optional[tqdm] = None

    def _get_profile_key(self, profile: m.MIGProfileBase) -> str:
        return profile.profile_type.short_name

    def start_sending(self, duration_s: float) -> asyncio.Future:
        """Set up and launch all dispatch/metric tasks, returning a Future that
        resolves when all loops have finished.  The caller should ``await`` this
        Future and then call :meth:`cleanup` in its own ``finally`` block."""
        logger.info(f"Starting dispatcher for {duration_s} seconds")

        # Initialize metrics and slot queues based on initial SYSTEM_STATE
        for gpu_idx, gpu_state in SYSTEM_STATE.gpus.items():
            for slot in gpu_state.slots:
                if slot.agent_id:
                    self.slot_queues[(gpu_idx, slot.profile_placement.start_slice)] = 0
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

        self.pbar = tqdm(total=total_requests, desc="Benchmark", unit="req", leave=True)

        self._loop_tasks = []
        for agent_id, requests in agent_to_requests.items():
            self._loop_tasks.append(
                asyncio.create_task(self._dispatch_loop(agent_id, requests, duration_s))
            )
            self._loop_tasks.append(
                asyncio.create_task(self._metric_loop(agent_id, duration_s))
            )

        return asyncio.gather(*self._loop_tasks, return_exceptions=True)

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

        if self.pbar:
            self.pbar.close()
            self.pbar = None

        # 3. Print final benchmark summary
        self.report_metrics()

    def _get_active_slots(self, agent_id: m.AgentId) -> List[MIGSlotState]:
        """Return slots ready to receive NEW requests."""
        active_slots = []
        for gpu_idx, gpu_state in SYSTEM_STATE.gpus.items():
            for slot in gpu_state.slots:
                is_simulated = slot.mig_uuid.startswith("SIM-MIG-")
                if (
                    slot.agent_id == agent_id
                    and (slot.port is not None or is_simulated)
                    and not slot.is_draining
                ):
                    active_slots.append(slot)
        return active_slots

    def _get_all_slots(self, agent_id: m.AgentId) -> List[MIGSlotState]:
        """Return all slots currently owned by the agent, including draining ones."""
        slots = []
        for gpu_idx, gpu_state in SYSTEM_STATE.gpus.items():
            for slot in gpu_state.slots:
                if slot.agent_id == agent_id:
                    slots.append(slot)
        return slots

    async def _dispatch_loop(
        self, agent_id: m.AgentId, requests: List[m.Request], duration_s: float
    ):
        start_time = time.time()
        dispatch_tasks = []

        for req in sorted(requests, key=lambda x: x.arrival_time):
            now = time.time()
            elapsed = now - start_time
            if elapsed > duration_s:
                break

            wait_time = req.arrival_time - elapsed
            if wait_time > 0:
                await asyncio.sleep(wait_time)

            now = time.time()
            if now - start_time > duration_s:
                break

            active_slots = self._get_active_slots(agent_id)
            assert active_slots, f"No active slots for agent {agent_id}"

            def selection_key(slot: MIGSlotState):
                q_len = self.slot_queues.get(
                    (slot.gpu_idx, slot.profile_placement.start_slice), 0
                )
                has_waiting = q_len > 0
                try:
                    size = slot.profile_placement.profile.size
                except ValueError:
                    size = 1
                return (has_waiting, q_len, -size)

            best_slot = min(active_slots, key=selection_key)
            slot_key = (best_slot.gpu_idx, best_slot.profile_placement.start_slice)

            self.slot_queues[slot_key] = self.slot_queues.get(slot_key, 0) + 1

            task = asyncio.create_task(self._handle_request(agent_id, req, best_slot))
            self._request_tasks.add(task)
            task.add_done_callback(self._request_tasks.discard)
            dispatch_tasks.append(task)

    async def _handle_request(
        self, agent_id: m.AgentId, req: m.Request, slot: MIGSlotState
    ):
        OBS_COLLECTOR.record_arrival(agent_id)
        messages = [{"role": "user", "content": req.prompt}]

        slot_key = (slot.gpu_idx, slot.profile_placement.start_slice)
        try:
            lookup_id = req.original_id if req.original_id else req.id
            timeout = DEPLOY_CONFIG.vllm.request_timeout_s
            is_simulated = slot.mig_uuid.startswith("SIM-MIG-")

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

            # Update tokens processed - we count these even for simulated engines
            profile_key = self._get_profile_key(slot.profile_placement.profile)
            if profile_key not in metrics.tokens_by_profile:
                metrics.tokens_by_profile[profile_key] = 0
            metrics.tokens_by_profile[profile_key] += tokens_generated

            if not is_simulated:
                metrics.ttfts.append(response["ttft"])
                metrics.completion_times.append(response["total_time"])

                # Record completion to OBS_COLLECTOR for physical slots
                ttft = response["ttft"]
                total_time = response["total_time"]
                tpot = (
                    (total_time - ttft) / tokens_generated
                    if tokens_generated > 0
                    else 0.0
                )
                is_permanent = False
                mig_idx = slot.profile_placement.profile.profile_type.value
                OBS_COLLECTOR.record_completion(
                    agent_id, ttft, tpot, is_permanent, mig_idx, tokens_generated
                )

        except asyncio.TimeoutError:
            logger.warning(
                f"Request {req.id} timed out after {timeout} seconds on slot {slot_key}!"
            )
        except Exception as e:
            logger.error(f"Request {req.id} failed on slot {slot_key}: {e}")
        finally:
            self.slot_queues[slot_key] = max(0, self.slot_queues.get(slot_key, 0) - 1)
            if self.pbar:
                self.pbar.update(1)

    async def _metric_loop(self, agent_id: m.AgentId, duration_s: float):
        start_time = time.time()
        metrics = self.agent_metrics[agent_id]

        while True:
            now = time.time()
            if now - start_time >= duration_s:
                break

            all_slots = self._get_all_slots(agent_id)
            slot_samples: Dict[int, Dict[str, float]] = {}

            if all_slots:
                total_q = 0.0
                n_phy_slot = 0
                for s in all_slots:
                    if s.port is not None:
                        try:
                            client = VLLMMetricsClient(s.port, timeout=1.0)
                            data = await asyncio.to_thread(client.collect)
                            running = data["running_requests"]
                            total_q += running
                            n_phy_slot += 1

                            idx = s.profile_placement.profile.profile_type.value
                            slot_samples[idx] = {
                                "running": running,
                                "waiting": data["queue_length"],
                                "kv_util": data["kv_cache_util"],
                                "tpot": data["tpot_mean_s"],
                            }
                        except Exception as e:
                            logger.debug(f"Metrics fetch failed for port {s.port}: {e}")

                metrics.queue_length_samples.append(
                    total_q / n_phy_slot if n_phy_slot != 0 else 0.0
                )
                if slot_samples:
                    OBS_COLLECTOR.record_samples(agent_id, slot_samples)
            else:
                metrics.queue_length_samples.append(0.0)

            for slot in all_slots:
                profile_key = self._get_profile_key(slot.profile_placement.profile)
                if profile_key not in metrics.profile_existence_time:
                    metrics.profile_existence_time[profile_key] = 0.0
                metrics.profile_existence_time[profile_key] += 1.0

            metrics.total_observation_time += 1.0

            await asyncio.sleep(1.0)

    def report_metrics(self):
        print("\n--- Dispatcher Benchmark Results ---")
        for agent_id, metrics in self.agent_metrics.items():
            print(f"Agent: {agent_id.value}")

            if metrics.ttfts:
                p25 = np.percentile(metrics.ttfts, 25)
                p50 = np.percentile(metrics.ttfts, 50)
                p75 = np.percentile(metrics.ttfts, 75)
                p99 = np.percentile(metrics.ttfts, 99)
                print(
                    f"  TTFT (s)  : p25={p25:.3f}, median={p50:.3f}, p75={p75:.3f}, p99={p99:.3f}"
                )
            else:
                print("  TTFT (s)  : N/A")

            if metrics.completion_times:
                p99_ct = np.percentile(metrics.completion_times, 99)
                print(f"  Total Time: p99={p99_ct:.3f}")

            if metrics.queue_length_samples:
                avg_q = sum(metrics.queue_length_samples) / len(
                    metrics.queue_length_samples
                )
                print(f"  Avg Queue Length: {avg_q:.2f}")

            print("  MIG Existence Percentages (over benchmark period):")
            for p in m.MIGProfile:
                p_key = p.short_name
                t = metrics.profile_existence_time.get(p_key, 0.0)
                pct = (
                    (t / metrics.total_observation_time * 100)
                    if metrics.total_observation_time > 0
                    else 0
                )
                print(f"    {p_key}: {pct:.1f}%")

            total_tokens = sum(metrics.tokens_by_profile.values())
            print("  Token Generation by MIGs:")
            for p in m.MIGProfile:
                p_key = p.short_name
                count = metrics.tokens_by_profile.get(p_key, 0)
                pct = (count / total_tokens * 100) if total_tokens > 0 else 0
                print(f"    {p_key}: {pct:.1f}% ({count} tokens)")

        print("------------------------------------\n")
