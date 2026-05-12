import asyncio
import time
import logging
from typing import Dict, List, Tuple

import numpy as np

import src.share.models as m
import src.simulation.utils as utils
from src.deploy.vllm import VLLMManager
from src.deploy.models import MIGSlotState
from src.deploy.system import SYSTEM_STATE
from src.deploy.metrics import VLLMMetricsClient
from src.share.request_loader import RequestLoader
from src.deploy.obs import OBS_COLLECTOR

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
        self.slot_queues: Dict[Tuple[int, int], int] = {}
        self.agent_metrics: Dict[m.AgentId, AgentMetrics] = {}
        self._active_tasks = set()

    def _get_profile_key(self, profile: m.MIGProfileBase) -> str:
        return profile.profile_type.short_name

    async def run_benchmark(self, duration_s: float):
        logger.info(f"Starting dispatcher benchmark for {duration_s} seconds")
        tasks = []

        # Initialize metrics and slot queues based on initial SYSTEM_STATE
        for gpu_idx, gpu_state in SYSTEM_STATE.gpus.items():
            for slot in gpu_state.slots:
                if slot.agent_id:
                    self.slot_queues[(gpu_idx, slot.profile_placement.start_slice)] = 0
                    if slot.agent_id not in self.agent_metrics:
                        self.agent_metrics[slot.agent_id] = AgentMetrics()

        for agent_id in self.agent_metrics.keys():
            requests = self.request_loader.generate_requests(agent_id)
            tasks.append(
                asyncio.create_task(self._dispatch_loop(agent_id, requests, duration_s))
            )
            tasks.append(asyncio.create_task(self._metric_loop(agent_id, duration_s)))

        try:
            await asyncio.gather(*tasks)
        except (asyncio.CancelledError, KeyboardInterrupt):
            logger.info("Benchmark interrupted. Cancelling outstanding requests...")
            for t in tasks:
                if not t.done():
                    t.cancel()
            for t in list(self._active_tasks):
                if not t.done():
                    t.cancel()

            all_tasks = tasks + list(self._active_tasks)
            if all_tasks:
                await asyncio.gather(*all_tasks, return_exceptions=True)
            raise
        finally:
            self.report_metrics()

    def _get_active_slots(self, agent_id: m.AgentId) -> List[MIGSlotState]:
        """Return slots ready to receive NEW requests."""
        active_slots = []
        for gpu_idx, gpu_state in SYSTEM_STATE.gpus.items():
            for slot in gpu_state.slots:
                if (
                    slot.agent_id == agent_id
                    and slot.port is not None
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
            if not active_slots:
                logger.warning(
                    f"No active slots for agent {agent_id} to serve request {req.id}"
                )
                continue

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
            self._active_tasks.add(task)
            task.add_done_callback(self._active_tasks.discard)
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
            max_tokens = 2048
            if slot.model_id:
                model_req_map = utils.TOKENS_MAP.get(agent_id, {}).get(slot.model_id)
                if model_req_map:
                    lookup_id = req.original_id if req.original_id else req.id
                    if lookup_id in model_req_map:
                        _, completion_tokens = model_req_map[lookup_id]
                        max_tokens = completion_tokens

            response = await self.vllm_manager.send_request(
                slot=slot, messages=messages, max_tokens=max_tokens
            )

            metrics = self.agent_metrics[agent_id]
            metrics.ttfts.append(response.get("ttft", 0.0))
            metrics.completion_times.append(response.get("total_time", 0.0))

            usage = response.get("usage", {})
            tokens_generated = usage.get("completion_tokens", 0)

            profile_key = self._get_profile_key(slot.profile_placement.profile)
            if profile_key not in metrics.tokens_by_profile:
                metrics.tokens_by_profile[profile_key] = 0
            metrics.tokens_by_profile[profile_key] += tokens_generated

            # Record completion to OBS_COLLECTOR
            ttft = response.get("ttft", 0.0)
            total_time = response.get("total_time", 0.0)
            tpot = (
                (total_time - ttft) / tokens_generated if tokens_generated > 0 else 0.0
            )
            is_permanent = False
            mig_idx = slot.profile_placement.profile.profile_type.value
            OBS_COLLECTOR.record_completion(
                agent_id, ttft, tpot, is_permanent, mig_idx, tokens_generated
            )

        except Exception as e:
            logger.error(f"Request {req.id} failed on slot {slot_key}: {e}")
        finally:
            self.slot_queues[slot_key] = max(0, self.slot_queues.get(slot_key, 0) - 1)

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
                for s in all_slots:
                    if s.port is not None:
                        try:
                            client = VLLMMetricsClient(s.port, timeout=1.0)
                            data = await asyncio.to_thread(client.collect)
                            total_q += data["running_requests"]

                            idx = s.profile_placement.profile.profile_type.value
                            slot_samples[idx] = {
                                "running": data["running_requests"],
                                "waiting": data["queue_length"],
                                "kv_util": data["kv_cache_util"],
                                "tpot": data["tpot_mean_s"],
                            }
                        except Exception as e:
                            logger.debug(f"Metrics fetch failed for port {s.port}: {e}")

                metrics.queue_length_samples.append(total_q / len(all_slots))
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

            total_time_all_profiles = sum(metrics.profile_existence_time.values())
            print("  MIG Existence Percentages (by active slot-seconds):")
            for p in m.MIGProfile:
                p_key = p.short_name
                t = metrics.profile_existence_time.get(p_key, 0.0)
                pct = (
                    (t / total_time_all_profiles * 100)
                    if total_time_all_profiles > 0
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
