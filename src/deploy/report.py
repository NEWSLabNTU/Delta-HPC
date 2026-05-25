import asyncio
import time
import logging
from typing import Dict, List
import numpy as np
import tabulate

import src.share.models as m
from src.bench.models import Workload
from src.deploy.obs import OBS_COLLECTOR
from src.deploy.models import MIGSlotState
from src.deploy.system import SYSTEM_STATE
from src.deploy.metrics import VLLMMetricsClient
from src.deploy.vllm import VLLMManager

logger = logging.getLogger(__name__)


class AgentMetrics:
    def __init__(self):
        self.ttfts: List[float] = []
        self.completion_times: List[float] = []
        self.tpot_samples: List[float] = []
        self.queue_length_samples: List[float] = []
        self.error_count: int = 0

        patterns = [w.value for w in Workload]
        self.tokens_by_mig = {
            pc: {pr: {prof: 0 for prof in m.MIGProfile} for pr in patterns}
            for pc in patterns
        }

        self.profile_existence_time: Dict[str, float] = {
            p.short_name: 0.0 for p in m.MIGProfile
        }
        self.total_observation_time: float = 0.0


def print_benchmark_report(agent_metrics: Dict[m.AgentId, AgentMetrics]):
    print("\n--- Dispatcher Benchmark Results ---")
    for agent_id, metrics in agent_metrics.items():
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
            
        print(f"  Errors    : {metrics.error_count}")

        if metrics.tpot_samples:
            p25_tp = np.percentile(metrics.tpot_samples, 25)
            p50_tp = np.percentile(metrics.tpot_samples, 50)
            p75_tp = np.percentile(metrics.tpot_samples, 75)
            p99_tp = np.percentile(metrics.tpot_samples, 99)
            print(
                f"  TPOT (s)  : p25={p25_tp:.3f}, median={p50_tp:.3f}, p75={p75_tp:.3f}, p99={p99_tp:.3f}"
            )
        else:
            print("  TPOT (s)  : N/A")

        if metrics.queue_length_samples:
            avg_q = sum(metrics.queue_length_samples) / len(
                metrics.queue_length_samples
            )
            print(f"  Avg Queue Length: {avg_q:.2f}")

        counts = OBS_COLLECTOR.get_reconfig_counts(agent_id)
        print(
            f"  S/M/T       : {counts['split']}/{counts['merge']}/{counts['transfer']}"
        )

        print("  MIG Existence Percentages (over benchmark period):")
        for p in m.MIGProfile:
            p_key = p.short_name
            t = metrics.profile_existence_time.get(p_key, 0.0)
            pct = (
                (t / metrics.total_observation_time * 100)
                if metrics.total_observation_time > 0
                else 0
            )
            print(f"    {p_key}: {pct:.1f}")

        # Derive total tokens per MIG profile from tokens_by_mig
        derived_tokens_by_profile = {
            p.short_name: sum(
                metrics.tokens_by_mig[pc][pr][p]
                for pc in metrics.tokens_by_mig
                for pr in metrics.tokens_by_mig[pc]
            )
            for p in m.MIGProfile
        }
        total_tokens = sum(derived_tokens_by_profile.values())
        print("  Token Generation by MIGs:")
        for p in m.MIGProfile:
            p_key = p.short_name
            count = derived_tokens_by_profile.get(p_key, 0)
            pct = (count / total_tokens * 100) if total_tokens > 0 else 0
            print(f"    {p_key}: {pct:.1f}% ({count} tokens)")

    patterns = [w.value for w in Workload]
    print("\n● Tokens by MIG Matrix (%)")
    print("Format: 7G | 4G | 3G | 2G | 1L | 1S")
    for agent_id, metrics in agent_metrics.items():
        print(f"\n[{agent_id.name} Agent]")
        headers = ["Coding \\ RAG"] + patterns
        mat_data: List[List[str]] = []

        # Calculate token_mig_percentages for this agent
        token_mig_percentages = {
            pc: {
                pr: {
                    k: (
                        v / sum(metrics.tokens_by_mig[pc][pr].values()) * 100
                        if sum(metrics.tokens_by_mig[pc][pr].values()) > 0
                        else 0
                    )
                    for k, v in metrics.tokens_by_mig[pc][pr].items()
                }
                for pr in patterns
            }
            for pc in patterns
        }

        for pat_c in patterns:
            row_data = [pat_c]
            for pat_r in patterns:
                pat_dict = token_mig_percentages[pat_c][pat_r]
                mig_vals: List[str] = []
                # Sort by logical profile index (MIGProfile idx or value)
                sorted_pat_migs = sorted(
                    pat_dict.keys(),
                    key=lambda x: x.idx if hasattr(x, "idx") else x.value,
                )
                for mig in sorted_pat_migs:
                    mig_vals.append(f"{pat_dict[mig]:3.0f}%")
                row_data.append(" | ".join(mig_vals))
            mat_data.append(row_data)

        print(
            tabulate.tabulate(
                mat_data,
                headers=headers,
                tablefmt="fancy_outline",
                stralign="right",
                headersglobalalign="center",
            )
        )

    print("------------------------------------\n")


class MetricsCollector:
    def __init__(self, agent_metrics: Dict[m.AgentId, AgentMetrics], vllm_manager: VLLMManager):
        self.agent_metrics = agent_metrics
        self.vllm_manager = vllm_manager
        self.dashboard = None
        self._loop_tasks: List[asyncio.Task] = []

    def start_collection(self, agent_id: m.AgentId, duration_s: float) -> asyncio.Task:
        task = asyncio.create_task(self._metric_loop(agent_id, duration_s))
        self._loop_tasks.append(task)
        return task

    def _get_profile_key(self, profile: m.MIGProfileBase) -> str:
        return profile.profile_type.short_name

    def _get_all_slots(self, agent_id: m.AgentId) -> List[MIGSlotState]:
        """Return all slots currently owned by the agent, including draining ones."""
        slots = []
        for gpu_idx, gpu_state in SYSTEM_STATE.gpus.items():
            for slot in gpu_state.slots:
                if slot.agent_id == agent_id:
                    slots.append(slot)
        return slots

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
                raw_samples: Dict[int, List[Dict[str, float]]] = {}

                for s in all_slots:
                    is_simulated = SYSTEM_STATE.gpus[s.gpu_idx].is_simulated
                    if not is_simulated:
                        try:
                            client = VLLMMetricsClient(s.port, timeout=1.0)
                            data = await asyncio.to_thread(client.collect)
                            waiting = data["queue_length"]
                            running = data["running_requests"]
                            total_q += waiting
                            n_phy_slot += 1
                            tpot = data["tpot_mean_s"]

                            # Save to live per-slot metrics cache for the dashboard
                            if self.dashboard:
                                self.dashboard.record_live_slot_metrics(
                                    s.mig_uuid,
                                    {
                                        "running": running,
                                        "waiting": waiting,
                                        "kv_util": data["kv_cache_util"],
                                        "tpot": tpot,
                                    },
                                )

                            idx = s.profile_placement.profile.profile_type.value
                            if idx not in raw_samples:
                                raw_samples[idx] = []
                            raw_samples[idx].append(
                                {
                                    "running": running,
                                    "waiting": waiting,
                                    "kv_util": data["kv_cache_util"],
                                    "tpot": tpot,
                                }
                            )
                        except Exception as e:
                            logger.debug(f"Metrics fetch failed for port {s.port}: {e}")
                    else:
                        # Simulated/Permanent backup engine slot — use real
                        # KV-cache accounting from VLLMManager instead of estimates.
                        running = float(
                            len(self.vllm_manager._active_reqs.get(s.mig_uuid, {}))
                        )
                        waiting = float(self.vllm_manager.get_sim_waiting(s.mig_uuid))
                        kv_util = self.vllm_manager.get_sim_kv_util(s.mig_uuid)
                        tpot = OBS_COLLECTOR.get_current_tpot(agent_id)

                        total_q += waiting
                        n_phy_slot += 1

                        # Save to live per-slot metrics cache for the dashboard
                        if self.dashboard:
                            self.dashboard.record_live_slot_metrics(
                                s.mig_uuid,
                                {
                                    "running": running,
                                    "waiting": waiting,
                                    "kv_util": kv_util,
                                    "tpot": tpot,
                                },
                            )

                        idx = 6
                        if idx not in raw_samples:
                            raw_samples[idx] = []
                        raw_samples[idx].append(
                            {
                                "running": running,
                                "waiting": waiting,
                                "kv_util": kv_util,
                                "tpot": tpot,
                            }
                        )

                # Aggregate raw_samples into slot_samples (sum running/waiting, avg kv/tpot)
                for idx, items in raw_samples.items():
                    if not items:
                        continue
                    slot_samples[idx] = {
                        "running": sum(item["running"] for item in items),
                        "waiting": sum(item["waiting"] for item in items),
                        "kv_util": sum(item["kv_util"] for item in items) / len(items),
                        "tpot": sum(item["tpot"] for item in items) / len(items),
                    }

                metrics.queue_length_samples.append(
                    total_q / n_phy_slot if n_phy_slot != 0 else 0.0
                )
                step_tpots = [
                    item["tpot"]
                    for items in raw_samples.values()
                    for item in items
                    if item["tpot"] > 0.0
                ]
                if step_tpots:
                    metrics.tpot_samples.append(sum(step_tpots) / len(step_tpots))
                if slot_samples:
                    OBS_COLLECTOR.record_samples(agent_id, slot_samples)
            else:
                metrics.queue_length_samples.append(0.0)

            for slot in all_slots:
                if not slot.is_ready:
                    continue
                profile_key = self._get_profile_key(slot.profile_placement.profile)
                if profile_key not in metrics.profile_existence_time:
                    metrics.profile_existence_time[profile_key] = 0.0
                metrics.profile_existence_time[profile_key] += 1.0

            metrics.total_observation_time += 1.0

            await asyncio.sleep(1.0)

