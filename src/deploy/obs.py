import json
from pathlib import Path
import math
import time
import logging
from collections import deque
from dataclasses import dataclass, field
import asyncio
from typing import Any, Dict, List, Tuple, Optional

import src.share.models as m
from src.deploy.system import SYSTEM_STATE
from src.training.config import TRAINING_CONFIG
from src.deploy.config import DEPLOY_CONFIG
from src.simulation.config import NUM_MIG_SLICES

logger = logging.getLogger(__name__)


NUM_LOGICAL_SLOTS = len(m.MIGProfile) + 1
CACHE_DIR = Path(".cache")
CACHE_FILE = CACHE_DIR / "avg_response_len.json"


@dataclass
class AgentStats:
    history: Dict[str, Any] = field(
        default_factory=lambda: {
            "arrival_rate": deque(
                [0.0] * TRAINING_CONFIG.arrival_rate_history_length,
                maxlen=TRAINING_CONFIG.arrival_rate_history_length,
            ),
            "queue_length": deque(maxlen=2),
            "running_requests": deque(maxlen=2),
            "kv_utilization": (0.0,) * NUM_LOGICAL_SLOTS,
        }
    )

    # Accumulators for the CURRENT interval (reset every action_interval)
    interval_requests_count: int = 0
    # Average response length tracking
    avg_response_len: float = 256.0  # Default initial guess
    total_completed_reqs: int = 0

    # Metric samples for averaging (collected at ~1Hz)
    metric_samples: List[Dict[str, Any]] = field(
        default_factory=lambda: [
            {
                "queue": [],
                "running": [],
                "kv": [],
                "latency": deque(maxlen=100),
                "tpot": deque(maxlen=100),
            }
            for _ in range(NUM_LOGICAL_SLOTS)
        ]
    )

    # Action history (cooldown tracking in terms of intervals)
    action_history: Dict[str, Dict[str, Any]] = field(
        default_factory=lambda: {
            "split": {"intervals": TRAINING_CONFIG.action_cooldown},
            "merge": {"intervals": TRAINING_CONFIG.action_cooldown},
            "give": {"intervals": TRAINING_CONFIG.action_cooldown, "amount": 0},
            "receive": {"intervals": TRAINING_CONFIG.action_cooldown, "amount": 0},
        }
    )

    reconfig_counts: Dict[str, int] = field(
        default_factory=lambda: {
            "split": 0,
            "merge": 0,
            "transfer": 0,
        }
    )


class ObservationCollector:
    def __init__(self):
        self._agent_stats: Dict[m.AgentId, AgentStats] = {
            m.AgentId.CODING: AgentStats(),
            m.AgentId.RAG: AgentStats(),
        }
        self._last_interval_start = time.time()
        self._current_budget = TRAINING_CONFIG.reconfig_budget
        self._reconfig_flag = False
        self._last_action_downtime = 0.0
        self._refresh_task: Optional[asyncio.Task] = None
        self.reconfig_history: List[Dict[str, Any]] = []
        self._last_observation: Optional[m.EnvironmentStateData] = None
        self._load_cache()

    def _load_cache(self):
        if CACHE_FILE.exists():
            try:
                with open(CACHE_FILE, "r") as f:
                    data = json.load(f)
                    for aid_str, val in data.items():
                        try:
                            aid = m.AgentId(aid_str)
                            if aid in self._agent_stats:
                                self._agent_stats[aid].avg_response_len = val["avg"]
                                self._agent_stats[aid].total_completed_reqs = val[
                                    "count"
                                ]
                        except ValueError:
                            continue
            except Exception as e:
                logger.warning(f"Failed to load response length cache: {e}")

    def _save_cache(self):
        try:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            data = {
                aid.value: {
                    "avg": stats.avg_response_len,
                    "count": stats.total_completed_reqs,
                }
                for aid, stats in self._agent_stats.items()
            }
            with open(CACHE_FILE, "w") as f:
                json.dump(data, f)
        except Exception as e:
            logger.warning(f"Failed to save response length cache: {e}")

    @property
    def reconfig_flag(self) -> bool:
        return self._reconfig_flag

    @property
    def current_budget(self) -> float:
        return self._current_budget

    # -----------------------------------------------------------------------
    # Event Hooks (Called by ReqPublisher or similar)
    # -----------------------------------------------------------------------

    def record_arrival(self, agent_id: m.AgentId):
        self._agent_stats[agent_id].interval_requests_count += 1

    def record_completion(
        self,
        agent_id: m.AgentId,
        ttft: float,
        tpot: float,
        is_permanent: bool,
        mig_idx: int,
        tokens: int = 0,
    ):
        w_t = TRAINING_CONFIG.w("ttft")
        w_p = TRAINING_CONFIG.w("tpot")
        composite = w_t * ttft + w_p * tpot

        stats = self._agent_stats[agent_id]
        idx = 6 if is_permanent else mig_idx
        stats.metric_samples[idx]["latency"].append(composite)

        if tokens > 0:
            n = stats.total_completed_reqs
            stats.avg_response_len = (stats.avg_response_len * n + tokens) / (n + 1)
            stats.total_completed_reqs += 1
            if stats.total_completed_reqs % 10 == 0:
                self._save_cache()

    def record_samples(
        self, agent_id: m.AgentId, samples_dict: Dict[int, Dict[str, float]]
    ):
        """
        samples_dict: {slot_idx: {"running": int, "waiting": int, "kv_util": float, "tpot": float}}
        slot_idx: 0-5 for MIG physical slots, index 6 for permanent engine slots.
        """
        stats = self._agent_stats[agent_id]
        for slot_idx, metrics in samples_dict.items():
            entry = stats.metric_samples[slot_idx]
            entry["queue"].append(int(metrics["waiting"]))
            entry["running"].append(int(metrics["running"]))
            entry["kv"].append(metrics["kv_util"])
            if metrics.get("tpot", 0.0) > 0:
                entry["tpot"].append(metrics["tpot"])

    # -----------------------------------------------------------------------
    # Interval Management
    # -----------------------------------------------------------------------

    def start_new_interval(self):
        """Finalize metrics for the current interval and reset counters."""
        now = time.time()
        duration = now - self._last_interval_start

        for agent_id, stats in self._agent_stats.items():
            # 1. Arrival Rate
            rate = stats.interval_requests_count / duration
            stats.history["arrival_rate"].appendleft(rate)
            stats.interval_requests_count = 0

            # 2. Avg Queue Lengths & Running Requests & KV Cache
            avg_qs = []
            avg_runs = []
            avg_kvs = []
            for i in range(NUM_LOGICAL_SLOTS):
                entry = stats.metric_samples[i]
                qs = entry["queue"]
                rs = entry["running"]
                kvs = entry["kv"]
                avg_qs.append(sum(qs) / len(qs) if qs else 0.0)
                avg_runs.append(sum(rs) / len(rs) if rs else 0.0)
                avg_kvs.append(sum(kvs) / len(kvs) if kvs else 0.0)
                qs.clear()
                rs.clear()
                kvs.clear()

            stats.history["queue_length"].appendleft(tuple(avg_qs))
            stats.history["running_requests"].appendleft(tuple(avg_runs))
            stats.history["kv_utilization"] = tuple(avg_kvs)

            # 3. Action Cooldowns
            for entry in stats.action_history.values():
                entry["intervals"] += 1

        self._last_interval_start = now
        # Budget refresh logic (simulating simulation/environment_state.py:91)
        # Note: Real-time budget refresh might need a separate timer.

    def set_last_action(self, agent_id: m.AgentId, action_type: str, amount: int = 0):
        entry = self._agent_stats[agent_id].action_history[action_type]
        entry["intervals"] = 0
        if "amount" in entry:
            entry["amount"] = amount

    def increment_reconfig_count(self, agent_id: m.AgentId, action_type: str):
        """Increment the split, merge, or transfer count for a given agent."""
        if agent_id in self._agent_stats:
            stats = self._agent_stats[agent_id]
            if action_type in stats.reconfig_counts:
                stats.reconfig_counts[action_type] += 1

    def get_reconfig_counts(self, agent_id: m.AgentId) -> Dict[str, int]:
        """Get the reconfiguration counts dictionary for a given agent."""
        if agent_id in self._agent_stats:
            return self._agent_stats[agent_id].reconfig_counts
        return {"split": 0, "merge": 0, "transfer": 0}

    # -----------------------------------------------------------------------
    # Budget & Reconfig Management
    # -----------------------------------------------------------------------

    def record_reconfig(self, action_name: str, cost: float, details: str):
        """Record a reconfiguration action in the history."""
        self.reconfig_history.append({
            "timestamp": time.time(),
            "action": action_name,
            "cost": cost,
            "details": details,
        })
        if len(self.reconfig_history) > 10:
            self.reconfig_history.pop(0)

    def consume_budget(self, cost: float):
        """Deduct reconfiguration cost from the current budget."""
        self._current_budget = max(0.0, self._current_budget - cost)
        self._last_action_downtime = cost
        self._reconfig_flag = True
        logger.info(
            f"Consumed {cost:.1f}s from budget. Remaining: {self._current_budget:.1f}s"
        )

    def mark_reconfig_complete(self):
        """Mark the reconfiguration process as finished."""
        self._reconfig_flag = False
        logger.info("Reconfiguration complete.")

    def refresh_budget(self):
        """Reset the reconfiguration budget to the default maximum."""
        self._current_budget = TRAINING_CONFIG.reconfig_budget
        logger.info(f"Budget refreshed to {self._current_budget}s.")

    def start_budget_refresh_loop(self):
        """Start a background task to refresh the budget periodically."""
        if self._refresh_task is not None:
            return

        async def _loop():
            while True:
                await asyncio.sleep(TRAINING_CONFIG.refresh_period)
                self.refresh_budget()

        self._refresh_task = asyncio.create_task(_loop())
        logger.info(
            f"Budget refresh loop started (period: {TRAINING_CONFIG.refresh_period}s)"
        )

    # -----------------------------------------------------------------------
    # Observation Generation
    # -----------------------------------------------------------------------

    def get_last_queue_length(self, gpu_idx: int, start_slice: int) -> float:
        """Return the most recent queue length sample for a specific slot."""
        # Find the agent owning this slot (if any)
        from src.deploy.system import get_slot

        slot = get_slot(gpu_idx, start_slice)
        if not slot or not slot.agent_id:
            return 0.0

        stats = self._agent_stats[slot.agent_id]
        is_permanent = SYSTEM_STATE.gpus[gpu_idx].is_simulated
        idx = 6 if is_permanent else slot.profile_placement.profile.profile_type.value
        q_samples = stats.metric_samples[idx]["queue"]
        return q_samples[-1] if q_samples else 0.0

    def get_observation(self) -> m.EnvironmentStateData:
        """Construct the full normalized observation dictionary."""
        agents = list(m.AgentId)

        # 1. Arrival Rate
        arrival_rate = {}
        arrival_rate_trend = {}
        arrival_rate_history = {}
        for aid in agents:
            stats = self._agent_stats[aid]
            divisor = DEPLOY_CONFIG.get_arrival_rate_divisor(aid)
            raw_history = stats.history["arrival_rate"]
            scaled_history = (
                [r / divisor for r in raw_history]
                if divisor > 0.0
                else list(raw_history)
            )

            curr = scaled_history[0]
            prev = scaled_history[1] if len(scaled_history) > 1 else 0.0

            arrival_rate[aid] = curr / TRAINING_CONFIG.norm_arrival_rate
            if prev == 0:
                arrival_rate_trend[aid] = 1.0 if curr > 0 else 0.0
            else:
                arrival_rate_trend[aid] = (curr - prev) / prev

            arrival_rate_history[aid] = tuple(
                r / TRAINING_CONFIG.norm_arrival_rate for r in scaled_history
            )

        predicted_arrival_rate = {
            aid: arrival_rate[aid] * (1 + arrival_rate_trend[aid]) for aid in agents
        }

        # 2. Queue Lengths & Running Requests
        avg_queue_length = {}
        avg_queue_length_trend = {}
        avg_running_requests = {}

        max_q_expected = TRAINING_CONFIG.norm_avg_queue_length
        q_denom = math.log10(1 + max_q_expected)

        for aid in agents:
            stats = self._agent_stats[aid]
            curr_qs = (
                stats.history["queue_length"][0]
                if stats.history["queue_length"]
                else [0.0] * NUM_LOGICAL_SLOTS
            )
            prev_qs = (
                stats.history["queue_length"][1]
                if len(stats.history["queue_length"]) > 1
                else [0.0] * NUM_LOGICAL_SLOTS
            )
            curr_runs = (
                stats.history["running_requests"][0]
                if stats.history["running_requests"]
                else [0.0] * NUM_LOGICAL_SLOTS
            )

            # Normalized logs
            avg_queue_length[aid] = tuple(math.log10(1 + q) / q_denom for q in curr_qs)

            trends = []
            for c, p in zip(curr_qs, prev_qs):
                if p == 0:
                    trends.append(1.0 if c > 0 else 0.0)
                else:
                    trd = (c - p) / p
                    trd = max(
                        -TRAINING_CONFIG.queue_length_trend_clamp,
                        min(trd, TRAINING_CONFIG.queue_length_trend_clamp),
                    )
                    trends.append(trd)
            avg_queue_length_trend[aid] = tuple(trends)

            avg_running_requests[aid] = tuple(
                r / TRAINING_CONFIG.norm_avg_running_requests for r in curr_runs
            )

        # 3. KV Cache & Latency
        kv_cache_util = self._get_kv_cache_utilization()
        latent_proportions, raw_latent_totals = self._get_avg_composite_latency()

        # 4. Topology & Ownership
        mig_profile_id_onehot = self._get_mig_profile_id_onehot()
        ownership_grid = self._get_ownership_grid()
        agent_owns_mig = self._get_agent_owns_mig()
        mig_geometry = self._get_mig_geometry()

        # 5. Global Ratios
        vram_ratio = self._get_total_vram_ratio()
        sm_ratio = self._get_total_sm_ratio()

        ratios = self._calculate_global_agent_ratios(
            arrival_rate,
            avg_queue_length,
            avg_running_requests,
            kv_cache_util,
            raw_latent_totals,
            vram_ratio,
            sm_ratio,
        )

        state_data: m.EnvironmentStateData = {
            "arrival_rate": arrival_rate,
            "predicted_arrival_rate": predicted_arrival_rate,
            "arrival_rate_history": arrival_rate_history,
            "avg_queue_length": avg_queue_length,
            "avg_queue_length_trend": avg_queue_length_trend,
            "avg_running_requests": avg_running_requests,
            "kv_cache_utilization": kv_cache_util,
            "avg_composite_latency": latent_proportions,
            "mig_profile_id_onehot": mig_profile_id_onehot,
            "ownership_grid": ownership_grid,
            "agent_owns_mig": agent_owns_mig,
            "mig_geometry": mig_geometry,
            "current_budget": self._current_budget
            / TRAINING_CONFIG.norm_current_budget,
            "downtime_ratio": self._last_action_downtime
            / TRAINING_CONFIG.action_interval,
            "total_sm_ratio": sm_ratio,
            "total_vram_ratio": vram_ratio,
            "recovery_flag": self._reconfig_flag,
            "last_split": {aid: self._get_action_norm(aid, "split") for aid in agents},
            "last_merge": {aid: self._get_action_norm(aid, "merge") for aid in agents},
            "last_give": {aid: self._get_action_norm(aid, "give") for aid in agents},
            "last_receive": {
                aid: self._get_action_norm(aid, "receive") for aid in agents
            },
            "last_give_amount": {
                aid: self._get_action_amount_norm(aid, "give") for aid in agents
            },
            "last_receive_amount": {
                aid: self._get_action_amount_norm(aid, "receive") for aid in agents
            },
            "requests": {aid: [] for aid in agents},
            "agent_arrival_rate_ratio": ratios["agent_arrival_rate_ratio"],
            "agent_avg_queue_len_ratio": ratios["agent_avg_queue_len_ratio"],
            "agent_avg_running_req_ratio": ratios["agent_avg_running_req_ratio"],
            "agent_avg_kv_cache_ratio": ratios["agent_avg_kv_cache_ratio"],
            "agent_avg_composite_latency_ratio": ratios[
                "agent_avg_composite_latency_ratio"
            ],
            "agent_vram_ratio": ratios["agent_vram_ratio"],
            "agent_sm_ratio": ratios["agent_sm_ratio"],
        }
        self._last_observation = state_data
        return state_data

    # -----------------------------------------------------------------------
    # Helper Methods (Calculations)
    # -----------------------------------------------------------------------

    def _get_kv_cache_utilization(self) -> Dict[m.AgentId, Tuple[float, ...]]:
        res = {}
        for aid in list(m.AgentId):
            stats = self._agent_stats[aid]
            res[aid] = stats.history["kv_utilization"]
        return res

    def _get_avg_composite_latency(
        self,
    ) -> Tuple[Dict[m.AgentId, Tuple[float, ...]], Dict[m.AgentId, float]]:
        proportions = {}
        raw_totals = {}
        for aid in list(m.AgentId):
            stats = self._agent_stats[aid]
            raw_avgs = []
            for i in range(NUM_LOGICAL_SLOTS):
                q = stats.metric_samples[i]["latency"]
                raw_avgs.append(sum(q) / len(q) if q else 0.0)

            total = sum(raw_avgs)
            raw_totals[aid] = total
            if total > 0:
                proportions[aid] = tuple(v / total for v in raw_avgs)
            else:
                proportions[aid] = tuple(0.0 for _ in raw_avgs)
        return proportions, raw_totals

    def get_avg_response_len(self, agent_id: m.AgentId) -> float:
        return self._agent_stats[agent_id].avg_response_len

    def get_current_tpot(self, agent_id: m.AgentId) -> float:
        stats = self._agent_stats[agent_id]
        all_samples = []
        for i in range(NUM_LOGICAL_SLOTS):
            all_samples.extend(stats.metric_samples[i]["tpot"])
        if not all_samples:
            return 0.05  # Default fallback
        return sum(all_samples) / len(all_samples)

    def _get_mig_profile_id_onehot(self) -> Dict[int, List[float]]:
        from src.share.mig_matrix import STATE_DEFINITIONS, STATE_ID_MAP

        onehot = {}
        for gpu_idx, gpu_state in SYSTEM_STATE.gpus.items():
            if gpu_state.is_simulated:
                # Simulated GPUs are not RL-managed; the observation consumer
                # only reads physical GPU indices so a zero vector is correct.
                onehot[gpu_idx] = [0.0] * 15
                continue

            # Reconstruction of state_id:
            # Current profiles as a sorted list of MIGProfile enum members
            current_profiles = sorted(
                [
                    slot.profile_placement.profile.profile_type
                    for slot in gpu_state.slots
                ],
                key=lambda x: x.value,
            )

            found_sid = None
            for sid, defs in STATE_DEFINITIONS.items():
                if sorted(defs, key=lambda x: x.value) == current_profiles:
                    found_sid = sid
                    break

            if found_sid is None:
                raise ValueError(
                    f"GPU {gpu_idx}: could not identify MIG state for profiles: {current_profiles}"
                )

            found_idx = STATE_ID_MAP.get(found_sid, 0)
            vec = [0.0] * 15
            vec[found_idx] = 1.0
            onehot[gpu_idx] = vec
        return onehot

    def _get_ownership_grid(self) -> Dict[int, List[int]]:
        grid_dict = {}
        for gpu_idx, gpu_state in SYSTEM_STATE.gpus.items():
            grid = [0] * NUM_MIG_SLICES
            for slot in gpu_state.slots:
                if slot.agent_id:
                    owner_val = 1 if slot.agent_id == m.AgentId.CODING else 2
                    size = slot.profile_placement.profile.size
                    start = slot.profile_placement.start_slice
                    for i in range(start, start + size):
                        if i < NUM_MIG_SLICES:
                            grid[i] = owner_val
            grid_dict[gpu_idx] = grid
        return grid_dict

    def _get_agent_owns_mig(self) -> Dict[m.AgentId, Tuple[float, ...]]:
        res = {}
        divisor = TRAINING_CONFIG.norm_mig_geometry
        for aid in list(m.AgentId):
            counts = [0] * len(m.MIGProfile)
            for gpu in SYSTEM_STATE.gpus.values():
                for slot in gpu.slots:
                    is_permanent = gpu.is_simulated
                    if slot.agent_id == aid and not is_permanent:
                        counts[slot.profile_placement.profile.profile_type.value] += 1
            res[aid] = tuple(c / divisor for c in counts)
        return res

    def _get_mig_geometry(self) -> Dict[int, List[float]]:
        res = {}
        divisor = TRAINING_CONFIG.norm_mig_geometry
        agents = list(m.AgentId)
        for gpu_idx, gpu in SYSTEM_STATE.gpus.items():
            sizes = [0.0] * len(agents)
            for slot in gpu.slots:
                is_permanent = gpu.is_simulated
                if slot.agent_id in agents and not is_permanent:
                    idx = agents.index(slot.agent_id)
                    sizes[idx] += slot.profile_placement.profile.size
            res[gpu_idx] = [s / divisor for s in sizes]
        return res

    def _get_total_sm_ratio(self) -> Dict[m.AgentId, float]:
        res = {}
        for aid in list(m.AgentId):
            total_size = 0
            for gpu in SYSTEM_STATE.gpus.values():
                for slot in gpu.slots:
                    if slot.agent_id == aid:
                        total_size += slot.profile_placement.profile.size
            res[aid] = total_size / TRAINING_CONFIG.norm_total_sm_ratio
        return res

    def _get_total_vram_ratio(self) -> Dict[m.AgentId, float]:
        res = {}
        for aid in list(m.AgentId):
            total_vram = 0
            for gpu in SYSTEM_STATE.gpus.values():
                for slot in gpu.slots:
                    if slot.agent_id == aid:
                        total_vram += slot.profile_placement.profile.vram
            res[aid] = total_vram / TRAINING_CONFIG.norm_total_vram_ratio
        return res

    def _get_action_norm(self, agent_id: m.AgentId, action: str) -> float:
        norm = float(TRAINING_CONFIG.action_cooldown)
        val = self._agent_stats[agent_id].action_history[action]["intervals"]
        return min(val, norm) / norm

    def _get_action_amount_norm(self, agent_id: m.AgentId, action: str) -> float:
        norm_action = float(TRAINING_CONFIG.action_cooldown)
        norm_amount = TRAINING_CONFIG.norm_vram_transfer_amount
        entry = self._agent_stats[agent_id].action_history[action]
        if entry["intervals"] == 0 or entry["intervals"] >= norm_action:
            return 0.0
        return entry["amount"] / norm_amount

    def _calculate_global_agent_ratios(
        self,
        arrival_rate,
        avg_queue_length,
        avg_running_requests,
        kv_cache_util,
        raw_latent_totals,
        vram_ratio,
        sm_ratio,
    ):
        # Implementation of simulation/environment_state.py:279
        def get_total(data_dict, aid):
            val = data_dict[aid]
            return sum(val) if isinstance(val, (tuple, list)) else float(val)

        metrics_map = {
            "agent_arrival_rate_ratio": arrival_rate,
            "agent_avg_queue_len_ratio": avg_queue_length,
            "agent_avg_running_req_ratio": avg_running_requests,
            "agent_avg_kv_cache_ratio": kv_cache_util,
            "agent_vram_ratio": vram_ratio,
            "agent_sm_ratio": sm_ratio,
        }

        ratios = {}
        epsilon = 1e-6
        for key, data in metrics_map.items():
            c_val = get_total(data, m.AgentId.CODING)
            r_val = get_total(data, m.AgentId.RAG)
            ratios[key] = (c_val - r_val) / (c_val + r_val + epsilon)

        # Latency ratio (Log-Normalized symmetric ratio)
        max_exp = TRAINING_CONFIG.norm_avg_composite_latency
        denom = math.log10(1 + max_exp)
        l_c_raw = raw_latent_totals[m.AgentId.CODING]
        l_r_raw = raw_latent_totals[m.AgentId.RAG]
        l_c_norm = math.log10(1 + l_c_raw) / denom
        l_r_norm = math.log10(1 + l_r_raw) / denom
        ratios["agent_avg_composite_latency_ratio"] = (l_c_norm - l_r_norm) / (
            l_c_norm + l_r_norm + epsilon
        )

        return ratios


# Global singleton
OBS_COLLECTOR = ObservationCollector()
