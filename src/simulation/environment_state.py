import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Tuple, Optional

import src.share.models as m
import src.simulation.models as sm
from src.share.mig_matrix import STATE_ID_MAP, SLICE_MAPPING
from src.training.config import TRAINING_CONFIG
from src.simulation.config import NUM_MIG_SLICES


@dataclass
class AgentStats:
    queue_length_integrals: List[float] = field(
        default_factory=lambda: [0.0] * NUM_MIG_SLICES
    )
    running_requests_integrals: List[float] = field(
        default_factory=lambda: [0.0] * NUM_MIG_SLICES
    )
    last_queue_lengths: List[int] = field(default_factory=lambda: [0] * NUM_MIG_SLICES)
    last_running_request_counts: List[int] = field(
        default_factory=lambda: [0] * NUM_MIG_SLICES
    )
    last_arrival_rate: Optional[float] = None
    last_avg_queue_lengths: Optional[Tuple[float, ...]] = None
    arrival_rate_history: Tuple[float, ...] = field(
        default_factory=lambda: tuple(
            [0.0] * TRAINING_CONFIG.arrival_rate_history_length
        )
    )
    interval_requests: List[m.Request] = field(default_factory=list[m.Request])
    mig_composite_latencies: Dict[int, deque[float]] = field(
        default_factory=lambda: {mig.value: deque(maxlen=100) for mig in m.MIGProfile}
    )
    # action_history keys: "split", "merge", "give", "receive"
    # Each value is a dict with "steps": int and (optionally) "amount": float
    action_history: Dict[
        sm.ActionHistoryKey,
        Dict[Literal["steps", "amount"], int],
    ] = field(
        default_factory=lambda: {
            "split": {"steps": TRAINING_CONFIG.action_cooldown},
            "merge": {"steps": TRAINING_CONFIG.action_cooldown},
            "give": {"steps": TRAINING_CONFIG.action_cooldown, "amount": 0},
            "receive": {"steps": TRAINING_CONFIG.action_cooldown, "amount": 0},
        }
    )
    give_to_steps: Dict[str, int] = field(
        default_factory=lambda: {
            aid.value: TRAINING_CONFIG.action_cooldown for aid in m.AgentId
        }
    )
    pending_request_count: int = 0


class EnvironmentStateImpl(sm.EnvironmentState):
    def __init__(self):
        self._agent_stats: Dict[m.AgentId, AgentStats] = defaultdict(AgentStats)
        self._last_queue_update_time: float = 0.0
        self._reconfig_flag: bool = False
        self._current_budget = TRAINING_CONFIG.reconfig_budget
        self._last_action_downtime: float = 0.0

    @property
    def current_budget(self) -> float:
        return self._current_budget

    @current_budget.setter
    def current_budget(self, v: float) -> None:
        self._current_budget = v

    @property
    def reconfig_flag(self) -> bool:
        return self._reconfig_flag

    @reconfig_flag.setter
    def reconfig_flag(self, v: bool):
        self._reconfig_flag = v

    @property
    def last_action_downtime(self) -> float:
        return self._last_action_downtime

    @last_action_downtime.setter
    def last_action_downtime(self, v: float) -> None:
        self._last_action_downtime = v

    @property
    def interval_requests(self) -> Dict[m.AgentId, List[m.Request]]:
        return {
            aid: self._agent_stats[aid].interval_requests
            for aid in self._agent_stats.keys()
        }

    def refresh_budget(self) -> None:
        self._current_budget = TRAINING_CONFIG.reconfig_budget

    def advance_all_last_action(self) -> None:
        for stats in self._agent_stats.values():
            for entry in stats.action_history.values():
                entry["steps"] += 1
            for aid in stats.give_to_steps.keys():
                stats.give_to_steps[aid] += 1

    def reset_last_actions(self) -> None:
        for stats in self._agent_stats.values():
            stats.action_history["split"]["steps"] = TRAINING_CONFIG.action_cooldown
            stats.action_history["merge"]["steps"] = TRAINING_CONFIG.action_cooldown
            stats.action_history["receive"]["steps"] = TRAINING_CONFIG.action_cooldown
            stats.action_history["receive"]["amount"] = 0
            stats.action_history["give"]["steps"] = TRAINING_CONFIG.action_cooldown
            stats.action_history["give"]["amount"] = 0
            for aid in m.AgentId:
                stats.give_to_steps[aid.value] = TRAINING_CONFIG.action_cooldown

    def set_last_action(
        self,
        agent_id: m.AgentId,
        event_type: sm.ActionHistoryKey,
        amount: int = 0,
        target_agent: Optional[m.AgentId] = None,
    ) -> None:
        entry = self._agent_stats[agent_id].action_history[event_type]
        entry["steps"] = 0
        if "amount" in entry:
            entry["amount"] = amount
        if event_type == "give" and target_agent is not None:
            self._agent_stats[agent_id].give_to_steps[target_agent.value] = 0

    def get_steps_since_transfer(
        self, sender_id: m.AgentId, receiver_id: m.AgentId
    ) -> int:
        return self._agent_stats[sender_id].give_to_steps[receiver_id.value]

    def decrement_pending_requests(self, agent_id: m.AgentId) -> None:
        stats = self._agent_stats[agent_id]
        if stats.pending_request_count > 0:
            stats.pending_request_count -= 1

    def set_pending_request_count(self, agent_id: m.AgentId, count: int) -> None:
        self._agent_stats[agent_id].pending_request_count = count

    def add_pending_request_count(self, agent_id: m.AgentId, count: int) -> None:
        self._agent_stats[agent_id].pending_request_count += count

    def get_pending_request_count(self, agent_id: m.AgentId) -> int:
        return self._agent_stats[agent_id].pending_request_count

    def get_steps_since(
        self, agent_id: m.AgentId, event_type: sm.ActionHistoryKey
    ) -> int:
        return self._agent_stats[agent_id].action_history[event_type]["steps"]

    def reset_for_next_interval(
        self,
        current_time: float,
        agents: Dict[m.AgentId, m.Agent],
    ):
        rates, _ = self._get_arrival_rate(agents, current_time)
        avg_qs, _ = self._get_avg_queue_length(agents)
        for agent_id, stats in self._agent_stats.items():
            if current_time == 0.0:
                stats.last_arrival_rate = None
                stats.last_avg_queue_lengths = None
                for mig in m.MIGProfile:
                    stats.mig_composite_latencies[mig.value].clear()
            else:
                stats.last_arrival_rate = rates[agent_id]
                stats.last_avg_queue_lengths = avg_qs[agent_id]
                stats.arrival_rate_history = (
                    rates[agent_id],
                ) + stats.arrival_rate_history[:-1]
            stats.queue_length_integrals = [0.0] * 7
            stats.running_requests_integrals = [0.0] * 7
            stats.interval_requests.clear()

        for agent_id, agent in agents.items():
            for eng in agent.engines:
                self._agent_stats[agent_id].interval_requests.extend(eng.waiting_queue)
                self._agent_stats[agent_id].interval_requests.extend(
                    eng.running_queue.all_requests
                )

            stats = self._agent_stats[agent_id]
            stats.last_queue_lengths = [0] * 7
            stats.last_running_request_counts = [0] * 7

            for e in agent.engines:
                if e.status == m.EngineStatus.BOOTING:
                    continue
                idx = 6 if e.is_permanent else e.mig_profile.idx
                stats.last_queue_lengths[idx] += len(e.waiting_queue)
                stats.last_running_request_counts[idx] += len(e.running_queue)

        self._last_queue_update_time = current_time

    def record_queue_length_advance(
        self, current_time: float, agents: Dict[m.AgentId, m.Agent]
    ):
        dt = current_time - self._last_queue_update_time
        if dt > 0:
            for agent_id in agents.keys():
                stats = self._agent_stats[agent_id]
                for i in range(7):
                    stats.queue_length_integrals[i] += stats.last_queue_lengths[i] * dt
                    stats.running_requests_integrals[i] += (
                        stats.last_running_request_counts[i] * dt
                    )

        self._last_queue_update_time = current_time
        for agent_id, agent in agents.items():
            stats = self._agent_stats[agent_id]
            stats.last_queue_lengths = [0] * 7
            stats.last_running_request_counts = [0] * 7
            for e in agent.engines:
                if e.status == m.EngineStatus.BOOTING:
                    continue
                idx = 6 if e.is_permanent else e.mig_profile.idx
                stats.last_queue_lengths[idx] += len(e.waiting_queue)
                stats.last_running_request_counts[idx] += len(e.running_queue)

    def register_arrival(self, request: m.Request):
        self._agent_stats[request.agent_id].interval_requests.append(request)

    def get_state(
        self,
        current_time: float,
        agents: Dict[m.AgentId, m.Agent],
        gpu_current_state: Dict[int, int],
    ) -> m.EnvironmentStateData:
        # 1. Collect all metrics via (normalized) getters
        arrival_rate = self._get_normalized_arrival_rate(agents, current_time)
        arrival_rate_trend = self._get_arrival_rate_trend(agents, current_time)
        predicted_arrival_rate = {
            aid: arrival_rate[aid] * (1 + arrival_rate_trend[aid])
            for aid in agents.keys()
        }
        avg_queue_length = self._get_normalized_avg_queue_length(agents)
        avg_queue_length_trend = self._get_avg_queue_length_trend(agents)

        run_reqs = self._get_avg_running_requests(agents)
        kv = self._get_kv_cache_utilization(agents)
        latent_proportions, raw_latent_totals = self._get_avg_composite_latency(agents)
        vram = self._get_total_vram_ratio(agents)
        sz = self._get_total_sm_ratio(agents)

        mig_profile_id_onehot = self._get_mig_profile_id_onehot(gpu_current_state)
        ownership_grid = self._get_ownership_grid(gpu_current_state, agents)

        ratios = self._calculate_global_agent_ratios(
            arrival_rate,
            avg_queue_length,
            run_reqs,
            kv,
            raw_latent_totals,
            vram,
            sz,
        )

        # 2. Create the state dictionary
        state_data: m.EnvironmentStateData = {
            "arrival_rate": arrival_rate,
            "predicted_arrival_rate": predicted_arrival_rate,
            "arrival_rate_history": self._get_normalized_arrival_rate_history(agents),
            "avg_queue_length": avg_queue_length,
            "avg_queue_length_trend": avg_queue_length_trend,
            "avg_running_requests": run_reqs,
            "kv_cache_utilization": kv,
            "avg_composite_latency": latent_proportions,
            "mig_profile_id_onehot": mig_profile_id_onehot,
            "ownership_grid": ownership_grid,
            "agent_owns_mig": self._get_agent_owns_mig(agents),
            "mig_geometry": self._get_mig_geometry(agents),
            "current_budget": self._get_current_budget(),
            "downtime_ratio": self._get_downtime_ratio(),
            "total_sm_ratio": sz,
            "total_vram_ratio": vram,
            "recovery_flag": self._reconfig_flag,
            "last_split": self._get_last_split(agents),
            "last_merge": self._get_last_merge(agents),
            "last_give": self._get_last_give(agents),
            "last_receive": self._get_last_receive(agents),
            "last_give_amount": self._get_last_give_amount(agents),
            "last_receive_amount": self._get_last_receive_amount(agents),
            "requests": self.interval_requests,
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
        if len(agents) == 3:
            state_data["give_to_steps"] = self._get_give_to_steps(agents)
        return state_data

    def _calculate_global_agent_ratios(
        self,
        rates: Dict[m.AgentId, float],
        avg_qs: Dict[m.AgentId, Tuple[float, ...]],
        run_reqs: Dict[m.AgentId, Tuple[float, ...]],
        kv: Dict[m.AgentId, Tuple[float, ...]],
        raw_latent_totals: Dict[m.AgentId, float],
        vram: Dict[m.AgentId, float],
        sz: Dict[m.AgentId, float],
    ) -> Dict[sm.AgentRatioKeys, tuple[float, ...]]:
        def get_total(data_dict: Dict[m.AgentId, Any], aid: m.AgentId):
            val = data_dict[aid]
            return sum(val) if isinstance(val, (tuple, list)) else float(val)  # type: ignore

        metrics: Dict[sm.AgentRatioKeys, Dict[m.AgentId, Any]] = {
            "agent_arrival_rate_ratio": rates,
            "agent_avg_queue_len_ratio": avg_qs,
            "agent_avg_running_req_ratio": run_reqs,
            "agent_avg_kv_cache_ratio": kv,
            "agent_vram_ratio": vram,
            "agent_sm_ratio": sz,
        }

        ratios: Dict[sm.AgentRatioKeys, tuple[float, ...]] = {}
        epsilon = 1e-6
        # Sort agents by enum value for deterministic pairing order
        # Pairs are formed as (Agent i, Agent j) where i < j
        agents_ordered = sorted(rates.keys(), key=lambda a: a.value)

        for key, data in metrics.items():
            pair_ratios = []
            for i in range(len(agents_ordered)):
                for j in range(i + 1, len(agents_ordered)):
                    c_val = get_total(data, agents_ordered[i])
                    r_val = get_total(data, agents_ordered[j])
                    pair_ratios.append((c_val - r_val) / (c_val + r_val + epsilon))
            ratios[key] = tuple(pair_ratios) if len(pair_ratios) > 1 else pair_ratios[0]

        # 4. Special Case: Latency (Log-Normalized symmetric ratio)
        max_exp = TRAINING_CONFIG.norm_avg_composite_latency
        denom = math.log10(1 + max_exp)
        latency_ratios = []
        for i in range(len(agents_ordered)):
            for j in range(i + 1, len(agents_ordered)):
                l_c_raw = raw_latent_totals[agents_ordered[i]]
                l_r_raw = raw_latent_totals[agents_ordered[j]]
                l_c_norm = math.log10(1 + l_c_raw) / denom
                l_r_norm = math.log10(1 + l_r_raw) / denom
                latency_ratios.append(
                    (l_c_norm - l_r_norm) / (l_c_norm + l_r_norm + epsilon)
                )

        ratios["agent_avg_composite_latency_ratio"] = (
            tuple(latency_ratios) if len(latency_ratios) > 1 else latency_ratios[0]
        )

        return ratios

    def _get_arrival_rate(
        self, agents: Dict[m.AgentId, m.Agent], current_time: float
    ) -> Tuple[Dict[m.AgentId, float], Dict[m.AgentId, float]]:
        """Returns raw (un-normalized) arrival rates and trends. Used internally."""
        rates: Dict[m.AgentId, float] = {}
        trends: Dict[m.AgentId, float] = {}
        start_time = current_time - TRAINING_CONFIG.action_interval
        for agent_id in agents.keys():
            stats = self._agent_stats[agent_id]
            arr = [
                r.arrival_time
                for r in stats.interval_requests
                if r.arrival_time >= start_time
            ]
            current_rate = len(arr) / TRAINING_CONFIG.action_interval
            rates[agent_id] = current_rate
            if stats.last_arrival_rate is None or stats.last_arrival_rate == 0.0:
                trends[agent_id] = 1.0 if current_rate > 0.0 else 0.0
            else:
                trends[agent_id] = (
                    current_rate - stats.last_arrival_rate
                ) / stats.last_arrival_rate
        return rates, trends

    def _get_normalized_arrival_rate(
        self, agents: Dict[m.AgentId, m.Agent], current_time: float
    ) -> Dict[m.AgentId, float]:
        rates, _ = self._get_arrival_rate(agents, current_time)
        return {k: v / TRAINING_CONFIG.norm_arrival_rate for k, v in rates.items()}

    def _get_arrival_rate_trend(
        self, agents: Dict[m.AgentId, m.Agent], current_time: float
    ) -> Dict[m.AgentId, float]:
        _, trends = self._get_arrival_rate(agents, current_time)
        return trends

    def _get_normalized_arrival_rate_history(
        self, agents: Dict[m.AgentId, m.Agent]
    ) -> Dict[m.AgentId, Tuple[float, ...]]:
        return {
            aid: tuple(
                r / TRAINING_CONFIG.norm_arrival_rate
                for r in stats.arrival_rate_history
            )
            for aid, stats in self._agent_stats.items()
            if aid in agents
        }

    def _get_avg_queue_length(
        self, agents: Dict[m.AgentId, m.Agent]
    ) -> Tuple[Dict[m.AgentId, Tuple[float, ...]], Dict[m.AgentId, Tuple[float, ...]]]:
        """Returns raw (un-normalized) avg queue lengths and trends. Used internally."""
        avg_q: Dict[m.AgentId, Tuple[float, ...]] = {}
        trends: Dict[m.AgentId, Tuple[float, ...]] = {}
        for agent_id in agents.keys():
            stats = self._agent_stats[agent_id]
            current_avgs: List[float] = []
            current_trends: List[float] = []
            for i in range(7):
                integral = stats.queue_length_integrals[i]
                current_avg = (
                    integral / TRAINING_CONFIG.action_interval
                    if TRAINING_CONFIG.action_interval > 0
                    else 0.0
                )
                current_avgs.append(current_avg)

                if stats.last_avg_queue_lengths is None:
                    current_trends.append(0.0)
                elif (
                    i < len(stats.last_avg_queue_lengths)
                    and stats.last_avg_queue_lengths[i] > 0.0
                ):
                    trd = (
                        current_avg - stats.last_avg_queue_lengths[i]
                    ) / stats.last_avg_queue_lengths[i]
                    trd = min(trd, TRAINING_CONFIG.queue_length_trend_clamp)  # clamp
                    trd = max(trd, -TRAINING_CONFIG.queue_length_trend_clamp)
                    current_trends.append(trd)
                else:
                    current_trends.append(1.0 if current_avg > 0.0 else 0.0)
            avg_q[agent_id] = tuple(current_avgs)
            trends[agent_id] = tuple(current_trends)
        return avg_q, trends

    def _get_normalized_avg_queue_length(
        self, agents: Dict[m.AgentId, m.Agent]
    ) -> Dict[m.AgentId, Tuple[float, ...]]:
        avg_qs, _ = self._get_avg_queue_length(agents)
        max_expected = TRAINING_CONFIG.norm_avg_queue_length
        denom = math.log10(1 + max_expected)

        res: Dict[m.AgentId, Tuple[float, ...]] = {}
        for aid, components in avg_qs.items():
            res[aid] = tuple(math.log10(1 + q) / denom for q in components)
        return res

    def _get_avg_queue_length_trend(
        self, agents: Dict[m.AgentId, m.Agent]
    ) -> Dict[m.AgentId, Tuple[float, ...]]:
        _, trends = self._get_avg_queue_length(agents)
        return trends

    def _get_avg_running_requests(
        self, agents: Dict[m.AgentId, m.Agent]
    ) -> Dict[m.AgentId, Tuple[float, ...]]:
        avg_run: Dict[m.AgentId, Tuple[float, ...]] = {}
        for agent_id in agents.keys():
            stats = self._agent_stats[agent_id]
            avgs: List[float] = []
            for i in range(7):
                integral = stats.running_requests_integrals[i]
                raw = (
                    integral / TRAINING_CONFIG.action_interval
                    if TRAINING_CONFIG.action_interval > 0
                    else 0.0
                )
                avgs.append(raw / TRAINING_CONFIG.norm_avg_running_requests)
            avg_run[agent_id] = tuple(avgs)
        return avg_run

    def _get_kv_cache_utilization(
        self, agents: Dict[m.AgentId, m.Agent]
    ) -> Dict[m.AgentId, Tuple[float, ...]]:
        num_logical = len(m.MIGProfile)
        result: Dict[m.AgentId, Tuple[float, ...]] = {}
        for agent_id, agent in agents.items():
            util_sums = [0.0] * num_logical
            counts = [0] * num_logical
            perm_util_sum = 0.0
            perm_count = 0
            for e in agent.engines:
                if e.status == m.EngineStatus.BOOTING:
                    continue
                if e.is_permanent:
                    perm_util_sum += e.current_kv_utilization
                    perm_count += 1
                else:
                    idx = e.mig_profile.idx
                    util_sums[idx] += e.current_kv_utilization
                    counts[idx] += 1
            normal_avgs = tuple(
                s / c if c > 0 else 0.0 for s, c in zip(util_sums, counts)
            )
            perm_avg = perm_util_sum / perm_count if perm_count > 0 else 0.0
            result[agent_id] = normal_avgs + (perm_avg,)
        return result

    def _get_avg_composite_latency(
        self, agents: Dict[m.AgentId, m.Agent]
    ) -> Tuple[
        Dict[m.AgentId, Tuple[float, ...]],
        Dict[m.AgentId, float],
    ]:
        result: Dict[m.AgentId, Tuple[float, ...]] = {}
        raw_totals: Dict[m.AgentId, float] = {}
        w_t = TRAINING_CONFIG.w("ttft")
        w_p = TRAINING_CONFIG.w("tpot")

        for agent_id, agent in agents.items():
            stats = self._agent_stats[agent_id]

            # Clear and repopulate from the last 100 valid completed requests
            for mig in m.MIGProfile:
                stats.mig_composite_latencies[mig.value].clear()

            perm_latencies: deque[float] = deque(maxlen=100)

            visited = 0
            for req in reversed(agent.completed_requests):
                if visited >= 100:
                    break
                if req.serving_engine is None:
                    continue
                visited += 1
                ttft = (req.first_token_time or req.arrival_time) - req.arrival_time
                tpot = (
                    req.decode_time / req.generated_tokens
                    if req.generated_tokens > 0
                    else 0.0
                )
                composite = w_t * ttft + w_p * tpot
                if req.serving_engine.is_permanent:
                    perm_latencies.append(composite)
                else:
                    stats.mig_composite_latencies[
                        req.serving_engine.mig_profile.idx
                    ].append(composite)

            # Raw averages for each MIG profile slot + permanent slot
            num_logical = len(m.MIGProfile)
            raw_avgs = [0.0] * (num_logical + 1)
            for mig in m.MIGProfile:
                q = stats.mig_composite_latencies[mig.value]
                if len(q) > 0:
                    raw_avgs[mig.value] = sum(q) / len(q)
            raw_avgs[num_logical] = (
                sum(perm_latencies) / len(perm_latencies) if perm_latencies else 0.0
            )

            # Normalize to percentages (proportion of total)
            total = sum(raw_avgs)
            raw_totals[agent_id] = total
            if total > 0.0:
                pct_avgs = tuple(v / total for v in raw_avgs)
            else:
                pct_avgs = tuple(0.0 for _ in raw_avgs)

            result[agent_id] = pct_avgs
        return result, raw_totals

    def _get_agent_owns_mig(
        self, agents: Dict[m.AgentId, m.Agent]
    ) -> Dict[m.AgentId, Tuple[float, ...]]:
        """Per-agent count of each MIG profile (normalized), used in the observation."""
        result: Dict[m.AgentId, Tuple[float, ...]] = {}
        divisor = TRAINING_CONFIG.norm_mig_geometry
        num_logical = len(m.MIGProfile)
        for agent_id, agent in agents.items():
            counts = [0] * num_logical
            for e in agent.engines:
                if e.status != m.EngineStatus.BOOTING and not e.is_permanent:
                    counts[e.mig_profile.idx] += 1
            result[agent_id] = tuple(c / divisor for c in counts)
        return result

    def _get_mig_geometry(
        self, agents: Dict[m.AgentId, m.Agent]
    ) -> Dict[int, List[float]]:
        """GPU-keyed normalized MIG size per agent: {gpu_idx: [size_agent0, size_agent1, ...]}.

        Permanent engines are excluded.
        Agents are ordered by their enum value for deterministic indexing.
        Values are normalized by norm_mig_geometry.
        """
        agents_ordered = sorted(agents.keys(), key=lambda a: a.value)
        raw: Dict[int, List[int]] = {}
        for agent_id, agent in agents.items():
            for e in agent.engines:
                if e.is_permanent:
                    continue  # skip permanent GPU
                gpu = e.gpu
                if gpu not in raw:
                    raw[gpu] = [0] * len(agents_ordered)
                agent_idx = agents_ordered.index(agent_id)
                raw[gpu][agent_idx] += e.mig_profile.size
        divisor = TRAINING_CONFIG.norm_mig_geometry
        return {gpu: [s / divisor for s in sizes] for gpu, sizes in raw.items()}

    def _get_total_sm_ratio(
        self, agents: Dict[m.AgentId, m.Agent]
    ) -> Dict[m.AgentId, float]:
        ratio: Dict[m.AgentId, float] = {}
        for agent_id, agent in agents.items():
            total_size = sum(e.mig_profile.size for e in agent.engines)
            ratio[agent_id] = total_size / TRAINING_CONFIG.norm_total_sm_ratio
        return ratio

    def _get_total_vram_ratio(
        self, agents: Dict[m.AgentId, m.Agent]
    ) -> Dict[m.AgentId, float]:
        ratio: Dict[m.AgentId, float] = {}
        for agent_id, agent in agents.items():
            total_vram = sum(e.mig_profile.vram for e in agent.engines)
            ratio[agent_id] = total_vram / TRAINING_CONFIG.norm_total_vram_ratio
        return ratio

    def _get_current_budget(self) -> float:
        return self._current_budget / TRAINING_CONFIG.norm_current_budget

    def _get_downtime_ratio(self) -> float:
        return self._last_action_downtime / TRAINING_CONFIG.action_interval

    def _get_last_split(
        self, agents: Dict[m.AgentId, m.Agent]
    ) -> Dict[m.AgentId, float]:
        norm = float(TRAINING_CONFIG.action_cooldown)
        return {
            aid: min(self._agent_stats[aid].action_history["split"]["steps"], norm)
            / norm
            for aid in agents.keys()
        }

    def _get_last_merge(
        self, agents: Dict[m.AgentId, m.Agent]
    ) -> Dict[m.AgentId, float]:
        norm = float(TRAINING_CONFIG.action_cooldown)
        return {
            aid: min(self._agent_stats[aid].action_history["merge"]["steps"], norm)
            / norm
            for aid in agents.keys()
        }

    def _get_last_give(
        self, agents: Dict[m.AgentId, m.Agent]
    ) -> Dict[m.AgentId, float]:
        norm = float(TRAINING_CONFIG.action_cooldown)
        return {
            aid: min(self._agent_stats[aid].action_history["give"]["steps"], norm)
            / norm
            for aid in agents.keys()
        }

    def _get_last_receive(
        self, agents: Dict[m.AgentId, m.Agent]
    ) -> Dict[m.AgentId, float]:
        norm = float(TRAINING_CONFIG.action_cooldown)
        return {
            aid: min(self.get_steps_since(aid, "receive"), norm) / norm
            for aid in agents.keys()
        }

    def _get_give_to_steps(self, agents: Dict[m.AgentId, m.Agent]) -> Any:
        norm = float(TRAINING_CONFIG.action_cooldown)
        ordered_targets = sorted(list(m.AgentId), key=lambda x: x.value)

        result = {}
        for aid in agents.keys():
            steps = []
            for target in ordered_targets:
                val = self._agent_stats[aid].give_to_steps[target.value]
                steps.append(min(val, norm) / norm)
            result[aid] = tuple(steps)
        return result

    def _get_last_give_amount(
        self, agents: Dict[m.AgentId, m.Agent]
    ) -> Dict[m.AgentId, float]:
        norm_action = float(TRAINING_CONFIG.action_cooldown)
        norm_amount = TRAINING_CONFIG.norm_vram_transfer_amount
        res: Dict[m.AgentId, float] = {}
        for aid in agents.keys():
            entry = self._agent_stats[aid].action_history["give"]
            steps = entry["steps"]
            norm_val = min(steps, norm_action) / norm_action
            if norm_val == 1.0 or steps == 0:
                res[aid] = 0.0
            else:
                res[aid] = entry["amount"] / norm_amount
        return res

    def _get_last_receive_amount(
        self, agents: Dict[m.AgentId, m.Agent]
    ) -> Dict[m.AgentId, float]:
        norm_action = float(TRAINING_CONFIG.action_cooldown)
        norm_amount = TRAINING_CONFIG.norm_vram_transfer_amount
        res: Dict[m.AgentId, float] = {}
        for aid in agents.keys():
            entry = self._agent_stats[aid].action_history["receive"]
            steps = entry["steps"]
            norm_val = min(steps, norm_action) / norm_action
            if norm_val == 1.0 or steps == 0:
                res[aid] = 0.0
            else:
                res[aid] = entry["amount"] / norm_amount
        return res

    def _get_mig_profile_id_onehot(
        self, gpu_current_state: Dict[int, int]
    ) -> Dict[int, List[float]]:
        onehot_dict = {}
        for gpu_id, state_id in gpu_current_state.items():
            vec = [0.0] * 15
            if state_id in STATE_ID_MAP:
                vec[STATE_ID_MAP[state_id]] = 1.0
            onehot_dict[gpu_id] = vec
        return onehot_dict

    def _get_ownership_grid(
        self, gpu_current_state: Dict[int, int], agents: Dict[m.AgentId, m.Agent]
    ) -> Dict[int, List[int]]:
        grid_dict = {}
        all_engines = [e for agent in agents.values() for e in agent.engines]
        # We need to find the owner of each of the 7 slices for each GPU.
        for gpu_id, state_id in gpu_current_state.items():
            grid = [0] * NUM_MIG_SLICES
            # Iterate through all engines on this GPU
            for eng in all_engines:
                if eng.gpu == gpu_id:
                    owner_val = 1 if eng.owner.agent_id == m.AgentId.CODING else 2
                    if state_id in SLICE_MAPPING and eng.mig_index != -1:
                        slices = SLICE_MAPPING[state_id][eng.mig_index]
                        for s_idx in slices:
                            grid[s_idx] = owner_val
            grid_dict[gpu_id] = grid
        return grid_dict
