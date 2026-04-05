from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

import src.simulation.models as m
from src.training.config import TRAINING_CONFIG


@dataclass
class AgentStats:
    queue_length_integral: float = 0.0
    running_requests_integral: float = 0.0
    last_queue_length: int = 0
    last_running_requests: int = 0
    interval_start_queue_length: int = 0
    last_arrival_rate: Optional[float] = None
    last_avg_queue_length: Optional[float] = None
    arrival_rate_history: Tuple[float, ...] = field(
        default_factory=lambda: tuple(
            [0.0] * TRAINING_CONFIG.arrival_rate_history_length
        )
    )
    interval_requests: List[m.Request] = field(default_factory=list[m.Request])
    mig_composite_latencies: Dict[int, deque] = field(
        default_factory=lambda: {mig.idx: deque(maxlen=100) for mig in m.MIGProfile}
    )


class EnvironmentStateImpl(m.EnvironmentState):
    def __init__(self):
        self._agent_stats: Dict[m.AgentId, AgentStats] = defaultdict(AgentStats)
        self._last_queue_update_time: float = 0.0
        self._reconfig_flag: bool = False
        self._current_budget = TRAINING_CONFIG.reconfig_budget
        self._last_action_downtime: float = 0.0
        self._steps_since_split: Dict[m.AgentId, int] = {aid: 5 for aid in m.AgentId}
        self._steps_since_merge: Dict[m.AgentId, int] = {aid: 5 for aid in m.AgentId}

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
    def steps_since_split(self) -> Dict[m.AgentId, int]:
        return self._steps_since_split

    @steps_since_split.setter
    def steps_since_split(self, v: Dict[m.AgentId, int]) -> None:
        self._steps_since_split = v

    @property
    def steps_since_merge(self) -> Dict[m.AgentId, int]:
        return self._steps_since_merge

    @steps_since_merge.setter
    def steps_since_merge(self, v: Dict[m.AgentId, int]) -> None:
        self._steps_since_merge = v

    def refresh_budget(self) -> None:
        self._current_budget = TRAINING_CONFIG.reconfig_budget

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
                stats.last_avg_queue_length = None
                for mig in m.MIGProfile:
                    stats.mig_composite_latencies[mig.idx].clear()
            else:
                stats.last_arrival_rate = rates[agent_id]
                stats.last_avg_queue_length = avg_qs[agent_id]
                stats.arrival_rate_history = (
                    rates[agent_id],
                ) + stats.arrival_rate_history[:-1]
            stats.queue_length_integral = 0.0
            stats.running_requests_integral = 0.0
            stats.interval_requests.clear()

        for agent_id, agent in agents.items():
            for eng in agent.engines:
                self._agent_stats[agent_id].interval_requests.extend(eng.waiting_queue)
                self._agent_stats[agent_id].interval_requests.extend(
                    eng.running_queue.all_requests
                )

            q_len = sum(len(e.waiting_queue) for e in agent.engines)
            run_len = sum(len(e.running_queue) for e in agent.engines)

            stats = self._agent_stats[agent_id]
            stats.interval_start_queue_length = q_len
            stats.last_queue_length = q_len
            stats.last_running_requests = run_len

        self._last_queue_update_time = current_time

    def record_queue_length_advance(
        self, current_time: float, agents: Dict[m.AgentId, m.Agent]
    ):
        dt = current_time - self._last_queue_update_time
        if dt > 0:
            for agent_id in agents.keys():
                stats = self._agent_stats[agent_id]
                stats.queue_length_integral += stats.last_queue_length * dt
                stats.running_requests_integral += stats.last_running_requests * dt

        self._last_queue_update_time = current_time
        for agent_id, agent in agents.items():
            run_len = sum(len(e.running_queue) for e in agent.engines)
            q_len = sum(len(e.waiting_queue) for e in agent.engines)

            stats = self._agent_stats[agent_id]
            stats.last_queue_length = q_len
            stats.last_running_requests = run_len

    def register_arrival(self, request: m.Request):
        self._agent_stats[request.agent_id].interval_requests.append(request)

    def get_state(
        self,
        current_time: float,
        agents: Dict[m.AgentId, m.Agent],
        engines: Dict[str, m.LLMEngine],
    ) -> m.EnvironmentStateData:
        return {
            "arrival_rate": self._get_normalized_arrival_rate(agents, current_time),
            "arrival_rate_trend": self._get_arrival_rate_trend(agents, current_time),
            "arrival_rate_history": self._get_normalized_arrival_rate_history(agents),
            "avg_queue_length": self._get_normalized_avg_queue_length(agents),
            "avg_queue_length_trend": self._get_avg_queue_length_trend(agents),
            "avg_running_requests": self._get_avg_running_requests(agents),
            "queue_delta": self._get_queue_delta(agents),
            "p99_ttft": self._get_p99_ttft(agents, current_time),
            "avg_tpot": self._get_avg_tpot(agents, current_time),
            "kv_cache_utilization": self._get_kv_cache_utilization(agents),
            "avg_composite_latency": self._get_avg_composite_latency(agents),
            "n_mig_instance": self._get_n_mig_instance(agents),
            "mig_geometry": self._get_mig_geometry(agents),
            "current_budget": self._get_current_budget(),
            "downtime_ratio": self._get_downtime_ratio(),
            "mig_total_ratio": self._get_mig_total_ratio(agents),
            "recovery_flag": self._reconfig_flag,
            "last_split": self._get_last_split(agents),
            "last_merge": self._get_last_merge(agents),
            "requests": {
                aid: self._agent_stats[aid].interval_requests for aid in agents.keys()
            },
        }

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
                if r.agent_id == agent_id and r.arrival_time >= start_time
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
    ) -> Tuple[Dict[m.AgentId, float], Dict[m.AgentId, float]]:
        """Returns raw (un-normalized) avg queue lengths and trends. Used internally."""
        avg_q: Dict[m.AgentId, float] = {}
        trends: Dict[m.AgentId, float] = {}
        for agent_id in agents.keys():
            stats = self._agent_stats[agent_id]
            integral = stats.queue_length_integral
            current_avg = (
                integral / TRAINING_CONFIG.action_interval
                if TRAINING_CONFIG.action_interval > 0
                else 0.0
            )
            avg_q[agent_id] = current_avg

            if stats.last_avg_queue_length is None:
                trends[agent_id] = 0.0
            elif stats.last_avg_queue_length > 0.0:
                trends[agent_id] = (
                    current_avg - stats.last_avg_queue_length
                ) / stats.last_avg_queue_length
            else:
                trends[agent_id] = 1.0 if current_avg > 0.0 else 0.0
        return avg_q, trends

    def _get_normalized_avg_queue_length(
        self, agents: Dict[m.AgentId, m.Agent]
    ) -> Dict[m.AgentId, float]:
        avg_qs, _ = self._get_avg_queue_length(agents)
        return {k: v / TRAINING_CONFIG.norm_avg_queue_length for k, v in avg_qs.items()}

    def _get_avg_queue_length_trend(
        self, agents: Dict[m.AgentId, m.Agent]
    ) -> Dict[m.AgentId, float]:
        _, trends = self._get_avg_queue_length(agents)
        return trends

    def _get_avg_running_requests(
        self, agents: Dict[m.AgentId, m.Agent]
    ) -> Dict[m.AgentId, float]:
        avg_run: Dict[m.AgentId, float] = {}
        for agent_id in agents.keys():
            integral = self._agent_stats[agent_id].running_requests_integral
            raw = (
                integral / TRAINING_CONFIG.action_interval
                if TRAINING_CONFIG.action_interval > 0
                else 0.0
            )
            avg_run[agent_id] = raw / TRAINING_CONFIG.norm_avg_running_requests
        return avg_run

    def _get_queue_delta(
        self, agents: Dict[m.AgentId, m.Agent]
    ) -> Dict[m.AgentId, float]:
        delta: Dict[m.AgentId, float] = {}
        for agent_id in agents.keys():
            start_q = self._agent_stats[agent_id].interval_start_queue_length
            end_q = self._agent_stats[agent_id].last_queue_length
            delta[agent_id] = (end_q - start_q) / TRAINING_CONFIG.norm_queue_delta
        return delta

    def _get_p99_ttft(
        self, agents: Dict[m.AgentId, m.Agent], current_time: float
    ) -> Dict[m.AgentId, float]:
        p99: Dict[m.AgentId, float] = {}
        start_time = current_time - TRAINING_CONFIG.action_interval
        for agent_id in agents.keys():
            ttfts: List[float] = []
            for r in self._agent_stats[agent_id].interval_requests:
                if (
                    r.agent_id == agent_id
                    and r.first_token_time is not None
                    and r.first_token_time >= start_time
                ):
                    ttfts.append(r.first_token_time - r.arrival_time)

            if not ttfts:
                p99[agent_id] = 0.0
            else:
                ttfts.sort()
                idx = int(0.99 * len(ttfts))
                idx = min(idx, len(ttfts) - 1)
                p99[agent_id] = ttfts[idx] / TRAINING_CONFIG.norm_p99_ttft
        return p99

    def _get_avg_tpot(
        self, agents: Dict[m.AgentId, m.Agent], current_time: float
    ) -> Dict[m.AgentId, float]:
        avg_tpot: Dict[m.AgentId, float] = {}
        start_time = current_time - TRAINING_CONFIG.action_interval
        for agent_id in agents.keys():
            tpots: List[float] = []
            for r in self._agent_stats[agent_id].interval_requests:
                if (
                    r.agent_id == agent_id
                    and r.start_time is not None
                    and r.finish_time is not None
                    and r.start_time >= start_time
                    and r.finish_time <= current_time
                    and r.completion_tokens > 0
                ):
                    duration = r.finish_time - r.start_time
                    if duration > 0:
                        tpots.append(duration / r.completion_tokens)

            avg_tpot[agent_id] = sum(tpots) / len(tpots) if tpots else 0.0
        return avg_tpot

    def _get_kv_cache_utilization(
        self, agents: Dict[m.AgentId, m.Agent]
    ) -> Dict[m.AgentId, Tuple[float, float, float, float, float, float]]:
        num_profiles = len(m.MIGProfile)
        result: Dict[m.AgentId, Tuple[float, float, float, float, float, float]] = {}
        for agent_id, agent in agents.items():
            util_sums = [0.0] * num_profiles
            counts = [0] * num_profiles
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
            result[agent_id] = normal_avgs + (perm_avg,)  # type: ignore
        return result

    def _get_avg_composite_latency(
        self, agents: Dict[m.AgentId, m.Agent]
    ) -> Dict[m.AgentId, Tuple[float, float, float, float, float, float]]:
        result: Dict[m.AgentId, Tuple[float, float, float, float, float, float]] = {}
        w_t = TRAINING_CONFIG.w("ttft")
        w_p = TRAINING_CONFIG.w("tpot")

        for agent_id, agent in agents.items():
            stats = self._agent_stats[agent_id]

            # Clear and repopulate from the bounded deque (maxlen=100)
            for mig in m.MIGProfile:
                stats.mig_composite_latencies[mig.idx].clear()

            perm_latencies: deque = deque(maxlen=100)

            for req in agent.completed_requests:
                if req.serving_engine is None:
                    continue
                mig = req.serving_engine.mig_profile
                ttft = (req.first_token_time or req.arrival_time) - req.arrival_time
                tpot = (
                    req.decode_time / req.generated_tokens
                    if req.generated_tokens > 0
                    else 0.0
                )
                q_j = TRAINING_CONFIG.qf(mig)
                composite = (w_t * ttft + w_p * tpot) / q_j
                if req.serving_engine.is_permanent:
                    perm_latencies.append(composite)
                else:
                    stats.mig_composite_latencies[mig.idx].append(composite)

            # Raw averages for each MIG profile slot + permanent slot
            raw_avgs = [0.0] * (len(m.MIGProfile) + 1)
            for mig in m.MIGProfile:
                q = stats.mig_composite_latencies[mig.idx]
                if len(q) > 0:
                    raw_avgs[mig.idx] = sum(q) / len(q)
            raw_avgs[len(m.MIGProfile)] = (
                sum(perm_latencies) / len(perm_latencies) if perm_latencies else 0.0
            )

            # Normalize to percentages (proportion of total)
            total = sum(raw_avgs)
            if total > 0.0:
                pct_avgs = tuple(v / total for v in raw_avgs)
            else:
                pct_avgs = tuple(0.0 for _ in raw_avgs)

            result[agent_id] = pct_avgs  # type: ignore
        return result

    def _get_n_mig_instance(
        self, agents: Dict[m.AgentId, m.Agent]
    ) -> Dict[m.AgentId, int]:
        stats: Dict[m.AgentId, int] = {}
        for agent_id, agent in agents.items():
            instances = sum(
                1 for e in agent.engines if e.status != m.EngineStatus.BOOTING
            )
            stats[agent_id] = instances / TRAINING_CONFIG.norm_mig_geometry
        return stats

    def _get_mig_geometry(
        self, agents: Dict[m.AgentId, m.Agent]
    ) -> Dict[m.AgentId, Tuple[float, float, float, float, float]]:
        result: Dict[m.AgentId, Tuple[float, float, float, float, float]] = {}
        divisor = TRAINING_CONFIG.norm_mig_geometry
        for agent_id, agent in agents.items():
            counts = [0] * len(m.MIGProfile)
            for e in agent.engines:
                if e.status != m.EngineStatus.BOOTING:
                    counts[e.mig_profile.idx] += 1
            result[agent_id] = tuple(c / divisor for c in counts)  # type: ignore
        return result

    def _get_mig_total_ratio(
        self, agents: Dict[m.AgentId, m.Agent]
    ) -> Dict[m.AgentId, float]:
        ratio: Dict[m.AgentId, float] = {}
        for agent_id, agent in agents.items():
            total_size = sum(e.mig_profile.size for e in agent.engines)
            ratio[agent_id] = total_size / TRAINING_CONFIG.norm_mig_total_ratio
        return ratio

    def _get_current_budget(self) -> float:
        return self._current_budget / TRAINING_CONFIG.norm_current_budget

    def _get_downtime_ratio(self) -> float:
        return self._last_action_downtime / TRAINING_CONFIG.action_interval

    def _get_last_split(
        self, agents: Dict[m.AgentId, m.Agent]
    ) -> Dict[m.AgentId, float]:
        norm = TRAINING_CONFIG.norm_last_action
        return {
            aid: min(self._steps_since_split[aid], norm) / norm for aid in agents.keys()
        }

    def _get_last_merge(
        self, agents: Dict[m.AgentId, m.Agent]
    ) -> Dict[m.AgentId, float]:
        norm = TRAINING_CONFIG.norm_last_action
        return {
            aid: min(self._steps_since_merge[aid], norm) / norm for aid in agents.keys()
        }
