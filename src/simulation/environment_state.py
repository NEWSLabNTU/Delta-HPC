from collections import defaultdict
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
    arrival_rate_history: Tuple[float, ...] = field(
        default_factory=lambda: tuple(
            [0.0] * TRAINING_CONFIG.arrival_rate_history_length
        )
    )
    interval_requests: List[m.Request] = field(default_factory=list)


class EnvironmentStateImpl(m.EnvironmentState):
    def __init__(self):
        self._agent_stats: Dict[m.AgentId, AgentStats] = defaultdict(AgentStats)
        self._last_queue_update_time: float = 0.0
        self._reconfig_flag: bool = False
        self._current_budget = TRAINING_CONFIG.reconfig_budget

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

    def refresh_budget(self) -> None:
        self._current_budget = TRAINING_CONFIG.reconfig_budget

    def reset_for_next_interval(
        self,
        current_time: float,
        agents: Dict[m.AgentId, m.Agent],
    ):
        rates, _ = self._get_arrival_rate(agents, current_time)
        for agent_id, stats in self._agent_stats.items():
            if current_time == 0.0:
                stats.last_arrival_rate = None
            else:
                stats.last_arrival_rate = rates[agent_id]
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
        rates, trends = self._get_arrival_rate(agents, current_time)
        return {
            "arrival_rate": rates,
            "arrival_rate_trend": trends,
            "arrival_rate_history": {
                aid: stats.arrival_rate_history
                for aid, stats in self._agent_stats.items()
            },
            "avg_queue_length": self._get_avg_queue_length(agents),
            "avg_running_requests": self._get_avg_running_requests(agents),
            "queue_delta": self._get_queue_delta(agents),
            "p99_ttft": self._get_p99_ttft(agents, current_time),
            "avg_tpot": self._get_avg_tpot(agents, current_time),
            "kv_cache_utilization": self._get_kv_cache_utilization(engines),
            "current_mig_profile": self._get_mig_config_encoding(engines),
            "current_budget": self._current_budget,
            "recovery_flag": self._reconfig_flag,
            "requests": {
                aid: self._agent_stats[aid].interval_requests for aid in agents.keys()
            },
        }

    def _get_arrival_rate(
        self, agents: Dict[m.AgentId, m.Agent], current_time: float
    ) -> Tuple[Dict[m.AgentId, float], Dict[m.AgentId, float]]:
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
            if stats.last_arrival_rate is not None:
                trends[agent_id] = current_rate - stats.last_arrival_rate
            else:
                trends[agent_id] = 0.0
        return rates, trends

    def _get_avg_queue_length(
        self, agents: Dict[m.AgentId, m.Agent]
    ) -> Dict[m.AgentId, float]:
        avg_q: Dict[m.AgentId, float] = {}
        for agent_id in agents.keys():
            integral = self._agent_stats[agent_id].queue_length_integral
            avg_q[agent_id] = (
                integral / TRAINING_CONFIG.action_interval
                if TRAINING_CONFIG.action_interval > 0
                else 0.0
            )
        return avg_q

    def _get_avg_running_requests(
        self, agents: Dict[m.AgentId, m.Agent]
    ) -> Dict[m.AgentId, float]:
        avg_run: Dict[m.AgentId, float] = {}
        for agent_id in agents.keys():
            integral = self._agent_stats[agent_id].running_requests_integral
            avg_run[agent_id] = (
                integral / TRAINING_CONFIG.action_interval
                if TRAINING_CONFIG.action_interval > 0
                else 0.0
            )
        return avg_run

    def _get_queue_delta(
        self, agents: Dict[m.AgentId, m.Agent]
    ) -> Dict[m.AgentId, int]:
        delta: Dict[m.AgentId, int] = {}
        for agent_id in agents.keys():
            start_q = self._agent_stats[agent_id].interval_start_queue_length
            end_q = self._agent_stats[agent_id].last_queue_length
            delta[agent_id] = end_q - start_q
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
                p99[agent_id] = ttfts[idx]
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
        self, engines: Dict[str, m.LLMEngine]
    ) -> Dict[int, List[float]]:
        num_profiles = len(m.MIGProfile)
        util_sums: Dict[int, List[float]] = defaultdict(lambda: [0.0] * num_profiles)
        counts: Dict[int, List[int]] = defaultdict(lambda: [0] * num_profiles)

        for engine in engines.values():
            if engine.status == m.EngineStatus.BOOTING:
                continue
            idx = engine.mig_profile.idx
            util_sums[engine.gpu][idx] += engine.current_kv_utilization
            counts[engine.gpu][idx] += 1

        result: Dict[int, List[float]] = {}
        for gpu, sums in util_sums.items():
            gpu_counts = counts[gpu]
            result[gpu] = [s / c if c > 0 else 0.0 for s, c in zip(sums, gpu_counts)]
        return result

    def _get_mig_config_encoding(
        self, engines: Dict[str, m.LLMEngine]
    ) -> Dict[int, m.MIGEncoding]:
        encoding: Dict[int, m.MIGEncoding] = defaultdict(m.MIGEncoding)
        for engine in engines.values():
            encoding[engine.gpu][engine.mig_profile.idx] += 1
        return encoding
