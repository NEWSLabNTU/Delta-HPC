from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List

import src.simulation.models as m
from src.training.config import TRAINING_CONFIG


@dataclass
class AgentStats:
    queue_length_integral: float = 0.0
    running_requests_integral: float = 0.0
    last_queue_length: int = 0
    last_running_requests: int = 0
    interval_start_queue_length: int = 0


class EnvironmentStateImpl(m.EnvironmentState):
    def __init__(self):
        self._agent_stats: Dict[m.AgentId, AgentStats] = defaultdict(AgentStats)
        self._last_queue_update_time: float = 0.0
        self._reconfig_in_interval: bool = False
        self._interval_requests: List[m.Request] = []
        self._current_budget = TRAINING_CONFIG.reconfig_budget

    @property
    def current_budget(self) -> float:
        return self._current_budget

    @current_budget.setter
    def current_budget(self, v: float) -> None:
        self._current_budget = v

    def refresh_budget(self) -> None:
        self._current_budget = TRAINING_CONFIG.reconfig_budget

    def reset_for_next_interval(
        self,
        current_time: float,
        agents: Dict[m.AgentId, m.Agent],
    ):
        for stats in self._agent_stats.values():
            stats.queue_length_integral = 0.0
            stats.running_requests_integral = 0.0

        self._interval_requests = []

        for agent_id, agent in agents.items():
            for e in agent.engines:
                self._interval_requests.extend(e.running_queue.all_requests)
                self._interval_requests.extend(e.waiting_queue)

            q_len = sum(len(e.waiting_queue) for e in agent.engines)
            run_len = sum(len(e.running_queue) for e in agent.engines)

            stats = self._agent_stats[agent_id]
            stats.interval_start_queue_length = q_len
            stats.last_queue_length = q_len
            stats.last_running_requests = run_len

        self._reconfig_in_interval = False
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
        self._interval_requests.append(request)

    def register_reconfig(self):
        self._reconfig_in_interval = True

    def get_state(
        self,
        current_time: float,
        agents: Dict[m.AgentId, m.Agent],
        engines: Dict[str, m.LLMEngine],
    ) -> m.EnvironmentStateData:
        return {
            "arrival_rate": self._get_arrival_rate(agents, current_time),
            "arrival_trend": self._get_arrival_trend(agents, current_time),
            "avg_queue_length": self._get_avg_queue_length(agents),
            "avg_running_requests": self._get_avg_running_requests(agents),
            "queue_delta": self._get_queue_delta(agents),
            "p99_ttft": self._get_p99_ttft(agents, current_time),
            "avg_tpot": self._get_avg_tpot(agents, current_time),
            "kv_cache_utilization": self._get_kv_cache_utilization(engines),
            "mig_config_encoding": self._get_mig_config_encoding(engines),
            "current_budget": self._current_budget,
            "recovery_flag": self._reconfig_in_interval,
        }

    def _get_arrival_rate(
        self, agents: Dict[m.AgentId, m.Agent], current_time: float
    ) -> Dict[m.AgentId, float]:
        rates: Dict[m.AgentId, float] = {}
        start_time = current_time - TRAINING_CONFIG.action_interval
        for agent_id in agents.keys():
            arr = [
                r.arrival_time
                for r in self._interval_requests
                if r.agent_id == agent_id and r.arrival_time >= start_time
            ]
            rates[agent_id] = (
                len(arr) / TRAINING_CONFIG.action_interval
                if TRAINING_CONFIG.action_interval > 0
                else 0.0
            )
        return rates

    def _get_arrival_trend(
        self, agents: Dict[m.AgentId, m.Agent], current_time: float
    ) -> Dict[m.AgentId, float]:
        trends: Dict[m.AgentId, float] = {}
        sub_wdw = TRAINING_CONFIG.action_interval / 3.0
        start_time = current_time - TRAINING_CONFIG.action_interval
        for agent_id in agents.keys():
            arrivals = [
                r.arrival_time
                for r in self._interval_requests
                if r.agent_id == agent_id and r.arrival_time >= start_time
            ]
            counts = [0, 0, 0]
            for t in arrivals:
                idx = int((t - start_time) / sub_wdw) if sub_wdw > 0 else 2
                idx = max(0, min(idx, 2))
                counts[idx] += 1
            trends[agent_id] = (counts[2] - counts[0]) / 2.0
        return trends

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
            for r in self._interval_requests:
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
            for r in self._interval_requests:
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
    ) -> Dict[int, List[int]]:
        num_profiles = len(m.MIGProfile)
        encoding: Dict[int, List[int]] = defaultdict(lambda: [0] * num_profiles)
        for engine in engines.values():
            encoding[engine.gpu][engine.mig_profile.idx] += 1
        return dict(encoding)
