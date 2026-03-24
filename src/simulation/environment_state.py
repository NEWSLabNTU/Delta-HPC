from collections import defaultdict
from typing import Dict, List

from src.simulation.models import *


class EnvironmentStateImpl(EnvironmentState):

    def __init__(self, action_interval: float):
        self._action_interval = action_interval

        self._interval_arrivals: Dict[AgentId, List[float]] = defaultdict(list)
        self._queue_length_integral: Dict[AgentId, float] = defaultdict(float)
        self._last_queue_length: Dict[AgentId, int] = defaultdict(int)
        self._last_queue_update_time: float = 0.0
        self._interval_start_queue_length: Dict[AgentId, int] = defaultdict(int)
        self._reconfig_in_interval: bool = False

    @property
    def action_interval(self) -> float:
        return self._action_interval

    def reset_for_next_interval(
        self,
        current_time: float,
        agents: Dict[AgentId, Agent],
    ):
        self._interval_arrivals.clear()
        self._queue_length_integral.clear()

        for agent_id, agent in agents.items():
            q_len = len(agent.dispatch_queue) + sum(
                len(e.waiting_queue) for e in agent.engines
            )
            self._interval_start_queue_length[agent_id] = q_len
            self._last_queue_length[agent_id] = q_len

        self._reconfig_in_interval = False
        self._last_queue_update_time = current_time

    def record_queue_length_advance(
        self, current_time: float, agents: Dict[AgentId, Agent]
    ):
        dt = current_time - self._last_queue_update_time
        if dt > 0:
            for agent_id in agents.keys():
                self._queue_length_integral[agent_id] += (
                    self._last_queue_length[agent_id] * dt
                )

        self._last_queue_update_time = current_time
        for agent_id, agent in agents.items():
            q_len = len(agent.dispatch_queue) + sum(
                len(e.waiting_queue) for e in agent.engines
            )
            self._last_queue_length[agent_id] = q_len

    def register_arrival(self, agent_id: AgentId, time: float):
        self._interval_arrivals[agent_id].append(time)

    def register_reconfig(self):
        self._reconfig_in_interval = True

    def get_state(self, simulator: Simulator) -> EnvironmentStateData:
        return {
            "arrival_rate": self._get_arrival_rate(simulator),
            "arrival_trend": self._get_arrival_trend(simulator),
            "avg_queue_length": self._get_avg_queue_length(simulator),
            "queue_delta": self._get_queue_delta(simulator),
            "p99_ttft": self._get_p99_ttft(simulator),
            "avg_tpot": self._get_avg_tpot(simulator),
            "kv_cache_utilization": self._get_kv_cache_utilization(simulator),
            "mig_config_encoding": self._get_mig_config_encoding(simulator),
            "recovery_flag": self._reconfig_in_interval,
        }

    def _get_arrival_rate(self, simulator: Simulator) -> Dict[AgentId, float]:
        rates: Dict[AgentId, float] = {}
        for agent_id in simulator.agents.keys():
            arr = self._interval_arrivals.get(agent_id, [])
            rates[agent_id] = (
                len(arr) / self.action_interval if self.action_interval > 0 else 0.0
            )
        return rates

    def _get_arrival_trend(self, simulator: Simulator) -> Dict[AgentId, float]:
        trends: Dict[AgentId, float] = {}
        sub_wdw = self.action_interval / 3.0
        for agent_id in simulator.agents.keys():
            arrivals = self._interval_arrivals.get(agent_id, [])
            counts = [0, 0, 0]
            start_time = simulator.current_time - self.action_interval
            for t in arrivals:
                idx = int((t - start_time) / sub_wdw) if sub_wdw > 0 else 2
                idx = max(0, min(idx, 2))
                counts[idx] += 1
            trends[agent_id] = (counts[2] - counts[0]) / 2.0
        return trends

    def _get_avg_queue_length(self, simulator: Simulator) -> Dict[AgentId, float]:
        avg_q: Dict[AgentId, float] = {}
        for agent_id in simulator.agents.keys():
            integral = self._queue_length_integral.get(agent_id, 0.0)
            avg_q[agent_id] = (
                integral / self.action_interval if self.action_interval > 0 else 0.0
            )
        return avg_q

    def _get_queue_delta(self, simulator: Simulator) -> Dict[AgentId, int]:
        delta: Dict[AgentId, int] = {}
        for agent_id in simulator.agents.keys():
            start_q = self._interval_start_queue_length.get(agent_id, 0)
            end_q = self._last_queue_length.get(agent_id, 0)
            delta[agent_id] = end_q - start_q
        return delta

    def _get_p99_ttft(self, simulator: Simulator) -> Dict[AgentId, float]:
        p99: Dict[AgentId, float] = {}
        start_time = simulator.current_time - self.action_interval
        for agent_id, agent in simulator.agents.items():
            ttfts: List[float] = []

            sorted_done_reqs: List[Request] = sorted(
                agent.completed_requests,
                key=lambda r: (
                    r.finish_time if r.finish_time is not None else -float("inf")
                ),
                reverse=True,
            )
            for r in sorted_done_reqs:
                if r.finish_time is not None and r.finish_time < start_time:
                    break
                if r.first_token_time is not None and r.first_token_time > start_time:
                    ttfts.append(r.first_token_time - r.arrival_time)

            for e in agent.engines:
                for r in e.running_queue.all_requests:
                    if (
                        r.first_token_time is not None
                        and r.first_token_time > start_time
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

    def _get_avg_tpot(self, simulator: Simulator) -> Dict[AgentId, float]:
        """Average time-per-output-token (s/token) for requests that both
        started and finished within the current action interval."""
        avg_tpot: Dict[AgentId, float] = {}
        start_time = simulator.current_time - self.action_interval
        for agent_id, agent in simulator.agents.items():
            tpots: List[float] = []

            # Search completed_requests in reverse-finish-time order;
            # break early once finish_time predates the interval.
            sorted_done_reqs: List[Request] = sorted(
                agent.completed_requests,
                key=lambda r: (
                    r.finish_time if r.finish_time is not None else -float("inf")
                ),
                reverse=True,
            )
            for r in sorted_done_reqs:
                if r.finish_time is None or r.finish_time < start_time:
                    break
                if (
                    r.start_time is not None
                    and r.start_time >= start_time
                    and r.finish_time <= simulator.current_time
                ):
                    duration = r.finish_time - r.start_time
                    if duration > 0 and r.completion_tokens > 0:
                        tpots.append(duration / r.completion_tokens)

            avg_tpot[agent_id] = sum(tpots) / len(tpots) if tpots else 0.0
        return avg_tpot

    def _get_kv_cache_utilization(self, simulator: Simulator) -> Dict[int, List[float]]:
        util: Dict[int, List[float]] = {0: [0.0] * 5, 1: [0.0] * 5}
        for engine in simulator.engines.values():
            if engine.status == EngineStatus.BOOTING:
                continue
            util[engine.gpu][engine.mig_profile.idx] = engine.current_kv_utilization
        return util

    def _get_mig_config_encoding(self, simulator: Simulator) -> Dict[int, List[int]]:
        encoding: Dict[int, List[int]] = {0: [0] * 5, 1: [0] * 5}
        for engine in simulator.engines.values():
            encoding[engine.gpu][engine.mig_profile.idx] += 1
        return dict(encoding)
