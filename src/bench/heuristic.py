import math
import src.simulation.models as m
import src.simulation.utils as utils
from src.bench.config import BENCH_CONFIG
from src.bench.models import Workload
from src.training.config import TRAINING_CONFIG


class RuleBasedHeuristic:
    def __init__(self):
        self.q_threshold_high = BENCH_CONFIG.q_threshold_high
        self.q_threshold_low = BENCH_CONFIG.q_threshold_low
        self.busy_threshold = BENCH_CONFIG.busy_threshold
        self.idle_threshold = BENCH_CONFIG.idle_threshold

        # Precompute normalization denominator for log-scaled queues
        self.q_log_denom = math.log10(1 + TRAINING_CONFIG.norm_avg_queue_length)

    def _denormalize_arrival_rate(self, val: float) -> float:
        return val * TRAINING_CONFIG.norm_arrival_rate

    def _denormalize_waiting_queue(self, val: float) -> float:
        return 10 ** (val * self.q_log_denom) - 1

    def _denormalize_running_queue(self, val: float) -> float:
        return val * TRAINING_CONFIG.norm_avg_running_requests

    def _get_agent_avg_waiting(
        self, state: m.EnvironmentStateData, aid: m.AgentId, agent: m.Agent
    ) -> float:
        qs = state["avg_queue_length"][aid]
        active_indices = {
            e.mig_profile.idx
            for e in agent.engines
            if e.status != m.EngineStatus.BOOTING and not e.is_permanent
        }
        active_indices.update(
            {
                5
                for e in agent.engines
                if e.status != m.EngineStatus.BOOTING and e.is_permanent
            }
        )

        if not active_indices:
            return 0.0

        raw_qs = [self._denormalize_waiting_queue(qs[i]) for i in active_indices]
        return sum(raw_qs) / len(raw_qs)

    def _get_agent_avg_running(
        self, state: m.EnvironmentStateData, aid: m.AgentId, agent: m.Agent
    ) -> float:
        rs = state["avg_running_requests"][aid]
        active_indices = {
            e.mig_profile.idx
            for e in agent.engines
            if e.status != m.EngineStatus.BOOTING and not e.is_permanent
        }
        active_indices.update(
            {
                5
                for e in agent.engines
                if e.status != m.EngineStatus.BOOTING and e.is_permanent
            }
        )

        if not active_indices:
            return 0.0

        raw_rs = [self._denormalize_running_queue(rs[i]) for i in active_indices]
        return sum(raw_rs) / len(raw_rs)

    def decide_action(self, sim: m.Simulator) -> m.ResourceManagerAction:
        state = sim.get_state(0)
        arrival_rates = {
            aid: self._denormalize_arrival_rate(r)
            for aid, r in state["arrival_rate"].items()
        }

        # 1. Categorize workloads
        workloads = {}
        for aid in m.AgentId:
            rate = arrival_rates[aid]
            if rate >= self.busy_threshold:
                workloads[aid] = Workload.BUSY
            elif rate <= self.idle_threshold:
                workloads[aid] = Workload.IDLE
            else:
                workloads[aid] = Workload.EVEN

        mask = sim.get_action_mask()
        all_actions = list(m.ResourceManagerAction)
        all_engines = [e for a in sim.agents.values() for e in a.engines]

        # 2. Scale-Up Logic (Busy Agents or Agents with High Queue)
        # We act on agents that are either categorized as BUSY or already experiencing high queue pressure.
        needs_scale_up = []
        for aid in m.AgentId:
            agent = sim.agents[aid]
            avg_waiting = self._get_agent_avg_waiting(state, aid, agent)
            if workloads[aid] == Workload.BUSY or avg_waiting >= self.q_threshold_high:
                needs_scale_up.append((aid, avg_waiting))

        if needs_scale_up:
            # Sort by waiting queue length (non-increasing)
            needs_scale_up.sort(key=lambda x: x[1], reverse=True)

            for aid, avg_waiting in needs_scale_up:
                agent = sim.agents[aid]
                # Attempt Split
                split_action = utils.MIG_RULES.select_best_split_action(
                    agent, mask, all_actions
                )
                if split_action is not None:
                    return split_action

                # Attempt VRAM Transfer (Iterative Search)
                potential_givers = sorted(
                    [oaid for oaid in m.AgentId if oaid != aid],
                    key=lambda k: arrival_rates[k],
                )
                for giver_aid in potential_givers:
                    transfer_action = utils.MIG_RULES.select_best_transfer_action(
                        sim.agents[giver_aid], aid, mask, all_actions, all_engines
                    )
                    if transfer_action is not None:
                        return transfer_action

        # 3. Scale-Down Logic (Idle Agents or Agents with Low Queue)
        # We act on agents that are either categorized as IDLE or have consistently low running queues.
        needs_scale_down = []
        for aid in m.AgentId:
            agent = sim.agents[aid]
            avg_running = self._get_agent_avg_running(state, aid, agent)
            if workloads[aid] == Workload.IDLE or avg_running <= self.q_threshold_low:
                needs_scale_down.append((aid, avg_running))

        if needs_scale_down:
            # Sort by arrival rates (non-decreasing) to prioritize idling resource release
            needs_scale_down.sort(key=lambda x: arrival_rates[x[0]])

            for aid, avg_running in needs_scale_down:
                agent = sim.agents[aid]
                # Attempt Merge
                merge_action = utils.MIG_RULES.select_best_merge_action(
                    agent, mask, all_actions
                )
                if merge_action is not None:
                    return merge_action

        return m.ResourceManagerAction.NO_ACTION
