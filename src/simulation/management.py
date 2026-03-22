from __future__ import annotations
from collections import defaultdict
from typing import Dict, List, Any

from models import *
import global_vars as g

# from simulator import ResourceManager


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
        engines: Dict[str, LLMEngine],
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

            for r in reversed(agent.completed_requests):
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


class WorkerImpl(Worker):
    def __init__(self, simulator: Simulator, resource_manager: Any):
        self.simulator = simulator
        self.resource_manager = resource_manager

    def queue_transfer(
        self, amount: int, giver_id: AgentId, receiver_id: AgentId, current_time: float
    ):
        """
        Adds a new transfer request to the queue and immediately attempts to process it.
        """
        self.resource_manager.pending_transfers.append(
            PendingTransfer(amount, giver_id, receiver_id)
        )
        self.step(current_time)

    def step(self, current_time: float):
        """
        Processes the next pending VRAM transfer in the queue.
        Resolution order (after exact-match):
          1. Merge two smaller engines to reach or exceed the target VRAM.
          2. Split a larger engine that yields an exact-VRAM child.
          3. Generic split of the smallest-sufficient larger engine.
          4. Drop the transfer if no engine configuration can satisfy it.
        """
        if not self.resource_manager.pending_transfers:
            return
        # Process only one transfer at a time; if a prior one is mid-flight, wait.
        if len(self.resource_manager.pending_transfers) > 1:
            return

        t = self.resource_manager.pending_transfers[0]
        giver = self.simulator.agents[t.giver_id]
        receiver = self.simulator.agents[t.receiver_id]

        # 0. Wait until every giver engine has settled (no draining / booting)
        if any(e.status != EngineStatus.ACTIVE for e in giver.engines):
            return

        # 1. Exact-match: hand off an engine whose VRAM == requested amount
        exact_matches = [
            e
            for e in giver.engines
            if e.status == EngineStatus.ACTIVE and e.mig_profile.vram == t.amount
        ]
        if exact_matches:
            engine_to_shift = min(
                exact_matches,
                key=lambda e: len(e.running_queue) + len(e.waiting_queue),
            )
            shutdown_payload: ShutdownReallocatePayload = {
                "engine_id": engine_to_shift.engine_id,
                "purpose": OperationPurpose.REALLOCATE,
                "receiver_id": receiver.agent_id,
            }
            self.simulator.logger.log_vram_transfer(
                current_time,
                giver.agent_id,
                receiver.agent_id,
                t.amount,
                engine_to_shift.engine_id,
            )
            evt = engine_to_shift.trigger_shutdown(shutdown_payload, current_time)
            if evt:
                self.simulator.events.add(evt)
            self.resource_manager.pending_transfers.pop(0)
            return

        # 2. Merge: combine two smaller engines if the result exactly matches the target VRAM.
        # (Merging into a larger-than-needed engine would require a follow-up split and is
        # handled by steps 3a/3b once the merged engine exists.)
        merge_candidates = [
            (e1, e2, p)
            for e1, e2, p in g.find_merge_candidates(giver.engines)
            if p.vram == t.amount
        ]
        if merge_candidates:
            # Among qualifying pairs, prefer lowest combined queue load
            e1, e2, new_profile = min(
                merge_candidates,
                key=lambda c: len(c[0].running_queue) + len(c[1].running_queue),
            )
            merge_payload: ShutdownMergePayload = {
                "engine_id": e1.engine_id,
                "purpose": OperationPurpose.MERGE,
                "merge_engine_ids": (e1.engine_id, e2.engine_id),
                "drained_ids": [],
                "new_profile": new_profile,
                "agent_id": giver.agent_id,
                "gpu": e1.gpu,
                "receiver_id": receiver.agent_id,
            }
            for e in [e1, e2]:
                per_engine_payload: ShutdownMergePayload = {
                    **merge_payload,
                    "engine_id": e.engine_id,
                }
                evt = e.trigger_shutdown(per_engine_payload, current_time)
                if evt:
                    self.simulator.events.add(evt)
            self.simulator.logger.log_mig_merge_trigger(
                current_time, e1.engine_id, e2.engine_id, e1.gpu
            )
            self.simulator.logger.log_vram_transfer(
                current_time,
                giver.agent_id,
                receiver.agent_id,
                t.amount,
                e1.engine_id + " & " + e2.engine_id,
            )
            self.resource_manager.pending_transfers.pop(0)
            return

        # 3a. Split-to-exact: find a larger engine that can be split into an exact match
        split_candidates = g.find_split_candidates(giver.engines)
        exact_split = [
            (e, idx, children)
            for e, idx, children in split_candidates
            if e.mig_profile.vram > t.amount and children[idx].vram == t.amount
        ]
        if exact_split:
            e, transfer_index, children = min(
                exact_split,
                key=lambda c: len(c[0].running_queue) + len(c[0].waiting_queue),
            )
            split_payload: ShutdownSplitPayload = {
                "engine_id": e.engine_id,
                "purpose": OperationPurpose.SPLIT,
                "new_profiles": children,
                "agent_id": giver.agent_id,
                "gpu": e.gpu,
                "transfer_index": transfer_index,
                "transfer_receiver_id": receiver.agent_id,
            }
            evt = e.trigger_shutdown(split_payload, current_time)
            if evt:
                self.simulator.events.add(evt)
            self.simulator.logger.log_mig_split_trigger(
                current_time, e.engine_id, e.gpu
            )
            self.simulator.logger.log_vram_transfer(
                current_time, giver.agent_id, receiver.agent_id, t.amount, e.engine_id
            )
            self.resource_manager.pending_transfers.pop(0)
            return

        # 3b. Generic split: split the smallest engine that is still larger than target,
        # but only if at least one resulting child has vram >= target (productive progress).
        larger_engines = [
            e
            for e in giver.engines
            if e.status == EngineStatus.ACTIVE and e.mig_profile.vram > t.amount
        ]
        if larger_engines:
            engine_to_split = min(larger_engines, key=lambda e: e.mig_profile.vram)
            if engine_to_split.mig_profile in g.MIG_MERGE_RULES.inverse:
                children = list(g.MIG_MERGE_RULES.inverse[engine_to_split.mig_profile])
                # Only split if this brings us closer — at least one child can reach the target
                if any(child.vram >= t.amount for child in children):
                    split_payload_generic: ShutdownSplitPayload = {
                        "engine_id": engine_to_split.engine_id,
                        "purpose": OperationPurpose.SPLIT,
                        "new_profiles": children,
                        "agent_id": giver.agent_id,
                        "gpu": engine_to_split.gpu,
                        "transfer_index": None,
                        "transfer_receiver_id": None,
                    }
                    evt = engine_to_split.trigger_shutdown(
                        split_payload_generic, current_time
                    )
                    if evt:
                        self.simulator.events.add(evt)
                    self.simulator.logger.log_mig_split_trigger(
                        current_time, engine_to_split.engine_id, engine_to_split.gpu
                    )
                    return

        # 4. Give up — log and discard
        self.simulator.logger.log(
            f"[{current_time:.4f}] TRANSFER DROP | Giver {giver.agent_id.value} "
            f"cannot satisfy {t.amount}GB transfer"
        )
        self.resource_manager.pending_transfers.pop(0)
