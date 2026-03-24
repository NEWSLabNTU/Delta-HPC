from __future__ import annotations

import random
from tqdm import tqdm
from sortedcontainers import SortedList
from typing import Any, List, Dict, Tuple, cast

from src.simulation.models import *
import src.simulation.global_vars as g
from src.simulation.engine import LLMEngineImpl
from src.simulation.logger import SimulationLoggerImpl
from src.simulation.worker import WorkerImpl
from src.simulation.environment_state import EnvironmentStateImpl


class ResourceManager:
    def __init__(self, simulator: Simulator):
        self.simulator = simulator
        self.trigger_interval = 3600.0
        self.next_vram_transfer_time = 1800.0
        self.next_mig_trigger_time = 3600.0
        self.worker = WorkerImpl(simulator)

    def act(self, current_time: float, state: EnvironmentStateData):
        """
        Periodically checks if it is time to trigger VRAM transfer or MIG split/merge.
        """
        if current_time >= self.next_mig_trigger_time:
            self.trigger_mig(current_time)
            self.next_mig_trigger_time += self.trigger_interval

        if current_time >= self.next_vram_transfer_time:
            self.trigger_vram_transfer(current_time)
            self.next_vram_transfer_time += self.trigger_interval

    def trigger_vram_transfer(self, current_time: float):
        """
        Periodically transfers VRAM between agents.
        Randomly selects an agent. If it has >20GB active VRAM, transfers 10 or 20GB.
        If it has exactly 20GB, transfers 10GB.
        Otherwise, it attempts the same logic on the other agent.
        """
        agents = list(self.simulator.agents.values())
        if len(agents) < 2:
            return

        random.shuffle(agents)

        for i in range(2):
            giver = agents[i]
            receiver = agents[1 - i]

            active_vram = sum(
                e.mig_profile.vram
                for e in giver.engines
                if e.status == EngineStatus.ACTIVE
            )

            amount = 0
            if active_vram > 20:
                amount = random.choice([10, 20])
            elif active_vram == 20:
                amount = 10

            if amount > 0:
                self.worker.start_transfer(
                    current_time,
                    TransferDetails(
                        amount, giver_id=giver.agent_id, receiver_id=receiver.agent_id
                    ),
                )
                return

    def trigger_mig(self, current_time: float):
        sim = self.simulator

        candidates: List[Tuple[str, Any]] = []  # List of (action type, data)
        for agent in sim.agents.values():
            # 1. Look for Merges
            possible_merges = g.MIG_RULES.get_possible_merges(agent)
            for engs, new_profile in possible_merges:
                candidates.append(
                    (
                        "merge",
                        {
                            "engines": engs,
                            "new_profile": new_profile,
                            "agent_id": agent.agent_id,
                            "gpu": engs[0].gpu,
                        },
                    )
                )

            # 2. Look for Splits
            possible_splits = g.MIG_RULES.get_possible_splits(agent)
            for eng, new_profiles in possible_splits:
                candidates.append(
                    (
                        "split",
                        {
                            "engine": eng,
                            "new_profiles": new_profiles,
                            "agent_id": agent.agent_id,
                            "gpu": eng.gpu,
                        },
                    )
                )

        if not candidates:
            return

        action_type, data = random.choice(candidates)

        if action_type == "merge":
            engs: List[LLMEngine] = data["engines"]
            eids = [e.engine_id for e in engs]
            # Both engines carry the full merge payload; drained_ids starts empty.
            merge_payload: ShutdownMergePayload = {
                "engine_id": engs[0].engine_id,  # will be overwritten per engine below
                "purpose": OperationPurpose.MERGE,
                "merge_engine_ids": tuple(eids),
                "drained_ids": [],  # This list is shared between 2 engines
                "new_profile": data["new_profile"],
                "agent_id": data["agent_id"],
                "gpu": data["gpu"],
                "receiver_id": None,
            }
            for e in engs:
                per_engine_payload: ShutdownMergePayload = {
                    **merge_payload,
                    "engine_id": e.engine_id,
                }
                evt = e.trigger_shutdown(per_engine_payload, current_time)
                if evt:
                    sim.events.add(evt)
            sim.logger.log_mig_merge_trigger(current_time, eids, data["gpu"])

        elif action_type == "split":
            e: LLMEngine = data["engine"]
            split_payload: ShutdownSplitPayload = {
                "engine_id": e.engine_id,
                "purpose": OperationPurpose.SPLIT,
                "new_profiles": data["new_profiles"],
                "agent_id": data["agent_id"],
                "gpu": data["gpu"],
                "receiver_id": None,
                "received_profile": None,
            }
            evt = e.trigger_shutdown(split_payload, current_time)
            if evt:
                sim.events.add(evt)
            sim.logger.log_mig_split_trigger(current_time, e.engine_id, data["gpu"])


class SimulatorImpl(Simulator):
    def __init__(
        self,
        agents: Dict[AgentId, Agent],
        engines: Dict[str, LLMEngine],
        no_log: bool = False,
    ):
        self._agents = agents
        self._engines = engines
        self._events: SortedList[SimulationEvent] = SortedList()
        self._current_time: float = 0.0
        self.resource_manager = ResourceManager(self)
        self._logger = SimulationLoggerImpl(enabled=not no_log)
        self.total_requests = 0

        self.action_interval = g.SIM_CONFIG.get_rl_action_interval()
        self.environment_state = EnvironmentStateImpl(self.action_interval)
        self.environment_state.reset_for_next_interval(0.0, self._agents, self._engines)

        # Schedule Resource Manager Action at action_interval
        self._events.add(
            SimulationEvent(
                time=self.action_interval,
                event_type=EventType.RESOURCE_MANAGER_TRIGGER,
                payload={},
            ),
        )

    @property
    def agents(self) -> Dict[AgentId, Agent]:
        return self._agents

    @property
    def engines(self) -> Dict[str, LLMEngine]:
        return self._engines

    @property
    def events(self) -> SortedList[SimulationEvent]:
        return self._events

    @property
    def current_time(self) -> float:
        return self._current_time

    @property
    def logger(self) -> SimulationLogger:
        return self._logger

    def add_arrival_events(self, requests: List[Request]):
        self.total_requests = len(requests)

        for req in requests:
            self._events.add(
                SimulationEvent(
                    time=req.arrival_time,
                    event_type=EventType.REQUEST_ARRIVAL,
                    payload={"request": req},
                )
            )

        self._sample_rag_searches()

    def _sample_rag_searches(self) -> None:
        search_evts: List[SimulationEvent] = []
        for evt in self._events:
            if "request" in evt.payload:
                if evt.payload["request"].agent_id == AgentId.RAG:
                    search_evts.append(
                        SimulationEvent(
                            time=evt.time + g.SIM_CONFIG.get_rag_overhead(),
                            event_type=EventType.RAG_SEARCH_COMPLETE,
                            payload=evt.payload,
                        )
                    )

        self._events.update(search_evts)

    def has_active_work(self) -> bool:
        if any(
            e.event_type != EventType.RESOURCE_MANAGER_TRIGGER for e in self._events
        ):
            return True
        if any(
            len(e.waiting_queue) > 0 or len(e.running_queue) > 0
            for e in self._engines.values()
        ):
            return True
        if any(len(a.dispatch_queue) > 0 for a in self._agents.values()):
            return True
        return False

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _step_idle_engines(self):
        """Kick engines that have queued work but no scheduled step event."""
        for engine in self._engines.values():
            if engine.waiting_queue or engine.running_queue:
                evt = engine.step(self._current_time, next_arrival_time=None)
                if evt:
                    self._events.add(evt)

    def _step_draining_or_active_engines(self):
        """After a trigger event, nudge engines that have waiting work."""
        for eg in self._engines.values():
            if (
                eg.status in [EngineStatus.ACTIVE, EngineStatus.DRAINING]
                and len(eg.running_queue) == 0
                and len(eg.waiting_queue) > 0
            ):
                evt = eg.step(
                    self._current_time,
                    next_arrival_time=(
                        self._peak_next_stopping_evt(eg.owner.agent_id)
                        if eg.status == EngineStatus.ACTIVE
                        else None
                    ),
                )
                if evt:
                    self._events.add(evt)

    def _handle_resource_manager_trigger(self):
        self.environment_state.record_queue_length_advance(
            self._current_time, self._agents
        )
        state_data = self.environment_state.get_state(self)
        self._logger.log_environment_state(self._current_time, state_data)
        self.resource_manager.act(self._current_time, state_data)
        self.environment_state.reset_for_next_interval(
            self._current_time, self._agents, self._engines
        )

        self._events.add(
            SimulationEvent(
                time=self._current_time + self.action_interval,
                event_type=EventType.RESOURCE_MANAGER_TRIGGER,
                payload={},
            )
        )
        self._step_draining_or_active_engines()

    def _handle_shutdown_complete_reallocate(self, payload: ShutdownReallocatePayload):
        engine_id = payload["engine_id"]
        receiver_id = payload["receiver_id"]
        engine = self._engines[engine_id]

        new_model = g.SIM_CONFIG.get_model(receiver_id, engine.mig_profile)

        receiver = self._agents[receiver_id]

        engine.update_model(
            new_owner=receiver,
            model_name=new_model,
            max_batched_tokens=g.SIM_CONFIG.max_batched_tokens[new_model],
            prefill_params=g.SIM_CONFIG.get_prefill_params(
                receiver_id, engine.mig_profile
            ),
            tpot_params=g.SIM_CONFIG.get_tpot_params(receiver_id, engine.mig_profile),
            restart_time=g.SIM_CONFIG.get_restart_time(receiver_id, engine.mig_profile),
        )

        boot_payload: BootPayload = {
            "engine_id": engine_id,
            "purpose": OperationPurpose.REALLOCATE,
            "sibling_engine_ids": None,
        }
        self._events.add(engine.trigger_boot(boot_payload))

    def _handle_shutdown_complete_merge(self, payload: ShutdownMergePayload):
        engine_id = payload["engine_id"]

        # Mark this engine as drained; both sibling events share the same
        # drained_ids list object created in trigger_mig.
        payload["drained_ids"].append(engine_id)

        if len(payload["drained_ids"]) == len(payload["merge_engine_ids"]):
            giver_agent = self._agents[payload["agent_id"]]
            for eid in payload["merge_engine_ids"]:
                e = self._engines.pop(eid, None)
                if e is not None and e in giver_agent.engines:
                    giver_agent.engines.remove(e)

            new_profile = payload["new_profile"]
            new_eid = (
                f"GPU_{payload['gpu']}_{new_profile.string}_{int(self._current_time)}"
            )
            # If a receiver was specified (merge-for-transfer), boot directly on receiver
            receiver_id = payload.get("receiver_id")
            target_agent = self._agents[receiver_id] if receiver_id else giver_agent
            new_eng = LLMEngineImpl.create(
                payload["gpu"], new_eid, target_agent, new_profile, self._current_time
            )

            boot_plain: BootPayload = {
                "engine_id": new_eid,
                "purpose": OperationPurpose.PLAIN,
                "sibling_engine_ids": None,
            }
            self._events.add(new_eng.trigger_boot(boot_plain))
            self._engines[new_eid] = new_eng
            target_agent.add_engine(new_eng)
            self._logger.log_mig_merge_complete(self._current_time, new_eid)

    def _handle_shutdown_complete_split(self, payload: ShutdownSplitPayload):
        engine_id = payload["engine_id"]
        agent = self._agents[payload["agent_id"]]
        e = self._engines.pop(engine_id, None)
        if e is not None and e in agent.engines:
            agent.engines.remove(e)

        receiver_id = payload["receiver_id"]
        received_profile = payload["received_profile"]
        new_profiles = payload["new_profiles"]
        gpu = payload["gpu"]

        is_vram_transfer = receiver_id is not None
        if is_vram_transfer:
            assert received_profile is not None
        has_received = False

        new_eids = [
            f"GPU_{gpu}_{p.string}_{int(self._current_time)}_{i}"
            for i, p in enumerate(new_profiles)
        ]
        for new_eid, p in zip(new_eids, new_profiles):
            new_owner = agent
            if is_vram_transfer and p == received_profile and not has_received:
                new_owner = self._agents[receiver_id]
                has_received = True

            new_eng = LLMEngineImpl.create(
                payload["gpu"], new_eid, new_owner, p, self._current_time
            )
            new_owner.add_engine(new_eng)

            boot_split: BootPayload = {
                "engine_id": new_eid,
                "purpose": OperationPurpose.SPLIT,
                "sibling_engine_ids": new_eids,
            }

            self._events.add(new_eng.trigger_boot(boot_split))
            self._engines[new_eid] = new_eng
        self._logger.log_mig_split_complete(self._current_time, engine_id)

    def _handle_shutdown_complete(
        self,
        payload: ShutdownPayload,
    ):
        self.environment_state.register_reconfig()
        purpose = payload["purpose"]
        match purpose:
            case OperationPurpose.REALLOCATE:
                self._handle_shutdown_complete_reallocate(
                    cast(ShutdownReallocatePayload, payload)
                )
            case OperationPurpose.MERGE:
                self._handle_shutdown_complete_merge(
                    cast(ShutdownMergePayload, payload)
                )
            case OperationPurpose.SPLIT:
                self._handle_shutdown_complete_split(
                    cast(ShutdownSplitPayload, payload)
                )
            case _:
                raise ValueError(f"Unexpected shutdown purpose {purpose}")

    def _handle_boot_complete(self, payload: BootPayload) -> None:
        """Handle ENGINE_BOOT_COMPLETE.

        Activates the engine and processes the agent's waiting queue.
        Returns early if sibling engines from a split are still booting.
        """
        self.environment_state.register_reconfig()
        engine_id = payload["engine_id"]

        engine = self._engines[engine_id]
        engine.activate(self._current_time)
        agent = engine.owner

        self._logger.log_engine_boot_complete(self._current_time, engine_id)

        # Defer waiting-queue processing if split siblings are still booting
        if payload["purpose"] == OperationPurpose.SPLIT:
            assert payload["sibling_engine_ids"] is not None
            still_booting = any(
                self._engines[sid].status == EngineStatus.BOOTING
                for sid in payload["sibling_engine_ids"]
                if sid != engine_id
                and self._engines[sid].owner.agent_id == agent.agent_id
            )
            if still_booting:
                return  # Wait for remaining sibling engines

        agent.process_waiting_queue(self._current_time)
        for e in agent.engines:
            if (
                e.status == EngineStatus.ACTIVE
                and len(e.running_queue) == 0
                and len(e.waiting_queue) > 0
            ):
                evt = e.step(
                    self._current_time,
                    next_arrival_time=self._peak_next_stopping_evt(agent.agent_id),
                )
                if evt:
                    self._events.add(evt)

    def _handle_request_arrival(self, payload: RequestArrivalPayload):
        req = payload["request"]
        agent_id = req.agent_id

        self.environment_state.register_arrival(agent_id, self._current_time)

        # If it's a RAG request, wiat til RAG_SEARCH_COMPLETE
        if agent_id == AgentId.RAG:
            self._logger.log_request_arrival(self._current_time, req, None)
            return

        agent = self._agents[agent_id]
        engine = agent.dispatch(req, self._current_time)
        self._logger.log_request_arrival(self._current_time, req, engine)

        if engine and len(engine.running_queue) == 0 and len(engine.waiting_queue) > 0:
            evt = engine.step(
                self._current_time,
                next_arrival_time=self._peak_next_stopping_evt(agent_id),
            )
            if evt:
                self._events.add(evt)

    def _handle_rag_search_complete(self, payload: RequestArrivalPayload):
        req = payload["request"]
        agent_id = req.agent_id
        assert agent_id == AgentId.RAG

        agent = self._agents[agent_id]
        engine = agent.dispatch(req, self._current_time)
        self._logger.log_rag_search_complete(self._current_time, req, engine)
        if engine and len(engine.running_queue) == 0 and len(engine.waiting_queue) > 0:
            evt = engine.step(
                self._current_time,
                next_arrival_time=self._peak_next_stopping_evt(agent_id),
            )
            if evt:
                self._events.add(evt)

    def _handle_engine_step_complete(self, payload: EngineStepPayload):
        engine_id = payload["engine_id"]
        engine = self._engines[engine_id]

        next_arrival_time = self._peak_next_stopping_evt(engine.owner.agent_id)
        self._logger.log_engine_step(
            self._current_time, self._agents, engine, next_arrival_time
        )

        evt = engine.step(self._current_time, next_arrival_time=next_arrival_time)
        if evt:
            self._events.add(evt)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self):
        """Main event loop."""
        pbar = tqdm(total=self.total_requests, desc="Simulation Progress")
        last_completed = 0

        while self.has_active_work():
            if not self._events:
                self._step_idle_engines()
                if not self.has_active_work():
                    break

            current_event = self._events.pop(0)
            self.environment_state.record_queue_length_advance(
                current_event.time, self._agents
            )
            self._current_time = current_event.time

            match current_event.event_type:
                case EventType.RESOURCE_MANAGER_TRIGGER:
                    self._handle_resource_manager_trigger()
                case EventType.ENGINE_SHUTDOWN_COMPLETE:
                    self._handle_shutdown_complete(
                        cast(ShutdownPayload, current_event.payload)
                    )
                case EventType.ENGINE_BOOT_COMPLETE:
                    self._handle_boot_complete(cast(BootPayload, current_event.payload))
                case EventType.REQUEST_ARRIVAL:
                    self._handle_request_arrival(
                        cast(RequestArrivalPayload, current_event.payload)
                    )
                case EventType.RAG_SEARCH_COMPLETE:
                    self._handle_rag_search_complete(
                        cast(RequestArrivalPayload, current_event.payload)
                    )
                case EventType.ENGINE_STEP_COMPLETE:
                    self._handle_engine_step_complete(
                        cast(EngineStepPayload, current_event.payload)
                    )

            self.environment_state.record_queue_length_advance(
                self._current_time, self._agents
            )

            completed_now = sum(
                len(ag.completed_requests) for ag in self._agents.values()
            )
            if completed_now > last_completed:
                pbar.update(completed_now - last_completed)
                last_completed = completed_now

        pbar.close()
        self._logger.flush()

    def _peak_next_stopping_evt(self, agent_id: AgentId) -> float | None:
        for evt in self._events:
            if evt.event_type == EventType.RESOURCE_MANAGER_TRIGGER:
                return evt.time

            if evt.event_type == EventType.REQUEST_ARRIVAL:
                payload = cast(RequestArrivalPayload, evt.payload)
                if payload["request"].agent_id == agent_id and agent_id != AgentId.RAG:
                    return evt.time

            if evt.event_type == EventType.RAG_SEARCH_COMPLETE:
                payload = cast(RequestArrivalPayload, evt.payload)
                if payload["request"].agent_id == agent_id and agent_id == AgentId.RAG:
                    return evt.time
        return None
