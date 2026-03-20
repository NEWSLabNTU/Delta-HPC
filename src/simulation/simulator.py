import random
from typing import List, Dict, Tuple, cast
from tqdm import tqdm
from sortedcontainers import SortedList

import global_vars as g
from logger import SimulationLogger
from models import (
    EngineStatus,
    EventType,
    AgentId,
    SimulationEvent,
    Agent,
    LLMEngine,
    Simulator as SimulatorI,
    Request,
    OperationPurpose,
    RequestArrivalPayload,
    ShutdownReallocatePayload,
    ShutdownMergePayload,
    ShutdownSplitPayload,
    BootPlainPayload,
    BootReallocatePayload,
    BootSplitPayload,
)
from objects import LLMEngine as LLMEngineImpl


class ResourceManager:
    def __init__(self, simulator: "SimulatorI"):
        self.simulator = simulator
        self.trigger_interval = 3600.0

    def trigger(self, current_time: float):
        coding_agent = self.simulator.agents[AgentId.CODING]
        rag_agent = self.simulator.agents[AgentId.RAG]

        candidates_to_give: List[Tuple[Agent, Agent]] = []
        # Support multiple engines
        if len(coding_agent.engines) > 1:
            # We should only pick from ACTIVE engines
            active_engines = [
                e for e in coding_agent.engines if e.status == EngineStatus.ACTIVE
            ]
            if active_engines:
                candidates_to_give.append((coding_agent, rag_agent))

        if len(rag_agent.engines) > 1:
            active_engines = [
                e for e in rag_agent.engines if e.status == EngineStatus.ACTIVE
            ]
            if active_engines:
                candidates_to_give.append((rag_agent, coding_agent))

        if not candidates_to_give:
            return

        giver, receiver = random.choice(candidates_to_give)
        active_engines = [e for e in giver.engines if e.status == EngineStatus.ACTIVE]
        engine_to_shift = random.choice(active_engines)

        self.simulator.logger.log_reallocation(
            current_time, giver.agent_id, receiver.agent_id, engine_to_shift.engine_id
        )

        payload: ShutdownReallocatePayload = {
            "engine_id": engine_to_shift.engine_id,
            "purpose": OperationPurpose.REALLOCATE,
            "receiver_id": receiver.agent_id,
        }
        evt = engine_to_shift.trigger_shutdown(payload, current_time)
        if evt:
            self.simulator.events.add(evt)

    def trigger_mig(self, current_time: float):
        sim = self.simulator
        # Group active engines by (Agent, GPU)
        grouped_engines: Dict[Tuple[AgentId, int], List[LLMEngine]] = {}
        for engine in sim.engines.values():
            if engine.status == EngineStatus.ACTIVE:
                key = (engine.owner.agent_id, engine.gpu)
                if key not in grouped_engines:
                    grouped_engines[key] = []
                grouped_engines[key].append(engine)

        candidates = []  # List of (type, data)

        for (agent_id, gpu), engines in grouped_engines.items():
            # 1. Look for Merges using bidict
            for i in range(len(engines)):
                for j in range(i + 1, len(engines)):
                    e1 = engines[i]
                    e2 = engines[j]
                    canonical_key = (
                        (e1.mig_profile, e2.mig_profile)
                        if e1.mig_profile.size > e2.mig_profile.size
                        else (e2.mig_profile, e1.mig_profile)
                    )
                    new_profile = g.MIG_MERGE_RULES.get(canonical_key)
                    if new_profile:
                        candidates.append(
                            (
                                "merge",
                                {
                                    "engines": [e1, e2],
                                    "new_profile": new_profile,
                                    "agent_id": agent_id,
                                    "gpu": gpu,
                                },
                            )
                        )

            # 2. Look for Splits using bidict.inverse
            for e in engines:
                if e.mig_profile in g.MIG_MERGE_RULES.inverse:
                    new_profiles = g.MIG_MERGE_RULES.inverse[e.mig_profile]
                    candidates.append(
                        (
                            "split",
                            {
                                "engine": e,
                                "new_profiles": list(new_profiles),
                                "agent_id": agent_id,
                                "gpu": gpu,
                            },
                        )
                    )

        if not candidates:
            return

        action_type, data = random.choice(candidates)

        if action_type == "merge":
            e1, e2 = data["engines"]
            # Both engines carry the full merge payload; drained_ids starts empty.
            merge_payload: ShutdownMergePayload = {
                "engine_id": e1.engine_id,  # will be overwritten per engine below
                "purpose": OperationPurpose.MERGE,
                "merge_engine_ids": (e1.engine_id, e2.engine_id),
                "drained_ids": [],  # This list is shared between 2 engines
                "new_profile": data["new_profile"],
                "agent_id": data["agent_id"],
                "gpu": data["gpu"],
            }
            for e in [e1, e2]:
                per_engine_payload: ShutdownMergePayload = {
                    **merge_payload,
                    "engine_id": e.engine_id,
                }
                evt = e.trigger_shutdown(per_engine_payload, current_time)
                if evt:
                    sim.events.add(evt)
            sim.logger.log_mig_merge_trigger(
                current_time, e1.engine_id, e2.engine_id, data["gpu"]
            )

        elif action_type == "split":
            e = data["engine"]
            split_payload: ShutdownSplitPayload = {
                "engine_id": e.engine_id,
                "purpose": OperationPurpose.SPLIT,
                "new_profiles": data["new_profiles"],
                "agent_id": data["agent_id"],
                "gpu": data["gpu"],
            }
            evt = e.trigger_shutdown(split_payload, current_time)
            if evt:
                sim.events.add(evt)
            sim.logger.log_mig_split_trigger(current_time, e.engine_id, data["gpu"])


class Simulator(SimulatorI):
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
        self._logger = SimulationLogger(enabled=not no_log)
        self.total_requests = 0

        # Schedule Reallocation at 3600.0
        self._events.add(
            SimulationEvent(
                time=3600.0,
                event_type=EventType.REALLOCATION_TRIGGER,
                payload={},
            ),
        )
        # Schedule MIG at 1800.0
        self._events.add(
            SimulationEvent(
                time=1800.0,
                event_type=EventType.MIG_TRIGGER,
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
                    payload={
                        "request": req,
                        "target_agent": req.agent_id,
                    },
                )
            )

    def has_active_work(self) -> bool:
        if any(
            e.event_type not in [EventType.REALLOCATION_TRIGGER, EventType.MIG_TRIGGER]
            for e in self._events
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

    def _handle_reallocation_trigger(self):
        self.resource_manager.trigger(self._current_time)
        self._events.add(
            SimulationEvent(
                time=self._current_time + self.resource_manager.trigger_interval,
                event_type=EventType.REALLOCATION_TRIGGER,
                payload={},
            )
        )
        self._step_draining_or_active_engines()

    def _handle_mig_trigger(self):
        self.resource_manager.trigger_mig(self._current_time)
        self._events.add(
            SimulationEvent(
                time=self._current_time + self.resource_manager.trigger_interval,
                event_type=EventType.MIG_TRIGGER,
                payload={},
            )
        )
        self._step_draining_or_active_engines()

    def _handle_shutdown_complete_reallocate(self, payload: ShutdownReallocatePayload):
        engine_id = payload["engine_id"]
        receiver_id = payload["receiver_id"]
        engine = self._engines[engine_id]

        new_model = g.SIM_CONFIG.get_model(receiver_id, engine.mig_profile)
        engine.update_model(
            model_name=new_model,
            max_batched_tokens=g.SIM_CONFIG.max_batched_tokens[new_model],
            prefill_params=g.SIM_CONFIG.get_prefill_params(
                receiver_id, engine.mig_profile
            ),
            tpot_params=g.SIM_CONFIG.get_tpot_params(
                receiver_id, engine.mig_profile
            ),
            restart_time=g.SIM_CONFIG.get_restart_time(
                receiver_id, engine.mig_profile
            ),
        )

        giver = engine.owner
        receiver = self._agents[receiver_id]
        if engine in giver.engines:
            giver.engines.remove(engine)
        receiver.add_engine(engine)
        engine.owner = receiver

        boot_payload: BootReallocatePayload = {
            "engine_id": engine_id,
            "purpose": OperationPurpose.REALLOCATE,
            "giver_id": giver.agent_id,
            "receiver_id": receiver_id,
        }
        self._events.add(engine.trigger_boot(boot_payload))

    def _handle_shutdown_complete_merge(self, payload: ShutdownMergePayload):
        engine_id = payload["engine_id"]

        # Mark this engine as drained; both sibling events share the same
        # drained_ids list object created in trigger_mig.
        payload["drained_ids"].append(engine_id)

        if len(payload["drained_ids"]) == len(payload["merge_engine_ids"]):
            agent = self._agents[payload["agent_id"]]
            for eid in payload["merge_engine_ids"]:
                e = self._engines.pop(eid, None)
                if e is not None and e in agent.engines:
                    agent.engines.remove(e)

            new_profile = payload["new_profile"]
            new_eid = f"GPU_{payload['gpu']}_{new_profile.string}_{int(self._current_time)}"
            new_eng = LLMEngineImpl.create(
                payload["gpu"], new_eid, agent, new_profile, self._current_time
            )

            boot_plain: BootPlainPayload = {
                "engine_id": new_eid,
                "purpose": OperationPurpose.PLAIN,
            }
            self._events.add(new_eng.trigger_boot(boot_plain))
            self._engines[new_eid] = new_eng
            agent.add_engine(new_eng)
            self._logger.log_mig_merge_complete(self._current_time, new_eid)

    def _handle_shutdown_complete_split(self, payload: ShutdownSplitPayload):
        engine_id = payload["engine_id"]
        agent = self._agents[payload["agent_id"]]
        e = self._engines.pop(engine_id, None)
        if e is not None and e in agent.engines:
            agent.engines.remove(e)

        new_profiles = payload["new_profiles"]
        # Pre-compute all new engine IDs so every sibling payload
        # shares the same sibling_engine_ids list reference.
        new_eids = [
            f"GPU_{payload['gpu']}_{p.string}_{int(self._current_time)}_{i}"
            for i, p in enumerate(new_profiles)
        ]
        sibling_ids: List[str] = new_eids
        for new_eid, p in zip(new_eids, new_profiles):
            new_eng = LLMEngineImpl.create(
                payload["gpu"], new_eid, agent, p, self._current_time
            )
            boot_split: BootSplitPayload = {
                "engine_id": new_eid,
                "purpose": OperationPurpose.SPLIT,
                "new_profiles": new_profiles,
                "agent_id": payload["agent_id"],
                "gpu": payload["gpu"],
                "sibling_engine_ids": sibling_ids,
            }
            self._events.add(new_eng.trigger_boot(boot_split))
            self._engines[new_eid] = new_eng
            agent.add_engine(new_eng)
        self._logger.log_mig_split_complete(self._current_time, engine_id)

    def _handle_shutdown_complete(self, event: SimulationEvent):
        payload = event.payload
        purpose = payload["purpose"]
        match purpose:
            case OperationPurpose.REALLOCATE:
                self._handle_shutdown_complete_reallocate(payload)
            case OperationPurpose.MERGE:
                self._handle_shutdown_complete_merge(payload)
            case OperationPurpose.SPLIT:
                self._handle_shutdown_complete_split(payload)

    def _handle_boot_complete(self, event: SimulationEvent) -> None:
        """Handle ENGINE_BOOT_COMPLETE.

        Activates the engine and processes the agent's waiting queue.
        Returns early if sibling engines from a split are still booting.
        """
        payload = event.payload
        engine_id = payload["engine_id"]

        engine = self._engines[engine_id]
        engine.activate(self._current_time)
        agent = engine.owner

        boot_purpose = payload.get("purpose")

        if boot_purpose == OperationPurpose.REALLOCATE:
            bd_realloc = cast(BootReallocatePayload, payload)
            self._logger.log_engine_boot_complete(
                self._current_time,
                engine_id,
                bd_realloc["giver_id"],
                bd_realloc["receiver_id"],
            )
        else:
            self._logger.log_engine_boot_complete(self._current_time, engine_id)

        # Defer waiting-queue processing if split siblings are still booting
        if boot_purpose == OperationPurpose.SPLIT:
            bd_split = cast(BootSplitPayload, payload)
            still_booting = any(
                self._engines[sid].status == EngineStatus.BOOTING
                for sid in bd_split["sibling_engine_ids"]
                if sid != engine_id and sid in self._engines
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

    def _handle_request_arrival(self, event: SimulationEvent):
        payload = cast(RequestArrivalPayload, event.payload)
        req: Request = payload["request"]
        agent_id = payload["target_agent"]

        agent = self._agents[agent_id]
        engine = agent.dispatch(req, self._current_time)
        self._logger.log_request_arrival(
            self._current_time,
            req.id,
            agent_id,
            engine,
            self._peak_next_stopping_evt(agent_id),
        )

        if engine and len(engine.running_queue) == 0 and len(engine.waiting_queue) > 0:
            evt = engine.step(
                self._current_time,
                next_arrival_time=self._peak_next_stopping_evt(agent_id),
            )
            if evt:
                self._events.add(evt)

    def _handle_engine_step_complete(self, event: SimulationEvent):
        engine_id = event.payload["engine_id"]
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
            self._current_time = current_event.time

            match current_event.event_type:
                case EventType.REALLOCATION_TRIGGER:
                    self._handle_reallocation_trigger()
                case EventType.MIG_TRIGGER:
                    self._handle_mig_trigger()
                case EventType.ENGINE_SHUTDOWN_COMPLETE:
                    self._handle_shutdown_complete(current_event)
                case EventType.ENGINE_BOOT_COMPLETE:
                    self._handle_boot_complete(current_event)
                case EventType.REQUEST_ARRIVAL:
                    self._handle_request_arrival(current_event)
                case EventType.ENGINE_STEP_COMPLETE:
                    self._handle_engine_step_complete(current_event)

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
            if (
                (
                    evt.event_type == EventType.REQUEST_ARRIVAL
                    and evt.payload["target_agent"] == agent_id
                )
                or evt.event_type == EventType.MIG_TRIGGER
                or evt.event_type == EventType.REALLOCATION_TRIGGER
            ):
                return evt.time
        return None
