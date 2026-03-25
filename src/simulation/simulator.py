from __future__ import annotations

from sortedcontainers import SortedList
import uuid
from typing import Any, List, Dict, Tuple, cast, Mapping

from src.simulation.models import *
import src.simulation.global_vars as g
from src.simulation.worker import WorkerImpl
from src.simulation.engine import LLMEngineImpl
from src.simulation.logger import SimulationLoggerImpl
from src.simulation.environment_state import EnvironmentStateImpl

# from src.training.resource_manager import ResourceManager


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
        # self.resource_manager = ResourceManager()
        self.worker = WorkerImpl()
        self._logger = SimulationLoggerImpl(enabled=not no_log)

        self.environment_state = EnvironmentStateImpl(
            g.SIM_CONFIG.get_rl_action_interval()
        )
        self.environment_state.reset_for_next_interval(0.0, self._agents)

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

    @property
    def action_interval(self) -> float:
        return g.SIM_CONFIG.get_rl_action_interval()

    def add_arrival_events(self, requests: List[Request]) -> None:
        for req in requests:
            self._events.add(
                SimulationEvent(
                    time=req.arrival_time,
                    event_type=EventType.REQUEST_ARRIVAL,
                    payload={"request": req},
                )
            )

        self._sample_rag_searches(requests)

    def _sample_rag_searches(self, requests: List[Request]) -> None:
        search_evts: List[SimulationEvent] = []
        for req in requests:
            if req.agent_id == AgentId.RAG:
                search_evts.append(
                    SimulationEvent(
                        time=req.arrival_time + g.SIM_CONFIG.get_rag_overhead(),
                        event_type=EventType.RAG_SEARCH_COMPLETE,
                        payload={"request": req},
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

    def _handle_resource_manager_trigger_vram_transfer(
        self, vram_transfer: TransferDetails
    ):
        transfer_decision = self.worker.transfer(
            self._current_time, vram_transfer, self._agents
        )
        if not transfer_decision:
            self._logger.log_discard_vram_transfer(self._current_time, vram_transfer)
            return

        action_type, data = transfer_decision
        match action_type:
            case "exact":
                engine_to_shift = data["engine"]
                giver = data["giver"]
                receiver = data["receiver"]
                amount = data["amount"]

                shutdown_payload: ShutdownReallocatePayload = {
                    "engine_id": engine_to_shift.engine_id,
                    "purpose": OperationPurpose.REALLOCATE,
                    "receiver_id": receiver.agent_id,
                }
                self._logger.log_vram_transfer(
                    self._current_time,
                    giver.agent_id,
                    receiver.agent_id,
                    amount,
                    [engine_to_shift.engine_id],
                )
                evt = engine_to_shift.trigger_shutdown(
                    shutdown_payload, self._current_time
                )
                if evt:
                    self._events.add(evt)

            case "merge":
                engs = data["engines"]
                new_profile = data["new_profile"]
                giver = data["giver"]
                receiver = data["receiver"]
                amount = data["amount"]

                merge_payload: ShutdownMergePayload = {
                    "engine_id": engs[0].engine_id,
                    "purpose": OperationPurpose.MERGE,
                    "merge_engine_ids": tuple(e.engine_id for e in engs),
                    "drained_ids": [],
                    "new_profile": new_profile,
                    "agent_id": giver.agent_id,
                    "gpu": engs[0].gpu,
                    "receiver_id": receiver.agent_id,
                }
                for e in engs:
                    per_engine_payload: ShutdownMergePayload = {
                        **merge_payload,
                        "engine_id": e.engine_id,
                    }
                    evt = e.trigger_shutdown(per_engine_payload, self._current_time)
                    if evt:
                        self._events.add(evt)
                self._logger.log_mig_merge_trigger(
                    self._current_time, [e.engine_id for e in engs], engs[0].gpu
                )
                self._logger.log_vram_transfer(
                    self._current_time,
                    giver.agent_id,
                    receiver.agent_id,
                    amount,
                    [e.engine_id for e in engs],
                )

            case "split":
                eng = data["engine"]
                new_profiles = data["new_profiles"]
                mig_to_transfer = data["mig_to_transfer"]
                giver = data["giver"]
                receiver = data["receiver"]
                amount = data["amount"]

                split_payload: ShutdownSplitPayload = {
                    "engine_id": eng.engine_id,
                    "purpose": OperationPurpose.SPLIT,
                    "new_profiles": new_profiles,
                    "agent_id": giver.agent_id,
                    "gpu": eng.gpu,
                    "receiver_id": receiver.agent_id,
                    "received_profile": mig_to_transfer,
                }
                evt = eng.trigger_shutdown(split_payload, self._current_time)
                if evt:
                    self._events.add(evt)
                self._logger.log_mig_split_trigger(
                    self._current_time, eng.engine_id, eng.gpu
                )
                self._logger.log_vram_transfer(
                    self._current_time,
                    giver.agent_id,
                    receiver.agent_id,
                    amount,
                    [eng.engine_id],
                )

            case _:
                raise ValueError(f"Unknown action {action_type}")

    def _handle_resource_manager_trigger_mig_decision(
        self, mig_decision: Tuple[str, Any]
    ):
        action_type, data = mig_decision
        match action_type:
            case "merge":
                engs: List[LLMEngine] = data["engines"]
                eids = [e.engine_id for e in engs]
                merge_payload: ShutdownMergePayload = {
                    "engine_id": engs[0].engine_id,
                    "purpose": OperationPurpose.MERGE,
                    "merge_engine_ids": tuple(eids),
                    "drained_ids": [],
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
                    evt = e.trigger_shutdown(per_engine_payload, self._current_time)
                    if evt:
                        self._events.add(evt)
                self._logger.log_mig_merge_trigger(
                    self._current_time, eids, data["gpu"]
                )

            case "split":
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
                evt = e.trigger_shutdown(split_payload, self._current_time)
                if evt:
                    self._events.add(evt)
                self._logger.log_mig_split_trigger(
                    self._current_time, e.engine_id, data["gpu"]
                )

            case _:
                raise ValueError(f"Unknown action {action_type}")

    def handle_resource_manager_trigger(self, action: ResourceManagerAction):
        self.environment_state.record_queue_length_advance(
            self._current_time, self._agents
        )
        state_data = self.environment_state.get_state(
            self._current_time, self._agents, self._engines
        )
        self._logger.log_environment_state(self._current_time, state_data)

        match action:
            case ResourceManagerAction.NO_ACTION:
                pass

            case (
                ResourceManagerAction.TRANSFER_10_CODING_RAG
                | ResourceManagerAction.TRANSFER_10_RAG_CODING
                | ResourceManagerAction.TRANSFER_20_CODING_RAG
                | ResourceManagerAction.TRANSFER_20_RAG_CODING
            ):
                v_action = action.value
                vram_transfer = TransferDetails(
                    amount=v_action.amount,
                    giver_id=v_action.giver,
                    receiver_id=v_action.receiver,
                )
                self._handle_resource_manager_trigger_vram_transfer(vram_transfer)

            case ResourceManagerAction.SPLIT_CODING | ResourceManagerAction.SPLIT_RAG:
                m_action = action.value
                agent_id = m_action.victim
                agent = self._agents[agent_id]
                possible_splits = g.MIG_RULES.get_possible_splits(agent)
                if possible_splits:
                    eng, new_profiles = min(
                        possible_splits,
                        key=lambda c: len(c[0].running_queue) + len(c[0].waiting_queue),
                    )
                    mig_decision: Tuple[str, Any] = (
                        "split",
                        {
                            "engine": eng,
                            "new_profiles": new_profiles,
                            "agent_id": agent_id,
                            "gpu": eng.gpu,
                        },
                    )
                    self._handle_resource_manager_trigger_mig_decision(mig_decision)

            case ResourceManagerAction.MERGE_CODING | ResourceManagerAction.MERGE_RAG:
                m_action = action.value
                agent_id = m_action.victim
                agent = self._agents[agent_id]
                possible_merges = g.MIG_RULES.get_possible_merges(agent)
                if possible_merges:
                    engs, new_profile = min(
                        possible_merges,
                        key=lambda c: sum(
                            len(e.running_queue) + len(e.waiting_queue) for e in c[0]
                        ),
                    )
                    mig_decision = (
                        "merge",
                        {
                            "engines": engs,
                            "new_profile": new_profile,
                            "agent_id": agent_id,
                            "gpu": engs[0].gpu,
                        },
                    )
                    self._handle_resource_manager_trigger_mig_decision(mig_decision)

            case _:
                raise ValueError(f"Unknown RL action {action}")

        self.environment_state.reset_for_next_interval(self._current_time, self._agents)
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
            # If a receiver was specified (merge-for-transfer), boot directly on receiver
            receiver_id = payload.get("receiver_id")
            target_agent = self._agents[receiver_id] if receiver_id else giver_agent

            new_eid = g.generate_engine_id(
                target_agent.agent_id.value, payload["gpu"], new_profile.string
            )
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
        new_owners = []
        temp_has_received = False
        for p in new_profiles:
            new_owner = agent
            if is_vram_transfer and p == received_profile and not temp_has_received:
                new_owner = self._agents[receiver_id]
                temp_has_received = True
            new_owners.append(new_owner)

        new_eids = [
            g.generate_engine_id(own.agent_id.value, gpu, p.string)
            for own, p in zip(new_owners, new_profiles)
        ]

        for new_eid, new_owner, p in zip(new_eids, new_owners, new_profiles):
            new_eng = LLMEngineImpl.create(
                gpu, new_eid, new_owner, p, self._current_time
            )
            new_owner.add_engine(new_eng)
            self._engines[new_eid] = new_eng

            boot_split: BootPayload = {
                "engine_id": new_eid,
                "purpose": OperationPurpose.SPLIT,
                "sibling_engine_ids": new_eids,
            }

            self._events.add(new_eng.trigger_boot(boot_split))
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

        self.environment_state.register_arrival(agent_id, self._current_time, req)

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

    def run(self) -> bool:
        """Executes events until a RESOURCE_MANAGER_TRIGGER is popped."""
        while self.has_active_work():
            if not self._events:
                self._step_idle_engines()
                if not self.has_active_work():
                    break

            current_event = self._events[0]
            if current_event.event_type == EventType.RESOURCE_MANAGER_TRIGGER:
                self._events.pop(0)
                self._current_time = current_event.time
                return True  # Reached trigger point

            current_event = self._events.pop(0)
            self.environment_state.record_queue_length_advance(
                current_event.time, self._agents
            )
            self._current_time = current_event.time

            match current_event.event_type:
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
                case EventType.RESOURCE_MANAGER_TRIGGER:
                    raise ValueError(f"Unexpected event {current_event.event_type}")

        return False  # Finished simulation

    def reset(self) -> None:
        """Resets the simulator to its initial hardware and agent state."""
        self._current_time = 0.0
        self._events.clear()
        self._agents.clear()
        self._engines.clear()

        for aid in AgentId:
            self._agents[aid] = AgentImpl(aid)

        for eng_conf in g.SIM_CONFIG.initial_state:
            mig = MIGProfile.from_string(eng_conf["mig"])
            gpu = int(eng_conf["gpu"])
            eid = f"GPU_{gpu}_{mig.string}"
            agent = self._agents[AgentId(eng_conf["agent"])]

            eng = LLMEngineImpl.create(
                gpu=gpu,
                engine_id=eid,
                owner=agent,
                mig_profile=mig,
                current_time=0.0,
            )
            agent.add_engine(eng)
            self._engines[eid] = eng

        self.environment_state.reset_for_next_interval(0.0, self._agents)

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
