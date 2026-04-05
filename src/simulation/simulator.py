from __future__ import annotations

from collections import defaultdict
from sortedcontainers import SortedList
from typing import Any, List, Dict, Tuple, cast

import src.simulation.models as m
import src.simulation.utils as utils
from src.simulation.agent import AgentImpl
from src.simulation.worker import WorkerImpl
from src.simulation.engine import LLMEngineImpl
from src.simulation.logger import SimulationLoggerImpl
from src.simulation.environment_state import EnvironmentStateImpl
from src.training.config import TRAINING_CONFIG


class SimulatorImpl(m.Simulator):
    def __init__(
        self,
        agents: Dict[m.AgentId, m.Agent],
        engines: Dict[str, m.LLMEngine],
        no_log: bool = False,
    ):
        self._agents = agents
        self._engines = engines
        self._events: SortedList[m.SimulationEvent] = SortedList()
        self._current_time: float = 0.0
        self.worker = WorkerImpl()
        self._logger = SimulationLoggerImpl(enabled=not no_log)

        self._environment_state = EnvironmentStateImpl()
        self._environment_state.reset_for_next_interval(0.0, self._agents)

    @property
    def agents(self) -> Dict[m.AgentId, m.Agent]:
        return self._agents

    @property
    def engines(self) -> Dict[str, m.LLMEngine]:
        return self._engines

    @property
    def events(self) -> SortedList[m.SimulationEvent]:
        return self._events

    @property
    def current_time(self) -> float:
        return self._current_time

    @property
    def logger(self) -> m.SimulationLogger:
        return self._logger

    @property
    def environment_state(self) -> m.EnvironmentState:
        return self._environment_state

    def need_requests_replenish(self) -> List[m.AgentId]:
        counts = {aid: 0 for aid in m.AgentId}
        for e in self._events:
            if e.event_type == m.EventType.REQUEST_ARRIVAL:
                e.payload = cast(m.RequestArrivalPayload, e.payload)
                counts[e.payload["request"].agent_id] += 1
            if all(c >= 1000 for c in counts.values()):
                break
        return [aid for aid, c in counts.items() if c < 1000]

    def latest_arrival_time(self, agent_id: m.AgentId) -> float:
        for e in reversed(self._events):
            if e.event_type == m.EventType.REQUEST_ARRIVAL:
                e.payload = cast(m.RequestArrivalPayload, e.payload)
                if e.payload["request"].agent_id == agent_id:
                    return e.time
        return self._current_time

    def add_arrival_events(self, requests: List[m.Request]) -> None:
        for req in requests:
            self._events.add(
                m.SimulationEvent(
                    time=req.arrival_time,
                    event_type=m.EventType.REQUEST_ARRIVAL,
                    payload={"request": req},
                )
            )

        self._sample_rag_searches(requests)

    def _sample_rag_searches(self, requests: List[m.Request]) -> None:
        search_evts: List[m.SimulationEvent] = []
        for req in requests:
            if req.agent_id == m.AgentId.RAG:
                search_evts.append(
                    m.SimulationEvent(
                        time=req.arrival_time + utils.SIM_CONFIG.get_rag_overhead(),
                        event_type=m.EventType.RAG_SEARCH_COMPLETE,
                        payload={"request": req},
                    )
                )

        self._events.update(search_evts)

    def init_simulator(self, requests: List[m.Request], max_steps: int) -> None:
        # Fill up requests
        self.add_arrival_events(requests)

        # Add RL action events
        for i in range(1, max_steps + 2):  # extra +1 as the stopping condition
            t = i * TRAINING_CONFIG.action_interval
            self._events.add(
                m.SimulationEvent(
                    time=t,
                    event_type=m.EventType.RESOURCE_MANAGER_TRIGGER,
                    payload={},
                )
            )

        # Add budget refresh events
        self._events.add(
            m.SimulationEvent(
                time=0.0,
                event_type=m.EventType.REFRESH_ACTION_BUDGET,
                payload={},
            )
        )

        # Activate engines
        for e in self._engines.values():
            e.activate(self._current_time)

    def has_active_work(self) -> bool:
        if any(
            e.event_type
            in [
                m.EventType.REQUEST_ARRIVAL,
                m.EventType.RAG_SEARCH_COMPLETE,
                m.EventType.ENGINE_STEP_COMPLETE,
            ]
            for e in self._events
        ):
            return True
        if any(
            len(e.waiting_queue) > 0 or len(e.running_queue) > 0
            for e in self._engines.values()
        ):
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
                eg.status in [m.EngineStatus.ACTIVE, m.EngineStatus.DRAINING]
                and len(eg.running_queue) == 0
                and len(eg.waiting_queue) > 0
            ):
                evt = eg.step(
                    self._current_time,
                    next_arrival_time=(
                        self._peak_next_stopping_evt(eg.owner.agent_id)
                        if eg.status == m.EngineStatus.ACTIVE
                        else None
                    ),
                )
                if evt:
                    self._events.add(evt)

    def _handle_resource_manager_trigger_vram_transfer(
        self, vram_transfer: m.TransferDetails
    ):
        transfer_decision = self.worker.transfer(vram_transfer, self._agents)
        assert transfer_decision is not None
        # if not transfer_decision:
        #     self._logger.log_discard_vram_transfer(self._current_time, vram_transfer)
        #     return
        self._environment_state.reconfig_flag = True

        action_type, data = transfer_decision
        match action_type:
            case "exact":
                engine_to_shift = data["engine"]
                giver = data["giver"]
                receiver = data["receiver"]
                amount = data["amount"]

                shutdown_payload: m.ShutdownReallocatePayload = {
                    "engine_id": engine_to_shift.engine_id,
                    "purpose": m.OperationPurpose.REALLOCATE,
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

                merge_payload: m.ShutdownMergePayload = {
                    "engine_id": engs[0].engine_id,
                    "purpose": m.OperationPurpose.MERGE,
                    "merge_engine_ids": tuple(e.engine_id for e in engs),
                    "drained_ids": [],
                    "new_profile": new_profile,
                    "agent_id": giver.agent_id,
                    "gpu": engs[0].gpu,
                    "receiver_id": receiver.agent_id,
                }
                for e in engs:
                    per_engine_payload: m.ShutdownMergePayload = {
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

                split_payload: m.ShutdownSplitPayload = {
                    "engine_id": eng.engine_id,
                    "purpose": m.OperationPurpose.SPLIT,
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
        self._environment_state.reconfig_flag = True
        action_type, data = mig_decision
        match action_type:
            case "merge":
                engs: List[m.LLMEngine] = data["engines"]
                eids = [e.engine_id for e in engs]
                merge_payload: m.ShutdownMergePayload = {
                    "engine_id": engs[0].engine_id,
                    "purpose": m.OperationPurpose.MERGE,
                    "merge_engine_ids": tuple(eids),
                    "drained_ids": [],
                    "new_profile": data["new_profile"],
                    "agent_id": data["agent_id"],
                    "gpu": data["gpu"],
                    "receiver_id": None,
                }
                for e in engs:
                    per_engine_payload: m.ShutdownMergePayload = {
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
                e: m.LLMEngine = data["engine"]
                split_payload: m.ShutdownSplitPayload = {
                    "engine_id": e.engine_id,
                    "purpose": m.OperationPurpose.SPLIT,
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

    def _predict_action_cost(self, action: m.ResourceManagerAction) -> float:
        """Predict the reconfiguration cost (drain + boot time) for an action."""
        val = action.value
        cost = 0.0

        if isinstance(val, m.VramTransferAction):
            vram_transfer = m.TransferDetails(
                amount=val.amount,
                giver_id=val.giver,
                receiver_id=val.receiver,
            )
            transfer_decision = self.worker.transfer(vram_transfer, self._agents)
            if transfer_decision:
                atype, data = transfer_decision
                match atype:
                    case "exact":
                        affected = [data["engine"]]
                        boot_cost = affected[0]._restart_time
                    case "merge":
                        affected = data["engines"]
                        boot_cost = utils.SIM_CONFIG.get_restart_time(
                            data["receiver"].agent_id, data["new_profile"]
                        )
                    case "split":
                        affected = [data["engine"]]
                        boot_cost = utils.SIM_CONFIG.get_restart_time(
                            data["receiver"].agent_id, data["mig_to_transfer"]
                        )
                    case _:
                        raise ValueError(f"Unknown transfer action type: {atype}")
                drain_cost = max(
                    (e.predict_drain_time() for e in affected), default=0.0
                )
                cost = drain_cost + boot_cost

        elif isinstance(val, m.MigAction):
            victim = self._agents[val.victim]
            match val.action:
                case "split":
                    best_split = utils.MIG_RULES.get_best_specific_split(
                        victim, val.profiles
                    )
                    if best_split:
                        eng, new_profiles = best_split
                        drain_cost = eng.predict_drain_time()
                        boot_cost = max(
                            (
                                utils.SIM_CONFIG.get_restart_time(val.victim, p)
                                for p in new_profiles
                            ),
                            default=0.0,
                        )
                        cost = drain_cost + boot_cost
                case "merge":
                    best_merge = utils.MIG_RULES.get_best_specific_merge(
                        victim, val.profiles
                    )
                    if best_merge:
                        engs, new_profile = best_merge
                        drain_cost = max(
                            (e.predict_drain_time() for e in engs), default=0.0
                        )
                        boot_cost = utils.SIM_CONFIG.get_restart_time(
                            val.victim, new_profile
                        )
                        cost = drain_cost + boot_cost
                case _:
                    raise ValueError(f"Unknown MIG action: {val.action}")
        else:
            # NO ACTION
            cost = 0.0

        return cost

    def handle_resource_manager_trigger(self, action: m.ResourceManagerAction):
        self._environment_state.record_queue_length_advance(
            self._current_time, self._agents
        )
        state_data = self._environment_state.get_state(
            self._current_time, self._agents, self._engines
        )
        self._logger.log_environment_state(self._current_time, state_data)
        self._environment_state.reset_for_next_interval(
            self._current_time, self._agents
        )

        for aid in self._agents.keys():
            self._environment_state.steps_since_split[aid] += 1
            self._environment_state.steps_since_merge[aid] += 1

        if action != m.ResourceManagerAction.NO_ACTION and isinstance(
            action.value, m.MigAction
        ):
            aid = action.value.victim
            if action.value.action == "split":
                self._environment_state.steps_since_split[aid] = 0
            elif action.value.action == "merge":
                self._environment_state.steps_since_merge[aid] = 0

        # 1. Calculate and deduct cost
        cost = self._predict_action_cost(action)
        self._environment_state.current_budget -= cost
        self._environment_state.last_action_downtime = cost

        # 2. Perform action
        match action:
            case m.ResourceManagerAction.NO_ACTION:
                pass

            case (
                m.ResourceManagerAction.TRANSFER_10_CODING_RAG
                | m.ResourceManagerAction.TRANSFER_10_RAG_CODING
                | m.ResourceManagerAction.TRANSFER_20_CODING_RAG
                | m.ResourceManagerAction.TRANSFER_20_RAG_CODING
            ):
                v_action = action.value
                vram_transfer = m.TransferDetails(
                    amount=v_action.amount,
                    giver_id=v_action.giver,
                    receiver_id=v_action.receiver,
                )
                self._handle_resource_manager_trigger_vram_transfer(vram_transfer)

            case action if action.value.action == "split":
                m_action = action.value
                agent_id = m_action.victim
                agent = self._agents[agent_id]
                best_split = utils.MIG_RULES.get_best_specific_split(
                    agent, m_action.profiles
                )
                assert best_split is not None  # correctness of the action mask

                eng, new_profiles = best_split
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

            case action if action.value.action == "merge":
                m_action = action.value
                agent_id = m_action.victim
                agent = self._agents[agent_id]
                best_merge = utils.MIG_RULES.get_best_specific_merge(
                    agent, m_action.profiles
                )
                assert best_merge is not None  # correctness of the action mask

                engs, new_profile = best_merge
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

        self._step_draining_or_active_engines()

    def _handle_shutdown_complete_reallocate(
        self, payload: m.ShutdownReallocatePayload
    ):
        engine_id = payload["engine_id"]
        receiver_id = payload["receiver_id"]
        engine = self._engines[engine_id]

        new_model = utils.SIM_CONFIG.get_model(receiver_id, engine.mig_profile)

        receiver = self._agents[receiver_id]

        engine.update_model(
            new_owner=receiver,
            model_name=new_model,
            max_batched_tokens=utils.SIM_CONFIG.max_batched_tokens[new_model],
            prefill_params=utils.SIM_CONFIG.get_prefill_params(
                receiver_id, engine.mig_profile
            ),
            tpot_params=utils.SIM_CONFIG.get_tpot_params(
                receiver_id, engine.mig_profile
            ),
            restart_time=utils.SIM_CONFIG.get_restart_time(
                receiver_id, engine.mig_profile
            ),
        )

        boot_payload: m.BootPayload = {
            "engine_id": engine_id,
            "purpose": m.OperationPurpose.REALLOCATE,
            "sibling_engine_ids": None,
        }
        self._events.add(engine.trigger_boot(boot_payload))

    def _handle_shutdown_complete_merge(self, payload: m.ShutdownMergePayload):
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

            new_eid = utils.generate_engine_id(payload["gpu"], new_profile.string)
            new_eng = LLMEngineImpl.create(
                payload["gpu"], new_eid, target_agent, new_profile, self._current_time
            )

            boot_plain: m.BootPayload = {
                "engine_id": new_eid,
                "purpose": m.OperationPurpose.PLAIN,
                "sibling_engine_ids": None,
            }
            self._events.add(new_eng.trigger_boot(boot_plain))
            self._engines[new_eid] = new_eng
            target_agent.add_engine(new_eng)
            self._logger.log_mig_merge_complete(self._current_time, new_eid)

    def _handle_shutdown_complete_split(self, payload: m.ShutdownSplitPayload):
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
        new_owners: List[m.Agent] = []
        transfer_received = False
        for p in new_profiles:
            new_owner = agent
            if is_vram_transfer and p == received_profile and not transfer_received:
                new_owner = self._agents[receiver_id]
                transfer_received = True
            new_owners.append(new_owner)

        new_eids = [utils.generate_engine_id(gpu, p.string) for p in new_profiles]

        for new_eid, new_owner, p in zip(new_eids, new_owners, new_profiles):
            new_eng = LLMEngineImpl.create(
                gpu, new_eid, new_owner, p, self._current_time
            )
            new_owner.add_engine(new_eng)
            self._engines[new_eid] = new_eng

            boot_split: m.BootPayload = {
                "engine_id": new_eid,
                "purpose": m.OperationPurpose.SPLIT,
                "sibling_engine_ids": new_eids,
            }

            self._events.add(new_eng.trigger_boot(boot_split))
        self._logger.log_mig_split_complete(self._current_time, engine_id)

    def _handle_shutdown_complete(
        self,
        payload: m.ShutdownPayload,
    ):
        purpose = payload["purpose"]
        match purpose:
            case m.OperationPurpose.REALLOCATE:
                self._handle_shutdown_complete_reallocate(
                    cast(m.ShutdownReallocatePayload, payload)
                )
            case m.OperationPurpose.MERGE:
                self._handle_shutdown_complete_merge(
                    cast(m.ShutdownMergePayload, payload)
                )
            case m.OperationPurpose.SPLIT:
                self._handle_shutdown_complete_split(
                    cast(m.ShutdownSplitPayload, payload)
                )
            case _:
                raise ValueError(f"Unexpected shutdown purpose {purpose}")

    def _handle_boot_complete(self, payload: m.BootPayload) -> None:
        engine_id = payload["engine_id"]

        engine = self._engines[engine_id]
        engine.activate(self._current_time)
        agent = engine.owner

        self._logger.log_engine_boot_complete(self._current_time, engine_id)

        # Defer waiting-queue processing if split siblings are still booting
        if payload["purpose"] == m.OperationPurpose.SPLIT:
            assert payload["sibling_engine_ids"] is not None
            still_booting = any(
                self._engines[sid].status == m.EngineStatus.BOOTING
                for sid in payload["sibling_engine_ids"]
                if sid != engine_id
                and self._engines[sid].owner.agent_id == agent.agent_id
            )
            if still_booting:
                return  # Wait for remaining sibling engines
        self._environment_state.reconfig_flag = False  # reconfig action done

    def _handle_request_arrival(self, payload: m.RequestArrivalPayload):
        req = payload["request"]
        agent_id = req.agent_id

        self._environment_state.register_arrival(req)

        # If it's a RAG request, wiat til RAG_SEARCH_COMPLETE
        if agent_id == m.AgentId.RAG:
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

    def _handle_rag_search_complete(self, payload: m.RequestArrivalPayload):
        req = payload["request"]
        agent_id = req.agent_id
        assert agent_id == m.AgentId.RAG

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

    def _handle_engine_step_complete(self, payload: m.EngineStepPayload):
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
            if current_event.event_type == m.EventType.RESOURCE_MANAGER_TRIGGER:
                self._events.pop(0)
                self._current_time = current_event.time
                return True  # Reached trigger point

            current_event = self._events.pop(0)
            self._environment_state.record_queue_length_advance(
                current_event.time, self._agents
            )
            self._current_time = current_event.time

            match current_event.event_type:
                case m.EventType.ENGINE_SHUTDOWN_COMPLETE:
                    self._handle_shutdown_complete(
                        cast(m.ShutdownPayload, current_event.payload)
                    )
                case m.EventType.ENGINE_BOOT_COMPLETE:
                    self._handle_boot_complete(
                        cast(m.BootPayload, current_event.payload)
                    )
                case m.EventType.REQUEST_ARRIVAL:
                    self._handle_request_arrival(
                        cast(m.RequestArrivalPayload, current_event.payload)
                    )
                case m.EventType.RAG_SEARCH_COMPLETE:
                    self._handle_rag_search_complete(
                        cast(m.RequestArrivalPayload, current_event.payload)
                    )
                case m.EventType.ENGINE_STEP_COMPLETE:
                    self._handle_engine_step_complete(
                        cast(m.EngineStepPayload, current_event.payload)
                    )
                case m.EventType.REFRESH_ACTION_BUDGET:
                    self._environment_state.refresh_budget()
                    self._events.add(
                        m.SimulationEvent(
                            time=self._current_time + TRAINING_CONFIG.refresh_period,
                            event_type=m.EventType.REFRESH_ACTION_BUDGET,
                            payload={},
                        )
                    )
                case m.EventType.RESOURCE_MANAGER_TRIGGER:
                    raise ValueError(f"Unexpected event {current_event.event_type}")

        return False  # Finished simulation

    def reset(self) -> None:
        """Resets the simulator to its initial hardware and agent state."""
        self._current_time = 0.0
        self._events.clear()
        self._agents.clear()
        self._engines.clear()
        utils.USED_EIDS.clear()

        for aid in m.AgentId:
            self._agents[aid] = AgentImpl(aid)

        for eng_conf in utils.SIM_CONFIG.initial_state:
            mig = m.MIGProfile.from_string(eng_conf["mig"])
            gpu = int(eng_conf["gpu"])
            agent_name = eng_conf["agent"]
            agent = self._agents[m.AgentId(agent_name)]
            eid = utils.generate_engine_id(gpu, mig.string)

            is_permanent = eng_conf.get("is-permanent", False)
            eng = LLMEngineImpl.create(
                gpu=gpu,
                engine_id=eid,
                owner=agent,
                mig_profile=mig,
                current_time=0.0,
                is_permanent=is_permanent,
            )
            agent.add_engine(eng)
            self._engines[eid] = eng

        self._environment_state.reset_for_next_interval(0.0, self._agents)
        self._environment_state.reconfig_flag = False
        self._environment_state.steps_since_split = {
            aid: 5 for aid in self._agents.keys()
        }
        self._environment_state.steps_since_merge = {
            aid: 5 for aid in self._agents.keys()
        }
        self._events.add(
            m.SimulationEvent(
                time=0.0,
                event_type=m.EventType.REFRESH_ACTION_BUDGET,
                payload={},
            )
        )

    def get_action_mask(self) -> List[bool]:
        mask: List[bool] = [False] * len(m.ResourceManagerAction)
        if self.environment_state.reconfig_flag:
            mask[
                list(m.ResourceManagerAction).index(m.ResourceManagerAction.NO_ACTION)
            ] = True
            return mask

        for act_id, action in enumerate(m.ResourceManagerAction):
            if action == m.ResourceManagerAction.NO_ACTION:
                mask[act_id] = True
                continue

            val = action.value
            if isinstance(val, m.VramTransferAction):
                giver = self._agents[val.giver]
                active_vram: Dict[int, int] = defaultdict(int)
                for e in giver.engines:
                    if e.status == m.EngineStatus.ACTIVE and not e.is_permanent:
                        active_vram[e.gpu] += e.mig_profile.vram
                mask[act_id] = any(v >= val.amount for v in active_vram.values())
            else:  # MIGAction
                victim = self._agents[val.victim]
                match val.action:
                    case "split":
                        mask[act_id] = (
                            utils.MIG_RULES.get_best_specific_split(
                                victim, val.profiles
                            )
                            is not None
                        )
                    case "merge":
                        mask[act_id] = (
                            utils.MIG_RULES.get_best_specific_merge(
                                victim, val.profiles
                            )
                            is not None
                        )
                    case _:
                        raise ValueError(f"Unknown MIG action: {val.action}")

            # Additional budget check
            if mask[act_id]:  # Only check if the action is otherwise possible
                cost = self._predict_action_cost(action)
                if cost > self._environment_state.current_budget:
                    mask[act_id] = False

        assert len(mask) == len(m.ResourceManagerAction)
        return mask

    def _peak_next_stopping_evt(self, agent_id: m.AgentId) -> float | None:
        for evt in self._events:
            if evt.event_type == m.EventType.RESOURCE_MANAGER_TRIGGER:
                return evt.time

            if evt.event_type == m.EventType.REQUEST_ARRIVAL:
                payload = cast(m.RequestArrivalPayload, evt.payload)
                if (
                    payload["request"].agent_id == agent_id
                    and agent_id != m.AgentId.RAG
                ):
                    return evt.time

            if evt.event_type == m.EventType.RAG_SEARCH_COMPLETE:
                payload = cast(m.RequestArrivalPayload, evt.payload)
                if (
                    payload["request"].agent_id == agent_id
                    and agent_id == m.AgentId.RAG
                ):
                    return evt.time
        return None
