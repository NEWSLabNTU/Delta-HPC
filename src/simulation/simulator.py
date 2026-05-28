from __future__ import annotations
from collections import defaultdict
from sortedcontainers import SortedList
from typing import List, Dict, Literal, Optional, cast

import src.share.models as m
import src.simulation.models as sm
import src.simulation.utils as utils
from src.simulation.agent import AgentImpl
from src.simulation.engine import LLMEngineImpl
from src.simulation.logger import SimulationLoggerImpl
from src.simulation.environment_state import EnvironmentStateImpl
from src.simulation.config import (
    GPU_MIG_PROFILE,
    GPU_VALID_COMBINATIONS,
)
from src.share.mig_matrix import (
    STATE_DEFINITIONS,
    TRANSITION_MATRIX,
    map_res_action_to_action,
)
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
        self._events: SortedList[sm.SimulationEvent] = SortedList()
        self._current_time: float = 0.0
        self._logger = SimulationLoggerImpl(enabled=not no_log)
        self._comming_budget_refresh: Optional[sm.SimulationEvent] = None

        self._environment_state = EnvironmentStateImpl()
        self._environment_state.reset_for_next_interval(0.0, self._agents)

        self._gpu_engines: Dict[int, List[m.LLMEngine]] = {
            gpu: [] for gpu in utils.SIM_CONFIG.cluster.keys()
        }
        if self._engines:
            self._sync_ownership()

    @property
    def agents(self) -> Dict[m.AgentId, m.Agent]:
        return self._agents

    @property
    def engines(self) -> Dict[str, m.LLMEngine]:
        return self._engines

    @property
    def events(self) -> SortedList[sm.SimulationEvent]:
        return self._events

    @property
    def current_time(self) -> float:
        return self._current_time

    @property
    def logger(self) -> sm.SimulationLogger:
        return self._logger

    @property
    def interval_requests(self) -> Dict[m.AgentId, List[m.Request]]:
        return self._environment_state.interval_requests

    @property
    def gpu_engines(self) -> Dict[int, List[m.LLMEngine]]:
        return self._gpu_engines

    def need_requests_replenish(self) -> List[m.AgentId]:
        return [
            aid
            for aid in m.AgentId
            if self._environment_state.get_pending_request_count(aid) < 1000
        ]

    def latest_arrival_time(self, agent_id: m.AgentId) -> float:
        for e in reversed(self._events):
            if e.event_type == m.EventType.REQUEST_ARRIVAL:
                e.payload = cast(sm.RequestArrivalPayload, e.payload)
                if e.payload["request"].agent_id == agent_id:
                    return e.time
        return self._current_time

    def add_arrival_events(self, requests: List[m.Request]) -> None:
        counts: Dict[m.AgentId, int] = defaultdict(int)
        for req in requests:
            counts[req.agent_id] += 1
            self._events.add(
                sm.SimulationEvent(
                    time=req.arrival_time,
                    event_type=m.EventType.REQUEST_ARRIVAL,
                    payload={"request": req},
                )
            )
        for aid, cnt in counts.items():
            self._environment_state.add_pending_request_count(aid, cnt)

        self._sample_rag_searches(requests)

    def _sample_rag_searches(self, requests: List[m.Request]) -> None:
        search_evts: List[sm.SimulationEvent] = []
        for req in requests:
            if req.agent_id == m.AgentId.RAG:
                search_evts.append(
                    sm.SimulationEvent(
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
                sm.SimulationEvent(
                    time=t,
                    event_type=m.EventType.RESOURCE_MANAGER_TRIGGER,
                    payload={},
                )
            )

        # Activate engines
        for e in self._engines.values():
            e.activate(self._current_time)

    @property
    def gpu_current_state(self) -> Dict[int, int]:
        return {
            gpu_id: self._identify_gpu_state(gpu_id)
            for gpu_id in range(utils.SIM_CONFIG.num_managed_gpus)
        }

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

    def _handle_resource_manager_trigger_mig_decision(self, action: m.Action):
        assert action.target_state_id is not None
        self._environment_state.reconfig_flag = True
        if action.action == m.ActionType.MERGE:
            engs = [self._gpu_engines[action.gpu_id][idx] for idx in action.mig_src]
            assert all(e is not None for e in engs)
            eids = [e.engine_id for e in engs]
            merge_payload: sm.ShutdownMergePayload = {
                "engine_id": engs[0].engine_id,
                "purpose": sm.OperationPurpose.MERGE,
                "merge_engine_ids": tuple(eids),
                "drained_ids": [],
                "target_mig_indices": action.mig_target,
                "agent_id": engs[0].owner.agent_id,
                "gpu": action.gpu_id,
                "receiver_id": action.receiver.receiver_id if action.receiver else None,
                "target_state_id": action.target_state_id,
            }
            for e in engs:
                per_engine_payload: sm.ShutdownMergePayload = {
                    **merge_payload,
                    "engine_id": e.engine_id,
                }
                evt = e.trigger_shutdown(per_engine_payload, self._current_time)
                if evt:
                    self._events.add(evt)
            self._logger.log_mig_merge_trigger(self._current_time, eids, action.gpu_id)

        elif action.action == m.ActionType.SPLIT:
            eng = self._gpu_engines[action.gpu_id][action.mig_src[0]]
            assert eng is not None
            received_profile = (
                STATE_DEFINITIONS[action.target_state_id][action.receiver.mig_idx]
                if action.receiver
                else None
            )
            split_payload: sm.ShutdownSplitPayload = {
                "engine_id": eng.engine_id,
                "purpose": sm.OperationPurpose.SPLIT,
                "target_mig_indices": action.mig_target,
                "agent_id": eng.owner.agent_id,
                "gpu": action.gpu_id,
                "receiver_id": action.receiver.receiver_id if action.receiver else None,
                "received_profile": received_profile,
                "target_state_id": action.target_state_id,
            }
            evt = eng.trigger_shutdown(split_payload, self._current_time)
            if evt:
                self._events.add(evt)
            self._logger.log_mig_split_trigger(
                self._current_time, eng.engine_id, action.gpu_id
            )

    def _handle_resource_manager_trigger_vram_transfer(self, action: m.Action) -> None:
        assert action.receiver is not None
        engine_to_shift = self._gpu_engines[action.gpu_id][action.mig_src[0]]
        assert engine_to_shift is not None

        self._environment_state.reconfig_flag = True

        shutdown_payload: sm.ShutdownReallocatePayload = {
            "engine_id": engine_to_shift.engine_id,
            "purpose": sm.OperationPurpose.REALLOCATE,
            "receiver_id": action.receiver.receiver_id,
        }
        self._logger.log_vram_transfer(
            self._current_time,
            engine_to_shift.owner.agent_id,
            action.receiver.receiver_id,
            engine_to_shift.mig_profile.size,
            [engine_to_shift.engine_id],
        )
        evt = engine_to_shift.trigger_shutdown(shutdown_payload, self._current_time)
        if evt:
            self._events.add(evt)

    def _invalidate_gpu_indices(
        self, gpu: int, pivot_idx: int, target_indices: List[int]
    ):
        """Invalidate indices affected by a state transition to avoid conflicts during sync.

        Args:
            gpu: The GPU ID.
            pivot_idx: The starting index of the split or merge operation.
            target_indices: The indices targeted by the new configuration.
        """
        if pivot_idx == -1:
            return

        # Indices to invalidate:
        # 1. Those after the pivot point (following physical MIG shift logic)
        # 2. Those explicitly targeted by the new config (to avoid overlaps)
        affected_indices = {i for i in range(pivot_idx + 1, 16)}
        affected_indices.update(target_indices)

        for eng in self._engines.values():
            if eng.gpu == gpu and eng.mig_index in affected_indices:
                eng.mig_index = -1

    def _sync_ownership(self):
        """Synchronize the slice ownership map and engine index list."""
        self._gpu_engines = {gpu: [] for gpu in utils.SIM_CONFIG.cluster.keys()}

        for gpu_id in utils.SIM_CONFIG.cluster.keys():
            sid = self._identify_gpu_state(gpu_id)
            if sid is None:
                continue

            target_profiles = STATE_DEFINITIONS[sid]
            eng_list: List[Optional[m.LLMEngine]] = [None] * len(target_profiles)
            gpu_engs = [e for e in self._engines.values() if e.gpu == gpu_id]

            # 1. First pass: Keep engines that already have valid indices in the new state
            for eng in gpu_engs:
                idx = eng.mig_index
                if (
                    idx != -1
                    and idx < len(target_profiles)
                    and target_profiles[idx] == eng.mig_profile.profile_type
                ):
                    another_eng = eng_list[idx]
                    if another_eng is None:
                        eng_list[idx] = eng
                    else:
                        raise AssertionError(
                            f"GPU {gpu_id} index conflict at {idx}: "
                            f"Engine {eng.engine_id} and {another_eng.engine_id} "
                            f"both claim index {idx}"
                        )
                else:
                    # Index is either -1, out of bounds, or profile mismatch
                    eng.mig_index = -1

            # 2. Second pass: Assign indices to engines that need them
            for eng in gpu_engs:
                if eng.mig_index == -1:
                    # Find a slot that matches this engine's profile
                    for i, p in enumerate(target_profiles):
                        if eng_list[i] is None and p == eng.mig_profile.profile_type:
                            eng.mig_index = i
                            eng_list[i] = eng
                            break

            assert all(e is not None for e in eng_list), (
                f"GPU {gpu_id} state {sid} has missing engines: "
                f"Expected {target_profiles}, found {[e.mig_profile.profile_type if e else None for e in eng_list]}"
            )
            self._gpu_engines[gpu_id] = eng_list  # type: ignore

    def _predict_action_cost(self, action: m.Action) -> float:
        """Predict the reconfiguration cost (drain + boot time) for an action."""
        cost = 0.0

        if action.action == m.ActionType.TRANSFER:
            # Get specific engine by index
            assert action.receiver is not None
            engine_to_shift = self._gpu_engines[action.gpu_id][action.mig_src[0]]
            boot_cost = utils.SIM_CONFIG.get_restart_time(
                action.receiver.receiver_id,
                engine_to_shift.mig_profile,
                gpu_id=engine_to_shift.gpu,
            )
            drain_cost = engine_to_shift.predict_drain_time()
            cost = drain_cost + boot_cost
        elif action.action in [m.ActionType.SPLIT, m.ActionType.MERGE]:
            # Find the engine(s) that will be split or merged
            assert action.target_state_id is not None
            if action.action == m.ActionType.SPLIT:
                eng = self._gpu_engines[action.gpu_id][action.mig_src[0]]
                drain_cost = eng.predict_drain_time()
                boot_cost = max(
                    (
                        utils.SIM_CONFIG.get_restart_time(
                            action.receiver.receiver_id
                            if action.receiver and p == action.receiver.mig_idx
                            else eng.owner.agent_id,
                            self._get_hardware_profile(
                                eng.gpu, STATE_DEFINITIONS[action.target_state_id][p]
                            ),
                            gpu_id=eng.gpu,
                        )
                        for p in action.mig_target
                    ),
                    default=0.0,
                )
                cost = drain_cost + boot_cost
            else:  # MERGE
                engs = [self._gpu_engines[action.gpu_id][idx] for idx in action.mig_src]
                if engs:
                    drain_cost = max(
                        (e.predict_drain_time() for e in engs), default=0.0
                    )
                    new_profile_idx = action.mig_target[0]
                    boot_cost = utils.SIM_CONFIG.get_restart_time(
                        action.receiver.receiver_id
                        if action.receiver
                        and new_profile_idx == action.receiver.mig_idx
                        else engs[0].owner.agent_id,
                        self._get_hardware_profile(
                            engs[0].gpu,
                            STATE_DEFINITIONS[action.target_state_id][new_profile_idx],
                        ),
                        gpu_id=engs[0].gpu,
                    )
                    cost = drain_cost + boot_cost
        return cost

    def _find_best_engine_index(self, gpu_id: int, profile: m.MIGProfile) -> int:
        candidates = [
            (i, e)
            for i, e in enumerate(self._gpu_engines[gpu_id])
            if e is not None and e.mig_profile.profile_type == profile
        ]
        if not candidates:
            return -1

        idx, _ = min(
            candidates,
            key=lambda x: len(x[1].waiting_queue) + len(x[1].running_queue),
        )
        return idx

    def handle_resource_manager_trigger(self, action: Optional[m.Action]):
        self._environment_state.record_queue_length_advance(
            self._current_time, self._agents
        )
        state_data = self._environment_state.get_state(
            self._current_time, self._agents, self.gpu_current_state
        )
        self._logger.log_environment_state(self._current_time, state_data)
        self._environment_state.reset_for_next_interval(
            self._current_time, self._agents
        )

        self._environment_state.advance_all_last_action()

        if action is None:
            self._step_draining_or_active_engines()
            return

        if action.action == m.ActionType.SPLIT:
            eng = self._gpu_engines[action.gpu_id][action.mig_src[0]]
            assert eng is not None
            self._environment_state.set_last_action(eng.owner.agent_id, "split")
        elif action.action == m.ActionType.MERGE:
            eng = self._gpu_engines[action.gpu_id][action.mig_src[0]]
            assert eng is not None
            self._environment_state.set_last_action(eng.owner.agent_id, "merge")

        if action.receiver is not None:
            # Find giver
            eng = self._gpu_engines[action.gpu_id][action.mig_src[0]]
            assert eng is not None
            giver_id = eng.owner.agent_id

            sid = (
                action.target_state_id
                if action.target_state_id is not None
                else self.gpu_current_state[action.gpu_id]
            )
            mig_size = STATE_DEFINITIONS[sid][action.receiver.mig_idx].size

            self._environment_state.set_last_action(giver_id, "give", mig_size)
            self._environment_state.set_last_action(
                action.receiver.receiver_id, "receive", mig_size
            )

        if self._comming_budget_refresh is not None:
            self._events.remove(self._comming_budget_refresh)
        refresh_evt = sm.SimulationEvent(
            time=self._current_time + TRAINING_CONFIG.refresh_period,
            event_type=m.EventType.REFRESH_ACTION_BUDGET,
            payload={},
        )
        self._comming_budget_refresh = refresh_evt
        self._events.add(refresh_evt)

        # 1. Calculate and deduct cost
        cost = self._predict_action_cost(action)
        self._environment_state.current_budget -= cost
        self._environment_state.last_action_downtime = cost

        # 2. Perform action
        if action.action == m.ActionType.TRANSFER:
            self._handle_resource_manager_trigger_vram_transfer(action)
        elif action.action in (m.ActionType.SPLIT, m.ActionType.MERGE):
            self._handle_resource_manager_trigger_mig_decision(action)
        self._step_draining_or_active_engines()

    def _handle_shutdown_complete_reallocate(
        self, payload: sm.ShutdownReallocatePayload
    ):
        engine_id = payload["engine_id"]
        receiver_id = payload["receiver_id"]
        engine = self._engines[engine_id]

        new_model = utils.SIM_CONFIG.get_model(
            receiver_id, engine.mig_profile, gpu_id=engine.gpu
        )

        receiver = self._agents[receiver_id]

        engine.update_model(
            new_owner=receiver,
            model_name=new_model,
            max_batched_tokens=utils.SIM_CONFIG.max_batched_tokens[new_model],
            prefill_params=utils.SIM_CONFIG.get_prefill_params(
                receiver_id, engine.mig_profile, gpu_id=engine.gpu
            ),
            tpot_params=utils.SIM_CONFIG.get_tpot_params(
                receiver_id, engine.mig_profile, gpu_id=engine.gpu
            ),
            restart_time=utils.SIM_CONFIG.get_restart_time(
                receiver_id, engine.mig_profile, gpu_id=engine.gpu
            ),
        )
        self._sync_ownership()

        boot_payload: sm.BootPayload = {
            "engine_id": engine_id,
            "purpose": sm.OperationPurpose.REALLOCATE,
            "sibling_engine_ids": None,
        }
        self._events.add(engine.trigger_boot(boot_payload))

    def _handle_shutdown_complete_merge(self, payload: sm.ShutdownMergePayload):
        engine_id = payload["engine_id"]

        # Mark this engine as drained; both sibling events share the same
        # drained_ids list object created in trigger_mig.
        payload["drained_ids"].append(engine_id)

        if len(payload["drained_ids"]) == len(payload["merge_engine_ids"]):
            gpu = payload["gpu"]

            # Get the starting index of the merge operation
            merge_indices = [
                self._engines[eid].mig_index
                for eid in payload["merge_engine_ids"]
                if eid in self._engines
            ]
            action_idx = min(merge_indices) if merge_indices else -1

            # Reset only affected indices to avoid conflicts during sync
            self._invalidate_gpu_indices(gpu, action_idx, payload["target_mig_indices"])

            giver_agent = self._agents[payload["agent_id"]]
            for eid in payload["merge_engine_ids"]:
                e = self._engines.pop(eid, None)
                if e is not None and e in giver_agent.engines:
                    giver_agent.engines.remove(e)

            receiver_id = payload.get("receiver_id")
            target_agent = self._agents[receiver_id] if receiver_id else giver_agent

            target_state_id = payload["target_state_id"]

            target_mig_idx = payload["target_mig_indices"][0]
            new_profile = self._get_hardware_profile(
                payload["gpu"], STATE_DEFINITIONS[target_state_id][target_mig_idx]
            )
            new_eid = utils.generate_engine_id(payload["gpu"], new_profile.string)

            new_eng = LLMEngineImpl.create(
                payload["gpu"],
                new_eid,
                target_agent,
                new_profile,
                self._current_time,
                target_mig_idx,
            )

            boot_plain: sm.BootPayload = {
                "engine_id": new_eid,
                "purpose": sm.OperationPurpose.PLAIN,
                "sibling_engine_ids": None,
            }
            self._events.add(new_eng.trigger_boot(boot_plain))
            self._engines[new_eid] = new_eng
            target_agent.add_engine(new_eng)
            self._sync_ownership()
            self._logger.log_mig_merge_complete(self._current_time, new_eid)

    def _handle_shutdown_complete_split(self, payload: sm.ShutdownSplitPayload):
        engine_id = payload["engine_id"]
        agent = self._agents[payload["agent_id"]]

        # Get index before popping
        e_to_split = self._engines.get(engine_id)
        action_idx = e_to_split.mig_index if e_to_split else -1

        e = self._engines.pop(engine_id, None)
        if e is not None and e in agent.engines:
            agent.engines.remove(e)
        receiver_id = payload["receiver_id"]
        received_profile = payload["received_profile"]
        gpu = payload["gpu"]

        # Reset only affected indices to avoid conflicts during sync
        self._invalidate_gpu_indices(gpu, action_idx, payload["target_mig_indices"])

        is_vram_transfer = receiver_id is not None
        if is_vram_transfer:
            assert received_profile is not None

        target_state_id = payload["target_state_id"]

        target_mig_indices = payload["target_mig_indices"]
        new_profiles = [
            self._get_hardware_profile(gpu, STATE_DEFINITIONS[target_state_id][idx])
            for idx in target_mig_indices
        ]

        new_owners: List[m.Agent] = []
        transfer_received = False
        for p in new_profiles:
            new_owner = agent
            if (
                is_vram_transfer
                and p.profile_type == received_profile
                and not transfer_received
            ):
                new_owner = self._agents[receiver_id]
                transfer_received = True
            new_owners.append(new_owner)

        new_eids = [utils.generate_engine_id(gpu, p.string) for p in new_profiles]

        for new_eid, new_owner, p, target_mig_idx in zip(
            new_eids, new_owners, new_profiles, target_mig_indices
        ):
            new_eng = LLMEngineImpl.create(
                gpu, new_eid, new_owner, p, self._current_time, target_mig_idx
            )
            new_owner.add_engine(new_eng)
            self._engines[new_eid] = new_eng

            boot_split: sm.BootPayload = {
                "engine_id": new_eid,
                "purpose": sm.OperationPurpose.SPLIT,
                "sibling_engine_ids": new_eids,
            }
            self._events.add(new_eng.trigger_boot(boot_split))

        self._sync_ownership()
        self._logger.log_mig_split_complete(self._current_time, engine_id)

    def _handle_shutdown_complete(
        self,
        payload: sm.ShutdownPayload,
    ):
        purpose = payload["purpose"]
        match purpose:
            case sm.OperationPurpose.REALLOCATE:
                self._handle_shutdown_complete_reallocate(
                    cast(sm.ShutdownReallocatePayload, payload)
                )
            case sm.OperationPurpose.MERGE:
                self._handle_shutdown_complete_merge(
                    cast(sm.ShutdownMergePayload, payload)
                )
            case sm.OperationPurpose.SPLIT:
                self._handle_shutdown_complete_split(
                    cast(sm.ShutdownSplitPayload, payload)
                )
            case _:
                raise ValueError(f"Unexpected shutdown purpose {purpose}")

    def _handle_boot_complete(self, payload: sm.BootPayload) -> None:
        engine_id = payload["engine_id"]

        engine = self._engines[engine_id]
        engine.activate(self._current_time)
        agent = engine.owner

        self._logger.log_engine_boot_complete(self._current_time, engine_id)

        # Defer waiting-queue processing if split siblings are still booting
        if payload["purpose"] == sm.OperationPurpose.SPLIT:
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

    def _handle_request_arrival(self, payload: sm.RequestArrivalPayload):
        req = payload["request"]
        agent_id = req.agent_id

        self._environment_state.register_arrival(req)
        self._environment_state.decrement_pending_requests(agent_id)

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

    def _handle_rag_search_complete(self, payload: sm.RequestArrivalPayload):
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

    def _handle_engine_step_complete(self, payload: sm.EngineStepPayload):
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
                        cast(sm.ShutdownPayload, current_event.payload)
                    )
                case m.EventType.ENGINE_BOOT_COMPLETE:
                    self._handle_boot_complete(
                        cast(sm.BootPayload, current_event.payload)
                    )
                case m.EventType.REQUEST_ARRIVAL:
                    self._handle_request_arrival(
                        cast(sm.RequestArrivalPayload, current_event.payload)
                    )
                case m.EventType.RAG_SEARCH_COMPLETE:
                    self._handle_rag_search_complete(
                        cast(sm.RequestArrivalPayload, current_event.payload)
                    )
                case m.EventType.ENGINE_STEP_COMPLETE:
                    self._handle_engine_step_complete(
                        cast(sm.EngineStepPayload, current_event.payload)
                    )
                case m.EventType.REFRESH_ACTION_BUDGET:
                    self._comming_budget_refresh = None
                    self._environment_state.refresh_budget()
                case m.EventType.RESOURCE_MANAGER_TRIGGER:
                    raise ValueError(f"Unexpected event {current_event.event_type}")

        return False  # Finished simulation

    def get_state(self) -> m.EnvironmentStateData:
        return self._environment_state.get_state(
            self._current_time, self._agents, self.gpu_current_state
        )

    def reset(
        self,
        initial_state_mode: Literal["random", "no_mig", "split_extreme"] = "random",
    ) -> None:
        """Resets the simulator to its initial hardware and agent state.

        Args:
            initial_state_mode: How to initialise MIG slices for each GPU.
                - "random"        : Pick a random valid combination (default).
                - "no_mig"        : Use a single 7G instance per GPU (STATIC_NO_MIG).
                - "split_extreme" : Use the most-split valid combination per GPU
                                    (STATIC_SPLIT_EXTREME).
        """
        self._current_time = 0.0
        self._events.clear()
        self._agents.clear()
        self._engines.clear()
        utils.USED_EIDS.clear()

        # Step 1: Generate the initial hardware state
        match initial_state_mode:
            case "no_mig":
                utils.SIM_CONFIG.generate_no_mig_initial_state()
            case "split_extreme":
                utils.SIM_CONFIG.generate_split_extreme_initial_state()
            case "random":
                utils.SIM_CONFIG.generate_initial_state()

        for aid in m.AgentId:
            self._agents[aid] = AgentImpl(aid)

        # Step 2: Initialize engines for all GPUs
        for eng_conf in utils.SIM_CONFIG.initial_state:
            gpu = int(eng_conf["gpu"])
            # Use GPU-specific profile class to parse the string
            mig = GPU_MIG_PROFILE[gpu].from_string(eng_conf["mig"])
            agent_name = eng_conf["agent"]
            agent = self._agents[m.AgentId(agent_name)]
            eid = utils.generate_engine_id(gpu, mig.string)

            is_permanent = eng_conf.get("is-permanent", False)

            # We need to assign indices. During reset, we don't know the state ID yet.
            # But we can identify the state after all engines are created.
            # For now, create with -1 index and fix later.
            eng = LLMEngineImpl.create(
                gpu=gpu,
                engine_id=eid,
                owner=agent,
                mig_profile=mig,
                current_time=0.0,
                mig_index=-1,  # Fixed later
                is_permanent=is_permanent,
            )
            if not eng_conf.get("is-unused", False):
                agent.add_engine(eng)
            self._engines[eid] = eng

        # Step 3: Identify state and fix mig_index
        self._sync_ownership()

        self._environment_state.reset_for_next_interval(0.0, self._agents)
        self._environment_state.refresh_budget()
        self._comming_budget_refresh = None
        self._environment_state.reconfig_flag = False
        for aid in self._agents.keys():
            self._environment_state.set_pending_request_count(aid, 0)
        self._environment_state.reset_last_actions()

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

    def get_action_mask(self, ignore_cooldowns: bool = False) -> List[bool]:
        mask: List[bool] = [False] * len(m.ResourceManagerAction)
        if self._environment_state.reconfig_flag:
            mask[
                list(m.ResourceManagerAction).index(m.ResourceManagerAction.NO_ACTION)
            ] = True
            return mask

        cooldown_steps = TRAINING_CONFIG.action_cooldown if not ignore_cooldowns else 0
        # Transfer cooldown
        transfer_blocked = any(
            self._environment_state.get_steps_since(aid, "give") < cooldown_steps
            for aid in self._agents.keys()
        )

        current_states = self.gpu_current_state
        for act_id, action in enumerate(m.ResourceManagerAction):
            if action == m.ResourceManagerAction.NO_ACTION:
                mask[act_id] = True
                continue

            val = action.value
            act_gpu_id, target_sid, trans_mig = (
                val.gpu_id,
                val.target_state_id,
                val.transfer_mig,
            )
            current_sid = current_states[act_gpu_id]

            # 0. Hardware Support Check
            if target_sid is not None:
                if (
                    STATE_DEFINITIONS[target_sid]
                    not in GPU_VALID_COMBINATIONS[act_gpu_id]
                ):
                    mask[act_id] = False
                    continue
            if trans_mig is not None:
                supported_hw = {p.profile_type for p in GPU_MIG_PROFILE[act_gpu_id]}
                unsupported = GPU_MIG_PROFILE[act_gpu_id].unsupported_profiles()
                if trans_mig not in supported_hw or trans_mig in unsupported:
                    mask[act_id] = False
                    continue

            # 1. State Transition Check
            if target_sid is not None:
                # Check if transition exists in matrix
                trans_action = TRANSITION_MATRIX.get((current_sid, target_sid))
                if not trans_action:
                    mask[act_id] = False
                    continue

                # RECONFIGURATION CONSTRAINT: All source engines must have the same owner
                src_indices = trans_action.mig_src
                owners = {
                    self._gpu_engines[act_gpu_id][idx].owner.agent_id
                    for idx in src_indices
                }
                if len(owners) > 1:
                    mask[act_id] = False
                    continue

            # 2. Transfer Check
            if trans_mig is not None:
                if transfer_blocked:
                    mask[act_id] = False
                    continue
                # If target_sid is None, it's a pure transfer. Check if gpu has the MIG.
                if target_sid is None:
                    has_mig = any(
                        eng is not None
                        and eng.mig_profile.profile_type == trans_mig
                        and not eng.is_permanent
                        for eng in self._gpu_engines[act_gpu_id]
                    )
                    mask[act_id] = has_mig
                else:
                    # It's a state + transition. The transfer MIG must be one of the results of the split/merge.
                    target_profiles = STATE_DEFINITIONS[target_sid]
                    resulting_profiles = [
                        target_profiles[idx]
                        for idx in TRANSITION_MATRIX[
                            (current_sid, target_sid)
                        ].mig_target
                    ]
                    if trans_mig not in resulting_profiles:
                        mask[act_id] = False
                        continue
                    mask[act_id] = True
            else:
                # Pure state transition
                mask[act_id] = True

            # 3. Budget Check
            if mask[act_id]:
                # We need a proper Action object to predict cost
                pred_action = self.map_to_action(action)
                if pred_action:
                    cost = self._predict_action_cost(pred_action)
                    if cost > self._environment_state.current_budget:
                        mask[act_id] = False

        return mask

    def map_to_action(self, res_action: m.ResourceManagerAction) -> Optional[m.Action]:
        if res_action == m.ResourceManagerAction.NO_ACTION:
            return None

        gpu_id = res_action.value.gpu_id
        current_sid = self.gpu_current_state[gpu_id]

        return map_res_action_to_action(
            res_action=res_action,
            current_sid=current_sid,
            find_best_index_fn=self._find_best_engine_index,
            get_owner_fn=self._get_engine_owner,
        )

    def _get_engine_owner(self, gpu_id: int, engine_idx: int) -> m.AgentId:
        return self._gpu_engines[gpu_id][engine_idx].owner.agent_id

    def _identify_gpu_state(self, gpu_id: int) -> int:
        engs = [e for e in self._engines.values() if e.gpu == gpu_id]
        profiles = sorted(
            [e.mig_profile.profile_type for e in engs],
            key=lambda x: x.value,
            reverse=True,
        )
        for sid, defs in STATE_DEFINITIONS.items():
            if sorted(list(defs), key=lambda x: x.value, reverse=True) == profiles:
                return sid
        raise AssertionError(
            f"Could not identify GPU state for engines on GPU {gpu_id}: {profiles}"
        )

    def _get_hardware_profile(
        self, gpu_id: int, logical_profile: m.MIGProfile
    ) -> m.MIGProfileBase:
        for hp in GPU_MIG_PROFILE[gpu_id]:
            if hp.profile_type == logical_profile:
                return hp
        raise ValueError(f"No hardware profile for {logical_profile} on GPU {gpu_id}")

    def _peak_next_stopping_evt(self, agent_id: m.AgentId) -> float | None:
        for evt in self._events:
            if evt.event_type == m.EventType.RESOURCE_MANAGER_TRIGGER:
                return evt.time
            if evt.event_type == m.EventType.REQUEST_ARRIVAL:
                payload = cast(sm.RequestArrivalPayload, evt.payload)
                if (
                    payload["request"].agent_id == agent_id
                    and agent_id != m.AgentId.RAG
                ):
                    return evt.time
            if evt.event_type == m.EventType.RAG_SEARCH_COMPLETE:
                payload = cast(sm.RequestArrivalPayload, evt.payload)
                if (
                    payload["request"].agent_id == agent_id
                    and agent_id == m.AgentId.RAG
                ):
                    return evt.time
        return None
