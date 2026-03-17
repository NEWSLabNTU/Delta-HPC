import heapq
import random
from collections import deque
from typing import List, Dict, Tuple

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
)


class ResourceManager:
    def __init__(self, simulator: "SimulatorI"):
        self.simulator = simulator
        self.trigger_interval = 1800.0
        self.engine_targets: Dict[str, AgentId] = {}

    def trigger(self, current_time: float):
        coding_agent = self.simulator.agents[AgentId.CODING]
        rag_agent = self.simulator.agents[AgentId.RAG]

        candidates_to_give: List[Tuple[Agent, Agent]] = []
        if len(coding_agent.engines) > 1:
            candidates_to_give.append((coding_agent, rag_agent))
        if len(rag_agent.engines) > 1:
            candidates_to_give.append((rag_agent, coding_agent))

        if not candidates_to_give:
            return

        giver, receiver = random.choice(candidates_to_give)
        engine_to_shift = random.choice(giver.engines)
        self.simulator.logger.log_reallocation(
            current_time, giver.agent_id, receiver.agent_id, engine_to_shift.engine_id
        )

        self.engine_targets[engine_to_shift.engine_id] = receiver.agent_id

        evt = engine_to_shift.trigger_reallocation(current_time)
        if evt:
            heapq.heappush(self.simulator.events, evt)

    def finish_reallocation(self, engine_id: str):
        sim = self.simulator
        engine = sim.engines[engine_id]

        receiver_id = self.engine_targets.pop(engine_id)
        new_model = g.SIM_CONFIG.get_model(receiver_id, engine.mig_profile)
        engine.update_model(
            model_name=new_model,
            max_batched_tokens=g.SIM_CONFIG.max_batched_tokens[new_model],
            prefill_params=g.SIM_CONFIG.get_prefill_params(
                receiver_id, engine.mig_profile
            ),
            tpot_params=g.SIM_CONFIG.get_tpot_params(receiver_id, engine.mig_profile),
            restart_time=g.SIM_CONFIG.get_restart_time(receiver_id, engine.mig_profile),
        )

        giver = engine.owner
        receiver = sim.agents[receiver_id]
        giver.engines.remove(engine)
        receiver.add_engine(engine)
        engine.owner = receiver


class Simulator(SimulatorI):
    def __init__(
        self,
        agents: Dict[AgentId, Agent],
        engines: Dict[str, LLMEngine],
        no_log: bool = False,
    ):
        self._agents = agents
        self._engines = engines
        self._events: List[SimulationEvent] = []
        self._current_time: float = 0.0
        self.resource_manager = ResourceManager(self)
        self._logger = SimulationLogger(enabled=not no_log)
        self.total_requests = 0
        self.agent_requests: Dict[AgentId, deque[Request]] = {
            aid: deque() for aid in agents
        }

        heapq.heappush(
            self._events,
            SimulationEvent(
                time=self.resource_manager.trigger_interval,
                event_type=EventType.REALLOCATION_TRIGGER,
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
    def events(self) -> List[SimulationEvent]:
        return self._events

    @property
    def current_time(self) -> float:
        return self._current_time

    @property
    def logger(self) -> SimulationLogger:
        return self._logger

    def add_arrival_events(self, requests: List[Request]):
        self.total_requests = len(requests)
        temp_reqs: Dict[AgentId, List[Request]] = {aid: [] for aid in self._agents}

        for req in requests:
            event = SimulationEvent(
                time=req.arrival_time,
                event_type=EventType.REQUEST_ARRIVAL,
                payload={
                    "request": req,
                    "target_agent": req.agent_id,
                },
            )
            heapq.heappush(self._events, event)
            temp_reqs[req.agent_id].append(req)

        for aid, req_list in temp_reqs.items():
            req_list.sort(key=lambda x: x.arrival_time)
            self.agent_requests[aid] = deque(req_list)

    def has_active_work(self) -> bool:
        if any(e.event_type != EventType.REALLOCATION_TRIGGER for e in self._events):
            return True
        if any(
            len(e.waiting_queue) > 0 or len(e.running_queue) > 0
            for e in self._engines.values()
        ):
            return True
        return False

    def run(self):
        """Main event loop."""
        from tqdm import tqdm

        pbar = tqdm(total=self.total_requests, desc="Simulation Progress")
        last_completed = 0

        while self.has_active_work():

            # Step any idle engines that have pending requests
            if not self._events:
                for engine in self._engines.values():
                    if engine.waiting_queue or engine.running_queue:
                        evt = engine.step(
                            self._current_time,
                            next_arrival_time=None,
                        )
                        if evt:
                            heapq.heappush(self._events, evt)

                # If no more non-reallocation events or engine work, finish
                if not self.has_active_work():
                    break

            # Pop the earliest event
            current_event = heapq.heappop(self._events)
            self._current_time = current_event.time

            match current_event.event_type:
                case EventType.REALLOCATION_TRIGGER:
                    self.resource_manager.trigger(self._current_time)
                    heapq.heappush(
                        self._events,
                        SimulationEvent(
                            time=self._current_time
                            + self.resource_manager.trigger_interval,
                            event_type=EventType.REALLOCATION_TRIGGER,
                            payload={},
                        ),
                    )

                case EventType.ENGINE_RESTART_COMPLETE:
                    assert "engine_id" in current_event.payload

                    engine_id = current_event.payload["engine_id"]
                    engine = self._engines[engine_id]

                    giver_id = engine.owner.agent_id
                    receiver_id = self.resource_manager.engine_targets[engine_id]

                    self._logger.log_engine_restart_complete(
                        self._current_time, engine_id, giver_id, receiver_id
                    )

                    self.resource_manager.finish_reallocation(engine_id)
                    engine.finish_restart(self._current_time)

                case EventType.REQUEST_ARRIVAL:
                    assert "request" in current_event.payload
                    assert "target_agent" in current_event.payload

                    payload = current_event.payload
                    req: Request = payload["request"]
                    agent_id = payload["target_agent"]
                    self.agent_requests[agent_id].popleft()

                    agent = self._agents[agent_id]
                    engine = agent.dispatch(req, self._current_time)
                    self._logger.log_request_arrival(
                        self._current_time,
                        req.id,
                        agent_id,
                        engine,
                    )

                    if len(engine.running_queue) == 0 and len(engine.waiting_queue) > 0:
                        evt = engine.step(
                            self._current_time,
                            next_arrival_time=self._peek_next_arrival_time(agent_id),
                        )
                        if evt:
                            heapq.heappush(self._events, evt)

                case EventType.ENGINE_STEP_COMPLETE:
                    assert "engine_id" in current_event.payload

                    engine_id = current_event.payload["engine_id"]
                    engine = self._engines[engine_id]

                    next_arrival_time = (
                        self._peek_next_arrival_time(engine.owner.agent_id)
                        if engine.status == EngineStatus.ACTIVE
                        else None
                    )
                    self._logger.log_engine_step(
                        self._current_time,
                        self._agents,
                        engine,
                        next_arrival_time,
                    )

                    # Immediate re-schedule
                    evt = engine.step(
                        self._current_time,
                        next_arrival_time=next_arrival_time,
                    )
                    if evt:
                        heapq.heappush(self._events, evt)

            # Update progress bar
            completed_now = sum(
                len(ag.completed_requests) for ag in self._agents.values()
            )
            if completed_now > last_completed:
                pbar.update(completed_now - last_completed)
                last_completed = completed_now

        pbar.close()
        self._logger.flush()

    def _peek_next_arrival_time(self, agent_id: AgentId) -> float | None:
        """Find the time of the next arrival event for fast-forwarding."""
        q = self.agent_requests[agent_id]
        if q:
            return q[0].arrival_time
        return None
