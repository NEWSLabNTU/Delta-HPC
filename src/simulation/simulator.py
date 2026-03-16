import heapq
from typing import List, Dict, Tuple, Set
from models import Request, EventType, AgentId
from engine import LLMEngine, SimulationEvent
from agent import Agent
import global_vars as g
import random
from logger import SimulationLogger


class ResourceManager:
    def __init__(self, simulator: "Simulator"):
        self.simulator = simulator
        self.trigger_interval = 180000.0
        self.engine_owners: Dict[str, AgentId] = {}
        self.engine_targets: Dict[str, AgentId] = {}
        self.agent_completed: Dict[AgentId, List[Request]] = {
            AgentId.CODING: [],
            AgentId.RAG: [],
        }
        # Track which requests have already been attributed
        self._attributed: Set[str] = set()

    def _attribute_completed(self, engine_id: str):
        """Attribute all newly completed requests from an engine to its current owner."""
        engine = self.simulator.engines[engine_id]
        owner = self.engine_owners[engine_id]
        for req in engine.completed_requests:
            if req.id not in self._attributed:
                self._attributed.add(req.id)
                self.agent_completed[owner].append(req)

    def finalize_accounting(self):
        """Attribute all remaining completed requests at simulation end."""
        for engine_id in self.simulator.engines:
            self._attribute_completed(engine_id)

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
            current_time, giver.agent_id, receiver.agent_id, engine_to_shift.mig_profile
        )

        giver.engines.remove(engine_to_shift)
        self.engine_targets[engine_to_shift.engine_id] = receiver.agent_id

        evt = engine_to_shift.trigger_reallocation(current_time)
        if evt:
            heapq.heappush(self.simulator.events, evt)

    def finish_reallocation(self, engine_id: str, current_time: float):
        receiver_id = self.engine_targets.pop(engine_id)
        self._attribute_completed(engine_id)

        sim = self.simulator
        engine = sim.engines[engine_id]

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

        sim.agents[receiver_id].add_engine(engine)
        self.engine_owners[engine_id] = receiver_id


class Simulator:
    def __init__(
        self,
        agents: Dict[AgentId, Agent],
        engines: Dict[str, LLMEngine],
        no_log: bool = False,
    ):
        self.agents = agents
        self.engines = engines
        self.events: List[SimulationEvent] = []
        self.current_time: float = 0.0
        self.resource_manager = ResourceManager(self)
        self.logger = SimulationLogger(enabled=not no_log)

        for aid, ag in agents.items():
            for eg in ag.engines:
                self.resource_manager.engine_owners[eg.engine_id] = aid

        heapq.heappush(
            self.events,
            SimulationEvent(
                time=self.resource_manager.trigger_interval,
                event_type=EventType.REALLOCATION_TRIGGER,
                payload={},
            ),
        )

    def add_arrival_events(self, requests: List[Request]):
        for req in requests:
            event = SimulationEvent(
                time=req.arrival_time,
                event_type=EventType.REQUEST_ARRIVAL,
                payload={
                    "request": req,
                    "target_agent": req.agent_id,
                },
            )
            heapq.heappush(self.events, event)

    def has_active_work(self) -> bool:
        if any(e.event_type != EventType.REALLOCATION_TRIGGER for e in self.events):
            return True
        if any(
            len(e.waiting_queue) > 0 or len(e.running_queue) > 0
            for e in self.engines.values()
        ):
            return True
        return False

    def run(self):
        """Main event loop."""

        while self.has_active_work():

            # Step any idle engines that have pending requests
            if not self.events:
                for engine in self.engines.values():
                    if not engine.is_busy and (
                        engine.waiting_queue or engine.running_queue
                    ):
                        next_arrival = self._peek_next_arrival_time()
                        evt = engine.step(
                            self.current_time, next_arrival_time=next_arrival
                        )
                        if evt:
                            heapq.heappush(self.events, evt)

                # If no more non-reallocation events or engine work, finish
                if not self.has_active_work():
                    break

            # Pop the earliest event
            current_event = heapq.heappop(self.events)
            self.current_time = current_event.time

            match current_event.event_type:
                case EventType.REALLOCATION_TRIGGER:
                    self.resource_manager.trigger(self.current_time)
                    heapq.heappush(
                        self.events,
                        SimulationEvent(
                            time=self.current_time
                            + self.resource_manager.trigger_interval,
                            event_type=EventType.REALLOCATION_TRIGGER,
                            payload={},
                        ),
                    )

                case EventType.ENGINE_RESTART_COMPLETE:
                    assert "engine_id" in current_event.payload

                    engine_id = current_event.payload["engine_id"]
                    engine = self.engines[engine_id]

                    giver_id = self.resource_manager.engine_owners[engine_id]
                    receiver_id = self.resource_manager.engine_targets[engine_id]

                    self.logger.log_engine_restart_complete(
                        self.current_time, engine_id, giver_id, receiver_id
                    )

                    self.resource_manager.finish_reallocation(
                        engine_id, self.current_time
                    )
                    engine.finish_restart(self.current_time)

                case EventType.REQUEST_ARRIVAL:
                    assert "request" in current_event.payload
                    assert "target_agent" in current_event.payload

                    payload = current_event.payload
                    req: Request = payload["request"]
                    agent_id = payload["target_agent"]

                    agent = self.agents[agent_id]
                    assigned_engine = agent.dispatch(req, self.current_time)
                    self.logger.log_request_arrival(
                        self.current_time,
                        req.id,
                        agent_id,
                        assigned_engine,
                    )

                    if (
                        assigned_engine and not assigned_engine.is_busy
                    ):  # Engine was idle, kickstart it
                        next_arrival = self._peek_next_arrival_time(agent_id)
                        evt = assigned_engine.step(
                            self.current_time, next_arrival_time=next_arrival
                        )
                        if evt:
                            heapq.heappush(self.events, evt)

                case EventType.ENGINE_STEP_COMPLETE:
                    assert "engine_id" in current_event.payload

                    engine_id = current_event.payload["engine_id"]
                    engine = self.engines[engine_id]
                    owner_id = self.resource_manager.engine_owners[engine_id]
                    self.logger.log_engine_step(
                        self.current_time, self.agents, engine, owner_id
                    )

                    # Immediate re-schedule
                    next_arrival = self._peek_next_arrival_time(
                        self.resource_manager.engine_owners[engine_id]
                    )
                    evt = engine.step(self.current_time, next_arrival_time=next_arrival)
                    if evt:
                        heapq.heappush(self.events, evt)

        self.resource_manager.finalize_accounting()
        self.logger.flush()

    def _peek_next_arrival_time(self, agent_id: AgentId) -> float | None:
        """Find the time of the next arrival event for fast-forwarding."""
        for evt in self.events:
            if (
                evt.time > self.current_time
                and evt.event_type == EventType.REQUEST_ARRIVAL
                and evt.payload["target_agent"] == agent_id
            ):
                return evt.time
        return None
