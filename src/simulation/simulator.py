import heapq
from typing import List, Dict
from src.simulation.models import Request, EventType, AgentId
from src.simulation.engine import LLMEngine, SimulationEvent
from src.simulation.agent import Agent


class Simulator:
    def __init__(
        self,
        agents: Dict[AgentId, Agent],
        engines: Dict[str, LLMEngine],
        tokens_map: dict = None,
    ):
        self.agents = agents
        self.engines = engines
        self.events: List[SimulationEvent] = []
        self.current_time: float = 0.0
        self.tokens_map = tokens_map or {}

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

    def run(self):
        """Main event loop."""

        while self.events or any(
            len(e.waiting_queue) > 0 or len(e.running_queue) > 0
            for e in self.engines.values()
        ):

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

                # If still no events, exit completely
                if not self.events:
                    break

            # Pop the earliest event
            current_event = heapq.heappop(self.events)
            self.current_time = current_event.time

            match current_event.event_type:
                case EventType.REQUEST_ARRIVAL:
                    payload = current_event.payload
                    req: Request = payload["request"]
                    agent_id = payload["target_agent"]

                    agent = self.agents[agent_id]
                    assigned_engine = agent.dispatch(req, self.current_time)

                    if assigned_engine and self.tokens_map:
                        p, c = self.tokens_map[assigned_engine.model_name][req.id]
                        req.prompt_tokens = p
                        req.completion_tokens = c

                    if (
                        assigned_engine and not assigned_engine.is_busy
                    ):  # Engine was idle, kickstart it
                        next_arrival = self._peek_next_arrival_time()
                        evt = assigned_engine.step(
                            self.current_time, next_arrival_time=next_arrival
                        )
                        if evt:
                            heapq.heappush(self.events, evt)

                case EventType.ENGINE_STEP_COMPLETE:
                    engine_id = current_event.payload["engine_id"]
                    engine = self.engines[engine_id]

                    # Immediate re-schedule
                    next_arrival = self._peek_next_arrival_time()
                    evt = engine.step(self.current_time, next_arrival_time=next_arrival)
                    if evt:
                        heapq.heappush(self.events, evt)

    def _peek_next_arrival_time(self) -> float:
        """Find the time of the next arrival event for fast-forwarding."""
        for evt in self.events:
            if evt.event_type == EventType.REQUEST_ARRIVAL:
                return evt.time
        return None
