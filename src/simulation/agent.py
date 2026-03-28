from __future__ import annotations
from typing import List, Optional

from src.simulation.models import *
import src.simulation.utils as utils


class AgentImpl(Agent):
    def __init__(self, agent_id: AgentId):
        self._agent_id = agent_id
        self._engines: List[LLMEngine] = []
        self._completed_requests: List[Request] = []

    @property
    def agent_id(self) -> AgentId:
        return self._agent_id

    @property
    def engines(self) -> List[LLMEngine]:
        return self._engines

    @property
    def completed_requests(self) -> List[Request]:
        return self._completed_requests

    def add_engine(self, engine: LLMEngine):
        self._engines.append(engine)

    def _pick_laziest_engine(self, engines: List[LLMEngine]) -> LLMEngine:
        # first compare waiting queue, than compare running queue
        return min(engines, key=lambda e: (len(e.waiting_queue), len(e.running_queue)))

    def dispatch(self, request: Request, current_time: float) -> Optional[LLMEngine]:
        """
        Dispatches an incoming request to the best engine based on simple work-balance.
        Finds engines matching the requested model size, and picks the one with the smallest queue length.
        Sets completion_tokens based on the chosen engine's model before queuing the request.
        If no active engines exist, queues the request in the agent's waiting queue.
        """
        active_engines = [e for e in self.engines if e.status == EngineStatus.ACTIVE]
        assert len(active_engines) > 0, f"No active engines for agent {self.agent_id}"

        regular_active = [e for e in active_engines if not e.is_permanent]
        permanent_active = [e for e in active_engines if e.is_permanent]

        if not regular_active and permanent_active:
            best_engine = self._pick_laziest_engine(permanent_active)
        elif regular_active and not permanent_active:
            best_engine = self._pick_laziest_engine(regular_active)
        else:
            # Both got active engines
            if all(len(e.waiting_queue) > 0 for e in active_engines):
                # if every engine is buzy, dispatch normally
                best_engine = self._pick_laziest_engine(active_engines)
            elif any(len(e.waiting_queue) == 0 for e in regular_active):
                # regular engine gets assigend first
                best_engine = self._pick_laziest_engine(regular_active)
            else:
                best_engine = self._pick_laziest_engine(permanent_active)

        # Resolve completion_tokens based on the engine's current model
        model_req_map = utils.TOKENS_MAP[self.agent_id][best_engine.model_name]
        lookup_id = request.original_id if request.original_id else request.id
        _, completion_tokens = model_req_map[lookup_id]
        request.completion_tokens = completion_tokens

        best_engine.add_request(request, current_time)
        return best_engine
