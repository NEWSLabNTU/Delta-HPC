from __future__ import annotations
from typing import List, Optional

import src.simulation.models as m
import src.simulation.utils as utils


class AgentImpl(m.Agent):
    def __init__(self, agent_id: m.AgentId):
        self._agent_id = agent_id
        self._engines: List[m.LLMEngine] = []
        self._completed_requests: List[m.Request] = []

    @property
    def agent_id(self) -> m.AgentId:
        return self._agent_id

    @property
    def engines(self) -> List[m.LLMEngine]:
        return self._engines

    @property
    def completed_requests(self) -> List[m.Request]:
        return self._completed_requests

    def add_engine(self, engine: m.LLMEngine):
        self._engines.append(engine)

    def _pick_laziest_engine(self, engines: List[m.LLMEngine]) -> m.LLMEngine:
        # first compare waiting queue, than compare running queue
        return min(engines, key=lambda e: (len(e.waiting_queue), len(e.running_queue)))

    def dispatch(
        self, request: m.Request, current_time: float
    ) -> Optional[m.LLMEngine]:
        active_engines = [e for e in self.engines if e.status == m.EngineStatus.ACTIVE]

        # At least permanent engines should be active
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

        # Resolve request's MIG instance
        request.mig = best_engine.mig_profile

        best_engine.add_request(request, current_time)
        return best_engine
