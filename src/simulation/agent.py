from __future__ import annotations
from collections import deque
from typing import List, Optional

import src.simulation.models as m
import src.simulation.utils as utils


class AgentImpl(m.Agent):
    def __init__(self, agent_id: m.AgentId):
        self._agent_id = agent_id
        self._engines: List[m.LLMEngine] = []
        self._completed_requests: deque[m.Request] = deque(maxlen=500)

    @property
    def agent_id(self) -> m.AgentId:
        return self._agent_id

    @property
    def engines(self) -> List[m.LLMEngine]:
        return self._engines

    @property
    def completed_requests(self) -> deque[m.Request]:
        return self._completed_requests

    def add_engine(self, engine: m.LLMEngine):
        self._engines.append(engine)

    def dispatch(
        self, request: m.Request, current_time: float
    ) -> Optional[m.LLMEngine]:
        active_engines = [e for e in self.engines if e.status == m.EngineStatus.ACTIVE]

        # At least permanent engines should be active
        assert len(active_engines) > 0, f"No active engines for agent {self.agent_id}"

        # Selection Logic:
        # 1. From the largest MIG, pick the first one having 0 waiting requests.
        # 2. If no such engine, pick the one with the shortest waiting queue (favoring larger MIGs).
        def selection_key(e: m.LLMEngine):
            has_waiting = len(e.waiting_queue) > 0
            # Tier 1: has_waiting (False/0 first)
            # Tier 2: len(waiting_queue) (Smallest first)
            # Tier 3: size (Largest first -> smallest negative size)
            return (has_waiting, len(e.waiting_queue), -e.mig_profile.size)

        best_engine = min(active_engines, key=selection_key)

        # Resolve completion_tokens based on the engine's current model
        model_req_map = utils.TOKENS_MAP[self.agent_id][best_engine.model_name]
        lookup_id = request.original_id if request.original_id else request.id
        _, completion_tokens = model_req_map[lookup_id]
        request.completion_tokens = completion_tokens

        # Resolve request's serving engine
        request.serving_engine = best_engine

        best_engine.add_request(request, current_time)
        return best_engine
