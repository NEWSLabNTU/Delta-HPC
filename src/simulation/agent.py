from typing import List

import global_vars as g
from engine import LLMEngine
from models import Request, AgentId, EngineStatus


class Agent:
    def __init__(self, agent_id: AgentId):
        self.agent_id = agent_id
        self.engines: List[LLMEngine] = []
        self.completed_requests: List[Request] = []

    def add_engine(self, engine: LLMEngine):
        self.engines.append(engine)

    def dispatch(self, request: Request, current_time: float) -> LLMEngine:
        """
        Dispatches an incoming request to the best engine based on simple work-balance.
        Finds engines matching the requested model size, and picks the one with the smallest queue length.
        Sets completion_tokens based on the chosen engine's model before queuing the request.
        """
        active_engines = [e for e in self.engines if e.status == EngineStatus.ACTIVE]
        if not active_engines:
            raise RuntimeError(
                f"[{self.agent_id}] No active engines to dispatch request {request.id}"
            )

        # Simple work-balance: Pick the active engine with the fewest requests
        best_engine = min(
            active_engines,
            key=lambda e: len(e.running_queue) + len(e.waiting_queue),
        )

        # Resolve completion_tokens based on the engine's current model
        model_req_map = g.TOKENS_MAP[self.agent_id][best_engine.model_name]
        lookup_id = request.original_id if request.original_id else request.id
        _, completion_tokens = model_req_map[lookup_id]
        request.completion_tokens = completion_tokens

        best_engine.add_request(request, current_time)
        return best_engine
