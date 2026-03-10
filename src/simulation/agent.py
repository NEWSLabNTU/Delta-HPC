from typing import List, Optional
from src.simulation.models import Request, AgentId
from src.simulation.engine import LLMEngine


class Agent:
    def __init__(self, agent_id: AgentId):
        self.agent_id = agent_id
        self.engines: List[LLMEngine] = []

    def add_engine(self, engine: LLMEngine):
        self.engines.append(engine)

    def dispatch(self, request: Request, current_time: float) -> Optional[LLMEngine]:
        """
        Dispatches an incoming request to the best engine based on simple work-balance.
        Finds engines matching the requested model size, and picks the one with the smallest queue length.
        """
        if not self.engines:
            print(f"[{self.agent_id}] Processing Failed: No engines allocated")
            return None

        # Simple work-balance: Pick the engine with the fewest running requests
        best_engine = min(
            self.engines,
            key=lambda e: len(e.running_queue),
        )

        best_engine.add_request(request, current_time)
        return best_engine
