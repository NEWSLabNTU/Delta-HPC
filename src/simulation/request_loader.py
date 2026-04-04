import random
from typing import List, Dict

import src.simulation.models as m
from src.training.models import AgentPattern
from src.simulation.request import RequestImpl
import src.simulation.utils as utils
from src.training.config import TRAINING_CONFIG


class RequestLoader:
    def __init__(self, phase: int = 0):
        self.phase = phase
        # State tracking for Phase 1 patterns
        self.agent_patterns = {
            m.AgentId.CODING: random.choice(list(AgentPattern)),
            m.AgentId.RAG: random.choice(list(AgentPattern)),
        }
        self.agent_pattern_end_times = {
            m.AgentId.CODING: 0.0,
            m.AgentId.RAG: 0.0,
        }
        self.current_rates: Dict[m.AgentId, float] = self._get_rates()

    def _get_pattern_duration(self, pattern: AgentPattern) -> float:
        min_v, max_v = TRAINING_CONFIG.pattern_duration(pattern)
        return random.uniform(min_v, max_v)

    def _get_rate_for_pattern(
        self, pattern: AgentPattern, agent_id: m.AgentId
    ) -> float:
        min_v, max_v = TRAINING_CONFIG.pattern_rate(pattern, agent_id)
        return random.uniform(min_v, max_v)

    def _get_rates(self) -> Dict[m.AgentId, float]:
        return {
            aid: self._get_rate_for_pattern(self.agent_patterns[aid], aid)
            for aid in m.AgentId
        }

    def generate_requests(
        self, agent_id: m.AgentId, start_time: float = 0.0, turn: int = 0
    ) -> List[m.Request]:
        """
        Loads arriving Requests.
        Returns a batch of new dynamically generated requests up to a static limit of 25,000 requests.
        """
        requests: List[m.Request] = []

        first_model = next(iter(utils.TOKENS_MAP[agent_id]))
        req_map = utils.TOKENS_MAP[agent_id][first_model]
        all_items = list(req_map.items())

        if agent_id == m.AgentId.RAG:
            # Expand RAG pool
            rag_items = all_items * 2
            needed = 25000 - len(rag_items)
            if needed > 0:
                rag_items.extend(random.sample(all_items, needed))
            elif needed < 0:
                rag_items = rag_items[:25000]
            items = rag_items
        else:
            items = random.sample(all_items, 25000)

        assert self.phase == 1

        # Phase 1: dynamic time-based sampling using Poisson arrivals
        # Reset random seed behavior so patterns are truly stochastic
        random.seed()

        current_time = start_time

        # Initialize the pattern end times properly relative to start_time
        if start_time == 0.0:
            self.agent_pattern_end_times[agent_id] = (
                current_time + self._get_pattern_duration(self.agent_patterns[agent_id])
            )

        while len(requests) < 25000:
            # Rotate pattern if time crossed
            if current_time >= self.agent_pattern_end_times[agent_id]:
                patterns = list(AgentPattern)
                if self.agent_patterns[agent_id] in patterns:
                    patterns.remove(self.agent_patterns[agent_id])
                new_pattern = random.choice(patterns)

                self.agent_patterns[agent_id] = new_pattern
                self.agent_pattern_end_times[agent_id] = (
                    current_time + self._get_pattern_duration(new_pattern)
                )
                self.current_rates[agent_id] = self._get_rate_for_pattern(
                    new_pattern, agent_id
                )

            rate = self.current_rates[agent_id]
            current_time += random.expovariate(rate)

            idx = len(requests)
            rid, (prompt_tokens, _) = items[idx]

            req = RequestImpl(
                id=f"{rid}_{agent_id.value}_t{turn}_i{idx}",
                agent_id=agent_id,
                prompt_tokens=prompt_tokens,
                original_id=rid,
            )
            req.arrival_time = current_time
            requests.append(req)

        return requests
