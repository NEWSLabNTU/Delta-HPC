import random
from typing import List, Dict, Tuple

import src.simulation.models as m
from src.simulation.request import RequestImpl
import src.simulation.utils as utils


class RequestLoader:
    def __init__(self, phase: int = 0):
        self.phase = phase
        # State tracking for Phase 1 patterns
        self.current_pattern: int = 3
        self.pattern_end_time: float = 0.0
        self.current_rates: Dict[m.AgentId, float] = self._get_rates()

    def _get_rates(self) -> Dict[m.AgentId, float]:
        rates = {
            "busy": random.uniform(3.5, 4.5),
            "idle": random.uniform(0.5, 1.0),
            "balanced": random.uniform(1.5, 2.5),
        }
        if self.current_pattern == 1:
            return {
                m.AgentId.CODING: rates["busy"],
                m.AgentId.RAG: rates["idle"],
            }
        elif self.current_pattern == 2:
            return {
                m.AgentId.CODING: rates["idle"],
                m.AgentId.RAG: rates["busy"],
            }
        else:
            return {
                m.AgentId.CODING: rates["balanced"],
                m.AgentId.RAG: rates["balanced"],
            }

    def generate_requests(
        self, start_time: float = 0.0, turn: int = 0
    ) -> List[m.Request]:
        """
        Loads arriving Requests.
        Returns a batch of new dynamically generated requests up to a static limit of 50,000 requests total.
        """
        requests: List[m.Request] = []

        # Load raw data maps
        agent_req_items: Dict[m.AgentId, List[Tuple[str, Tuple[int, int]]]] = {}
        for agent_id in m.AgentId:
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
                agent_req_items[agent_id] = rag_items
            else:
                agent_req_items[agent_id] = random.sample(all_items, 25000)

        if self.phase == 0:
            # Standard phase 0 logic (constant spacing)
            for agent_id, items in agent_req_items.items():
                for idx, (rid, (prompt_tokens, _)) in enumerate(items):
                    requests.append(
                        RequestImpl(
                            id=f"{rid}_a{agent_id.value}_t{turn}_i{idx}",
                            agent_id=agent_id,
                            prompt_tokens=prompt_tokens,
                            original_id=rid,
                        )
                    )
            # Shuffle slightly differently per turn
            random.seed(42 + turn)
            random.shuffle(requests)

            for i, req in enumerate(requests):
                req.arrival_time = start_time + i * 0.25
            return requests

        else:
            # Phase 1: dynamic time-based sampling using Poisson arrivals
            # Reset random seed behavior so patterns are truly stochastic
            random.seed()

            agents_next_time = {m.AgentId.CODING: start_time, m.AgentId.RAG: start_time}
            agents_idx = {m.AgentId.CODING: 0, m.AgentId.RAG: 0}

            while len(requests) < 50000:
                # Rotate pattern if time crossed
                min_time = min(agents_next_time.values())
                if min_time >= self.pattern_end_time:
                    pattern_list = [1, 2, 3]
                    pattern_list.remove(self.current_pattern)
                    self.current_pattern = random.choice(pattern_list)
                    self.pattern_end_time = min_time + random.uniform(1200.0, 2400.0)
                    self.current_rates = self._get_rates()

                # Pick the agent with earliest arrival that hasn't run out of samples
                valid_agents = [aid for aid in m.AgentId if agents_idx[aid] < 25000]
                if not valid_agents:
                    break

                agent = min(valid_agents, key=lambda a: agents_next_time[a])
                arr_time = agents_next_time[agent]

                idx = agents_idx[agent]
                rid, (prompt_tokens, _) = agent_req_items[agent][idx]

                req = RequestImpl(
                    id=f"{rid}_a{agent.value}_t{turn}_i{idx}",
                    agent_id=agent,
                    prompt_tokens=prompt_tokens,
                    original_id=rid,
                )
                req.arrival_time = arr_time
                requests.append(req)

                # Move index and update next arrival
                agents_idx[agent] += 1
                rate = self.current_rates[agent]
                agents_next_time[agent] += random.expovariate(rate)

            # Sort physically by arrival_time cleanly
            requests.sort(key=lambda r: r.arrival_time)
            return requests
