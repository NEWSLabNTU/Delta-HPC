import random
from typing import Dict, List, Optional, TypedDict

import src.simulation.models as m
from src.simulation.request import RequestImpl
import src.simulation.utils as utils
from src.bench.config import BENCH_CONFIG
from src.bench.models import Workload
from src.training.config import TRAINING_CONFIG


class PhaseHistoryType(TypedDict):
    pattern: str
    avg_rate: float
    duration: float


class BenchRequestLoader:
    def __init__(self, workload: Workload, seed: Optional[int] = None):
        self.workload = workload
        self._base_seed = seed if seed is not None else BENCH_CONFIG.seed
        self.phase_history: Dict[
            m.AgentId, List[PhaseHistoryType]
        ] = {}  # {agent_id: [ {pattern, avg_rate, duration} ]}
        if workload != Workload.HYBRID:
            self.min_rate, self.max_rate = BENCH_CONFIG.get_rate_range(workload)

    def generate_requests(
        self, agent_id: m.AgentId, start_time: float = 0.0, turn: int = 0
    ) -> List[m.Request]:
        requests: List[m.Request] = []

        # Pull request datasets from tokens map
        first_model = next(iter(utils.TOKENS_MAP[agent_id]))
        req_map = utils.TOKENS_MAP[agent_id][first_model]
        all_items = list(req_map.items())

        if agent_id == m.AgentId.RAG:
            items = all_items * 2
        else:
            items = all_items

        # Seed deterministically: base_seed combined with agent index for uniqueness
        agent_idx = list(m.AgentId).index(agent_id)
        random.seed(self._base_seed ^ (agent_idx * 0x9E3779B9))
        current_time = start_time

        max_time = (
            start_time
            + BENCH_CONFIG.benchmark_length * TRAINING_CONFIG.action_interval
            + 360  # small buffer
        )

        if self.workload == Workload.HYBRID:
            while current_time < max_time:
                # Pick a random workload phase
                phase_workload = random.choice(
                    [Workload.IDLE, Workload.BALANCED, Workload.BUSY]
                )
                min_rate, max_rate = BENCH_CONFIG.get_rate_range(phase_workload)
                min_dur, max_dur = BENCH_CONFIG.get_duration_range(phase_workload)

                duration = random.uniform(min_dur, max_dur)
                phase_start_time = current_time
                phase_end = current_time + duration

                rates_in_phase: List[float] = []
                while current_time < phase_end and current_time < max_time:
                    rate = random.uniform(min_rate, max_rate)
                    rates_in_phase.append(rate)
                    current_time += random.expovariate(rate)
                    if current_time >= max_time:
                        break

                    idx = len(requests)
                    rid, (prompt_tokens, _) = items[idx % len(items)]
                    req = RequestImpl(
                        id=f"{rid}_{agent_id.value}_t{turn}_i{idx}",
                        agent_id=agent_id,
                        prompt_tokens=prompt_tokens,
                        original_id=rid,
                    )
                    req.arrival_time = current_time
                    requests.append(req)

                if agent_id not in self.phase_history:
                    self.phase_history[agent_id] = []

                self.phase_history[agent_id].append(
                    {
                        "pattern": phase_workload.value,
                        "avg_rate": sum(rates_in_phase) / len(rates_in_phase)
                        if rates_in_phase
                        else 0,
                        "duration": current_time - phase_start_time,
                    }
                )
        else:
            # Discrete workload
            phase_start_time = current_time
            rates_in_phase = []
            while current_time < max_time:
                rate = random.uniform(self.min_rate, self.max_rate)
                rates_in_phase.append(rate)
                current_time += random.expovariate(rate)
                if current_time >= max_time:
                    break

                idx = len(requests)
                rid, (prompt_tokens, _) = items[idx % len(items)]

                req = RequestImpl(
                    id=f"{rid}_{agent_id.value}_t{turn}_i{idx}",
                    agent_id=agent_id,
                    prompt_tokens=prompt_tokens,
                    original_id=rid,
                )
                req.arrival_time = current_time
                requests.append(req)

            if agent_id not in self.phase_history:
                self.phase_history[agent_id] = []
            self.phase_history[agent_id].append(
                {
                    "pattern": self.workload.value,
                    "avg_rate": sum(rates_in_phase) / len(rates_in_phase)
                    if rates_in_phase
                    else 0,
                    "duration": current_time - phase_start_time,
                }
            )

        return requests
