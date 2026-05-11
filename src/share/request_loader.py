import random
from typing import List, Dict, Callable, Tuple, Optional, TypedDict

import datasets

import src.share.models as m
from src.simulation.request import RequestImpl
import src.simulation.utils as utils
from src.bench.models import Workload
from src.training.config import TRAINING_CONFIG


class PhaseHistoryType(TypedDict):
    pattern: str
    avg_rate: float
    duration: float
    start_time: float


class RequestLoader:
    def __init__(
        self,
        num_steps: int,
        get_rate_range: Callable[[str, m.AgentId], Tuple[float, float]],
        get_duration_range: Callable[[str], Tuple[float, float]],
        seed: Optional[int] = None,
        track_history: bool = False,
        load_actual_prompt: bool = False,
    ):
        self.num_steps = num_steps
        self.get_rate_range = get_rate_range
        self.get_duration_range = get_duration_range
        self.seed = seed
        self.track_history = track_history
        self.load_actual_prompt = load_actual_prompt
        self.phase_history: Dict[m.AgentId, List[PhaseHistoryType]] = {}

        if self.load_actual_prompt:
            self._init_dataset_cache()

    def _init_dataset_cache(self):
        self._coding_ds = datasets.load_from_disk("assets/processed_code_feedback")
        self._coding_id_map = {
            str(row["id"]): i
            for i, row in enumerate(self._coding_ds.select_columns(["id"]))
        }

        self._rag_ds = datasets.load_from_disk("assets/rag-dataset-sharegpt")
        self._rag_id_map = {
            str(row["id"]): i
            for i, row in enumerate(self._rag_ds.select_columns(["id"]))
        }

    def _get_actual_prompt(self, agent_id: m.AgentId, rid: str) -> Optional[str]:
        if agent_id == m.AgentId.CODING:
            idx = self._coding_id_map.get(str(rid))
            if idx is not None:
                msgs = self._coding_ds[idx]["messages"]
                return msgs[0]["value"] if msgs else None
        elif agent_id == m.AgentId.RAG:
            idx = self._rag_id_map.get(str(rid))
            if idx is not None:
                msgs = self._rag_ds[idx]["messages"]
                return msgs[0]["value"] if msgs else None
        return None

    def generate_requests(
        self, agent_id: m.AgentId, start_time: float = 0.0, turn: int = 0
    ) -> List[m.Request]:
        """
        Loads arriving Requests dynamically up to the computed maximum time based on num_steps.
        """
        requests: List[m.Request] = []

        first_model = next(iter(utils.TOKENS_MAP[agent_id]))
        req_map = utils.TOKENS_MAP[agent_id][first_model]
        items = list(req_map.items())

        if self.seed is not None:
            agent_idx = list(m.AgentId).index(agent_id)
            random.seed(self.seed ^ (agent_idx * 0x9E3779B9))
        else:
            random.seed()

        current_time = start_time
        max_time = start_time + self.num_steps * TRAINING_CONFIG.action_interval

        patterns = [w.value for w in Workload]

        while current_time < max_time:
            pattern = random.choice(patterns)
            min_rate, max_rate = self.get_rate_range(pattern, agent_id)
            min_dur, max_dur = self.get_duration_range(pattern)

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

                if self.load_actual_prompt:
                    actual_prompt = self._get_actual_prompt(agent_id, str(rid))
                    if not actual_prompt:
                        raise ValueError(
                            f"Prompt not found for ID {rid} in agent {agent_id.value} dataset."
                        )
                    req.prompt = actual_prompt

                req.arrival_time = current_time
                requests.append(req)

            if self.track_history:
                if agent_id not in self.phase_history:
                    self.phase_history[agent_id] = []
                self.phase_history[agent_id].append(
                    {
                        "pattern": pattern,
                        "avg_rate": sum(rates_in_phase) / len(rates_in_phase)
                        if rates_in_phase
                        else 0,
                        "duration": current_time - phase_start_time,
                        "start_time": phase_start_time,
                    }
                )

        return requests
