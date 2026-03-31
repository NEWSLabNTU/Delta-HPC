from __future__ import annotations

from typing import List, Optional

import src.simulation.models as m


class RequestImpl(m.Request):
    def __init__(
        self,
        id: str,
        agent_id: m.AgentId,
        prompt_tokens: int = 0,
        arrival_time: float = 0.0,
        original_id: str = "",
    ):
        self._id = id
        self._agent_id = agent_id
        self._prompt_tokens = prompt_tokens
        self._arrival_time = arrival_time
        self._original_id = original_id

        self._serving_engine = None
        self._completion_tokens = 0
        self._decode_time = 0.0
        self._state = m.RequestState.PENDING
        self._prefilled_tokens = 0
        self._generated_tokens = 0
        self._start_time: Optional[float] = None
        self._first_token_time: Optional[float] = None
        self._finish_time: Optional[float] = None

    @property
    def id(self) -> str:
        return self._id

    @property
    def serving_engine(self) -> Optional[m.LLMEngine]:
        return self._serving_engine

    @serving_engine.setter
    def serving_engine(self, e: m.LLMEngine):
        self._serving_engine = e

    @property
    def original_id(self) -> str:
        return self._original_id

    @property
    def agent_id(self) -> m.AgentId:
        return self._agent_id

    @property
    def prompt_tokens(self) -> int:
        return self._prompt_tokens

    @property
    def is_finished(self) -> bool:
        return self._generated_tokens >= self._completion_tokens

    @property
    def remaining_prefill_tokens(self) -> int:
        return self._prompt_tokens - self._prefilled_tokens

    @property
    def prefill_completed(self) -> bool:
        return self._prefilled_tokens >= self._prompt_tokens

    @property
    def completion_tokens(self) -> int:
        return self._completion_tokens

    @completion_tokens.setter
    def completion_tokens(self, value: int):
        self._completion_tokens = value

    @property
    def arrival_time(self) -> float:
        return self._arrival_time

    @arrival_time.setter
    def arrival_time(self, value: float):
        self._arrival_time = value

    @property
    def decode_time(self) -> float:
        return self._decode_time

    @decode_time.setter
    def decode_time(self, value: float):
        self._decode_time = value

    @property
    def state(self) -> m.RequestState:
        return self._state

    @state.setter
    def state(self, value: m.RequestState):
        self._state = value

    @property
    def prefilled_tokens(self) -> int:
        return self._prefilled_tokens

    @prefilled_tokens.setter
    def prefilled_tokens(self, value: int):
        self._prefilled_tokens = value

    @property
    def generated_tokens(self) -> int:
        return self._generated_tokens

    @generated_tokens.setter
    def generated_tokens(self, value: int):
        self._generated_tokens = value

    @property
    def start_time(self) -> Optional[float]:
        return self._start_time

    @start_time.setter
    def start_time(self, value: Optional[float]):
        self._start_time = value

    @property
    def first_token_time(self) -> Optional[float]:
        return self._first_token_time

    @first_token_time.setter
    def first_token_time(self, value: Optional[float]):
        self._first_token_time = value

    @property
    def finish_time(self) -> Optional[float]:
        return self._finish_time

    @finish_time.setter
    def finish_time(self, value: Optional[float]):
        self._finish_time = value


class RunningRequestsImpl(m.RunningRequests):
    def __init__(self):
        self._prefill_requests: List[m.Request] = []
        self._decoding_requests: List[m.Request] = []

    @property
    def prefill_requests(self) -> List[m.Request]:
        return self._prefill_requests

    @prefill_requests.setter
    def prefill_requests(self, value: List[m.Request]):
        self._prefill_requests = value

    @property
    def decoding_requests(self) -> List[m.Request]:
        return self._decoding_requests

    @property
    def all_requests(self) -> List[m.Request]:
        return self._prefill_requests + self._decoding_requests

    def __len__(self) -> int:
        return len(self._prefill_requests) + len(self._decoding_requests)
