import random
from typing import List, Dict, Optional
from models import (
    Request,
    RequestState,
    EventType,
    SimulationEvent,
    RunningRequests,
    EngineStatus,
    ParamDict,
    AgentId,
)


class LLMEngine:
    def __init__(
        self,
        engine_id: str,
        owner_id: AgentId,
        model_name: str,
        mig_profile: str,
        max_batched_tokens: int,
        prefill_params: ParamDict,
        tpot_params: ParamDict,
        restart_time: float,
    ):
        self.engine_id = engine_id
        self.model_name = model_name
        self.mig_profile = mig_profile
        self.max_batched_tokens = max_batched_tokens
        self.owner_id: AgentId = owner_id

        # Regression params
        self.prefill_alpha = prefill_params["alpha"]
        self.prefill_beta = prefill_params["beta"]
        self.prefill_sigma = prefill_params["sigma"]
        self.tpot_alpha = tpot_params["alpha"]
        self.tpot_beta = tpot_params["beta"]
        self.tpot_sigma = tpot_params["sigma"]

        # Queues
        self.waiting_queue: List[Request] = []
        self.running_queue = RunningRequests()

        # State
        self.current_time: float = 0.0
        self.status: EngineStatus = EngineStatus.ACTIVE
        self.restart_time = restart_time

        # Output collection
        self.completed_requests: List[Request] = []

    def get_tpot(self, concurrent_requests: int) -> float:
        """Calculate Time Per Output Token using linear regression params with Gaussian noise."""
        mu = self.tpot_alpha + self.tpot_beta * concurrent_requests
        return max(0.0, random.gauss(mu, self.tpot_sigma))

    def get_prefill_time(self, num_tokens: int) -> float:
        """Calculate prefill time using polynomial regression params with Gaussian noise"""
        mu = self.prefill_alpha + self.prefill_beta * num_tokens
        return max(0.0, random.gauss(mu, self.prefill_sigma))

    def update_model(
        self,
        model_name: str,
        max_batched_tokens: int,
        prefill_params: ParamDict,
        tpot_params: ParamDict,
        restart_time: float,
    ):
        self.model_name = model_name
        self.max_batched_tokens = max_batched_tokens
        self.prefill_alpha = prefill_params["alpha"]
        self.prefill_beta = prefill_params["beta"]
        self.prefill_sigma = prefill_params["sigma"]
        self.tpot_alpha = tpot_params["alpha"]
        self.tpot_beta = tpot_params["beta"]
        self.tpot_sigma = tpot_params["sigma"]
        self.restart_time = restart_time

    def add_request(self, request: Request, current_time: float):
        assert (
            self.status == EngineStatus.ACTIVE
        ), f"Cannot add request to {self.engine_id} while {self.status}"
        self.waiting_queue.append(request)
        self.current_time = max(self.current_time, current_time)

    def trigger_reallocation(self, current_time: float) -> Optional[SimulationEvent]:
        self.status = EngineStatus.DRAINING
        self.current_time = max(self.current_time, current_time)

        if len(self.running_queue) == 0 and not self.waiting_queue:
            return self._start_restart()
        return None

    def _start_restart(self) -> SimulationEvent:
        self.status = EngineStatus.RESTARTING
        return SimulationEvent(
            time=self.current_time + self.restart_time,
            event_type=EventType.ENGINE_RESTART_COMPLETE,
            payload={"engine_id": self.engine_id},
        )

    def finish_restart(self, current_time: float):
        self.status = EngineStatus.ACTIVE
        self.current_time = max(self.current_time, current_time)

    def step(
        self, current_time: float, next_arrival_time: Optional[float] = None
    ) -> Optional[SimulationEvent]:
        """
        Calculates the next forward pass duration and state updates.
        Returns the finish Event of this step, or None if idle.
        """
        self.current_time = max(self.current_time, current_time)

        steps_taken = 0
        while True:
            # 1. Break condition: Check if any request finished
            if any(r.is_finished for r in self.running_queue.all_requests):
                break

            # 2. Break condition: Reached next arrival time
            if next_arrival_time is not None and self.current_time >= next_arrival_time:
                break

            # Compute budget for prefill
            budget = self.max_batched_tokens - len(self.running_queue.decoding_requests)
            total_prefill_tokens = 0
            req_prefill_tokens: Dict[str, int] = {}

            # 3a. Schedule existing prefill requests (the chunked one from previous step, if any)
            for req in list(self.running_queue.prefill_requests):
                if budget <= 0:
                    break
                tokens = min(req.remaining_prefill_tokens, budget)
                req_prefill_tokens[req.id] = tokens
                total_prefill_tokens += tokens
                budget -= tokens

            # 3b. Pull from waiting_queue if we still have budget
            while budget > 0 and self.waiting_queue:
                req = self.waiting_queue.pop(0)
                req.state = RequestState.PREFILLING
                if req.start_time is None:
                    req.start_time = self.current_time
                self.running_queue.prefill_requests.append(req)

                tokens = min(req.remaining_prefill_tokens, budget)
                req_prefill_tokens[req.id] = tokens
                total_prefill_tokens += tokens
                budget -= tokens

            if len(self.running_queue) == 0:
                break

            # Determine steps and durations
            if total_prefill_tokens > 0:
                duration = self.get_prefill_time(total_prefill_tokens)
                for req in self.running_queue.prefill_requests:
                    req.prefilled_tokens += req_prefill_tokens.get(req.id, 0)
                for r in self.running_queue.decoding_requests:
                    r.generated_tokens += 1
                    r.decode_time += duration
                    if r.first_token_time is None:
                        r.first_token_time = self.current_time + duration

                # Cleanup prefill_requests
                new_prefill: List[Request] = []
                for req in self.running_queue.prefill_requests:
                    if req.prefill_completed:
                        req.state = RequestState.DECODING
                        self.running_queue.decoding_requests.append(req)
                    else:
                        new_prefill.append(req)
                self.running_queue.prefill_requests = new_prefill

            else:
                num_decodes = len(self.running_queue.decoding_requests)
                duration = self.get_tpot(num_decodes)

                for r in self.running_queue.decoding_requests:
                    r.generated_tokens += 1
                    r.decode_time += duration
                    if r.first_token_time is None:
                        r.first_token_time = self.current_time + duration

            # Advance Time
            self.current_time += duration
            steps_taken += 1

        # Once loop is broken, move finished out
        finished = [r for r in self.running_queue.all_requests if r.is_finished]
        for r in finished:
            r.state = RequestState.COMPLETED
            r.finish_time = self.current_time
            self.completed_requests.append(r)

            if r in self.running_queue.prefill_requests:
                self.running_queue.prefill_requests.remove(r)
            elif r in self.running_queue.decoding_requests:
                self.running_queue.decoding_requests.remove(r)

        if len(self.running_queue) == 0 and not self.waiting_queue:
            if self.status == EngineStatus.DRAINING:
                return self._start_restart()

        if steps_taken > 0 or finished:
            return SimulationEvent(
                time=self.current_time,
                event_type=EventType.ENGINE_STEP_COMPLETE,
                payload={"engine_id": self.engine_id, "steps_taken": steps_taken},
            )
        else:
            return None
