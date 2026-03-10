import heapq
import random
import math
from typing import List, Optional, Tuple, Dict
from src.simulation.models import (
    Request,
    RequestState,
    EventType,
    SimulationEvent,
    RunningRequests,
)


class LLMEngine:
    def __init__(
        self,
        engine_id: str,
        model_name: str,
        mig_profile: str,
        max_batched_tokens: int,
        prefill_params: dict,
        tpot_params: dict,
    ):
        self.engine_id = engine_id
        self.model_name = model_name
        self.mig_profile = mig_profile
        self.max_batched_tokens = max_batched_tokens

        # Regression params
        self.prefill_alpha = prefill_params.get("alpha", 0.0)
        self.prefill_beta = prefill_params.get("beta", 0.0)
        self.prefill_sigma = prefill_params.get("sigma", 0.0)
        self.tpot_alpha = tpot_params.get("alpha", 0.0)
        self.tpot_beta = tpot_params.get("beta", 0.0)
        self.tpot_sigma = tpot_params.get("sigma", 0.0)

        # Queues
        self.waiting_queue: List[Request] = []
        self.running_queue = RunningRequests()

        # State
        self.current_time: float = 0.0
        self.is_busy: bool = False

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

    def add_request(self, request: Request, current_time: float):
        self.waiting_queue.append(request)
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
            req_prefill_tokens = {}

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

            self.is_busy = True

            # Determine steps and durations
            if total_prefill_tokens > 0:
                step_duration = self.get_prefill_time(total_prefill_tokens)

                duration = step_duration
                for req in self.running_queue.prefill_requests:
                    req.prefilled_tokens += req_prefill_tokens.get(req.id, 0)
                for r in self.running_queue.decoding_requests:
                    r.generated_tokens += 1
                    if r.first_token_time is None:
                        r.first_token_time = self.current_time + step_duration

                # Cleanup prefill_requests
                new_prefill = []
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
            self.is_busy = False

        if steps_taken > 0 or finished:
            return SimulationEvent(
                time=self.current_time,
                event_type=EventType.ENGINE_STEP_COMPLETE,
                payload={"engine_id": self.engine_id, "steps_taken": steps_taken},
            )
        else:
            self.is_busy = False
            return None
