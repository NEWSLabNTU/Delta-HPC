from __future__ import annotations
import random
from typing import List, Dict, Optional

import src.simulation.models as m
import src.simulation.utils as utils
from src.simulation.request import RunningRequestsImpl


class LLMEngineImpl(m.LLMEngine):
    @classmethod
    def create(
        cls,
        gpu: int,
        engine_id: str,
        owner: m.Agent,
        mig_profile: m.MIGProfile,
        current_time: float,
        is_permanent: bool = False,
    ) -> LLMEngineImpl:
        mname = utils.SIM_CONFIG.get_model(owner.agent_id, mig_profile)

        return cls(
            gpu=gpu,
            engine_id=engine_id,
            owner=owner,
            model_name=mname,
            mig_profile=mig_profile,
            max_batched_tokens=utils.SIM_CONFIG.max_batched_tokens[mname],
            prefill_params=utils.SIM_CONFIG.get_prefill_params(
                owner.agent_id, mig_profile
            ),
            tpot_params=utils.SIM_CONFIG.get_tpot_params(owner.agent_id, mig_profile),
            restart_time=utils.SIM_CONFIG.get_restart_time(owner.agent_id, mig_profile),
            current_time=current_time,
            is_permanent=is_permanent,
        )

    def __init__(
        self,
        gpu: int,
        engine_id: str,
        owner: m.Agent,
        model_name: str,
        mig_profile: m.MIGProfile,
        max_batched_tokens: int,
        prefill_params: m.ParamDict,
        tpot_params: m.ParamDict,
        restart_time: float,
        current_time: float = 0.0,
        is_permanent: bool = False,
    ):
        self._gpu = gpu
        self._engine_id = engine_id
        self._owner = owner
        self._model_name = model_name
        self._mig_profile = mig_profile
        self._is_permanent = is_permanent

        # Regression params
        self._prefill_alpha = prefill_params["alpha"]
        self._prefill_beta = prefill_params["beta"]
        self._prefill_sigma = prefill_params["sigma"]
        self._tpot_alpha = tpot_params["alpha"]
        self._tpot_beta = tpot_params["beta"]
        self._tpot_sigma = tpot_params["sigma"]
        self._restart_time = restart_time
        self._max_batched_tokens = max_batched_tokens
        self._max_kv_cache_tokens = utils.SIM_CONFIG.get_max_kv_cache_tokens(
            owner.agent_id, mig_profile
        )

        # Queues
        self._waiting_queue: List[m.Request] = []
        self._running_queue = RunningRequestsImpl()

        # State
        self._current_time: float = current_time
        self._status: m.EngineStatus = m.EngineStatus.ACTIVE
        self._shutdown_pending: Optional[m.ShutdownPayload] = None

    @property
    def gpu(self) -> int:
        return self._gpu

    @property
    def engine_id(self) -> str:
        return self._engine_id

    @property
    def owner(self) -> m.Agent:
        return self._owner

    @owner.setter
    def owner(self, value: m.Agent):
        self._owner = value

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def mig_profile(self) -> m.MIGProfile:
        return self._mig_profile

    @property
    def current_time(self) -> float:
        return self._current_time

    @property
    def status(self) -> m.EngineStatus:
        return self._status

    @property
    def waiting_queue(self) -> List[m.Request]:
        return self._waiting_queue

    @property
    def running_queue(self) -> m.RunningRequests:
        return self._running_queue

    @property
    def is_permanent(self) -> bool:
        return self._is_permanent

    @property
    def current_kv_utilization(self) -> float:
        current = 0
        for req in self._running_queue.all_requests:
            match req.state:
                case m.RequestState.PREFILLING:
                    current += req.prefilled_tokens
                case m.RequestState.DECODING:
                    current += req.prompt_tokens + req.generated_tokens
                case m.RequestState.PENDING | m.RequestState.COMPLETED:
                    pass
        return current / self._max_kv_cache_tokens

    def _get_tpot(self, concurrent_requests: int) -> float:
        """Calculate Time Per Output Token using linear regression params with Gaussian noise."""
        mu = self._tpot_alpha + self._tpot_beta * concurrent_requests
        return max(0.0, random.gauss(mu, self._tpot_sigma))

    def _get_prefill_time(self, num_tokens: int) -> float:
        """Calculate prefill time using polynomial regression params with Gaussian noise"""
        mu = self._prefill_alpha + self._prefill_beta * num_tokens
        return max(0.0, random.gauss(mu, self._prefill_sigma))

    def update_model(
        self,
        new_owner: m.Agent,
        model_name: str,
        max_batched_tokens: int,
        prefill_params: m.ParamDict,
        tpot_params: m.ParamDict,
        restart_time: float,
    ):
        if self in self._owner.engines:
            self._owner.engines.remove(self)
        new_owner.add_engine(self)
        self._owner = new_owner

        self._model_name = model_name
        self._max_batched_tokens = max_batched_tokens
        self._max_kv_cache_tokens = utils.SIM_CONFIG.get_max_kv_cache_tokens(
            new_owner.agent_id, self.mig_profile
        )
        self._prefill_alpha = prefill_params["alpha"]
        self._prefill_beta = prefill_params["beta"]
        self._prefill_sigma = prefill_params["sigma"]
        self._tpot_alpha = tpot_params["alpha"]
        self._tpot_beta = tpot_params["beta"]
        self._tpot_sigma = tpot_params["sigma"]
        self._restart_time = restart_time

    def add_request(self, request: m.Request, current_time: float):
        assert (
            self._status == m.EngineStatus.ACTIVE
        ), f"Cannot add request to {self._engine_id} while {self._status}"
        self._waiting_queue.append(request)
        self._current_time = max(self._current_time, current_time)

    def trigger_shutdown(
        self, payload: m.ShutdownPayload, current_time: float
    ) -> Optional[m.SimulationEvent]:
        """Trigger engine shutdown for reallocation or MIG operations."""
        self._status = m.EngineStatus.DRAINING
        self._current_time = max(self._current_time, current_time)
        self._shutdown_pending = payload

        if len(self._running_queue) == 0 and not self._waiting_queue:
            return self._start_shutdown()
        return None

    def _start_shutdown(self) -> m.SimulationEvent:
        """Emit the stored shutdown payload as a SHUTDOWN_COMPLETE event."""
        assert self._shutdown_pending is not None
        payload = self._shutdown_pending
        self._shutdown_pending = None
        return m.SimulationEvent(
            time=self._current_time,
            event_type=m.EventType.ENGINE_SHUTDOWN_COMPLETE,
            payload=payload,
        )

    def trigger_boot(self, payload: m.BootPayload) -> m.SimulationEvent:
        """Move to BOOTING and schedule boot completion."""
        self._status = m.EngineStatus.BOOTING
        return m.SimulationEvent(
            time=self._current_time + self._restart_time,
            event_type=m.EventType.ENGINE_BOOT_COMPLETE,
            payload=payload,
        )

    def predict_drain_time(self) -> float:
        """Predict the time needed to drain current requests."""
        all_reqs = self._waiting_queue + self._running_queue.all_requests
        if not all_reqs:
            return 0.0

        # Estimate prefill time
        remaining_prefill = sum(
            r.remaining_prefill_tokens for r in all_reqs if not r.prefill_completed
        )
        prefill_time = (
            self._get_prefill_time(remaining_prefill) if remaining_prefill > 0 else 0.0
        )

        # Estimate decode time
        max_decode_steps = max(
            (r.completion_tokens - r.generated_tokens for r in all_reqs), default=0
        )
        decode_time = 0.0
        num_reqs = len(all_reqs)
        for _ in range(max_decode_steps):
            decode_time += self._get_tpot(num_reqs)

        return prefill_time + decode_time

    def activate(self, current_time: float):
        """Move from BOOTING to ACTIVE."""
        self._status = m.EngineStatus.ACTIVE
        self._current_time = max(self._current_time, current_time)

    def step(
        self, current_time: float, next_arrival_time: Optional[float] = None
    ) -> Optional[m.SimulationEvent]:
        """
        Calculates the next forward pass duration and state updates.
        Returns the finish Event of this step, or None if idle.
        """
        self._current_time = max(self._current_time, current_time)

        steps_taken = 0
        while True:
            # 1. Break condition: Check if any request finished
            if any(r.is_finished for r in self._running_queue.all_requests):
                break

            # 2. Break condition: Reached next arrival time
            if (
                next_arrival_time is not None
                and self._current_time >= next_arrival_time
            ):
                break

            # Compute budget for prefill
            budget = self._max_batched_tokens - len(
                self._running_queue.decoding_requests
            )
            total_prefill_tokens = 0
            req_prefill_tokens: Dict[str, int] = {}

            # 3a. Schedule existing prefill requests (the chunked one from previous step, if any)
            for req in list(self._running_queue.prefill_requests):
                if budget <= 0:
                    break
                tokens = min(req.remaining_prefill_tokens, budget)
                req_prefill_tokens[req.id] = tokens
                total_prefill_tokens += tokens
                budget -= tokens

            # 3b. Pull from waiting_queue if we still have budget
            while budget > 0 and self._waiting_queue:
                req = self._waiting_queue[0]

                # Check KV Cache limits before pulling
                req_tokens = req.prompt_tokens + req.completion_tokens
                current_reserved = sum(
                    r.prompt_tokens + r.completion_tokens
                    for r in self._running_queue.all_requests
                )
                if current_reserved + req_tokens > self._max_kv_cache_tokens:
                    break

                req = self._waiting_queue.pop(0)
                req.state = m.RequestState.PREFILLING
                if req.start_time is None:
                    req.start_time = self._current_time
                self._running_queue.prefill_requests.append(req)

                tokens = min(req.remaining_prefill_tokens, budget)
                req_prefill_tokens[req.id] = tokens
                total_prefill_tokens += tokens
                budget -= tokens

            if len(self._running_queue) == 0:
                break

            # Determine steps and durations
            if total_prefill_tokens > 0:
                duration = self._get_prefill_time(total_prefill_tokens)
                for req in self._running_queue.prefill_requests:
                    req.prefilled_tokens += req_prefill_tokens.get(req.id, 0)
                for r in self._running_queue.decoding_requests:
                    r.generated_tokens += 1
                    r.decode_time += duration
                    if r.first_token_time is None:
                        r.first_token_time = self._current_time + duration

                # Cleanup prefill_requests
                new_prefill: List[m.Request] = []
                for req in self._running_queue.prefill_requests:
                    if req.prefill_completed:
                        req.state = m.RequestState.DECODING
                        self._running_queue.decoding_requests.append(req)
                    else:
                        new_prefill.append(req)
                self._running_queue.prefill_requests = new_prefill

            else:
                num_decodes = len(self._running_queue.decoding_requests)
                duration = self._get_tpot(num_decodes)

                for r in self._running_queue.decoding_requests:
                    r.generated_tokens += 1
                    r.decode_time += duration
                    if r.first_token_time is None:
                        r.first_token_time = self._current_time + duration

            # Advance Time
            self._current_time += duration
            steps_taken += 1

        # Once loop is broken, move finished out
        finished = [r for r in self._running_queue.all_requests if r.is_finished]
        for r in finished:
            r.state = m.RequestState.COMPLETED
            r.finish_time = self._current_time
            self._owner.completed_requests.append(r)

            if r in self._running_queue.prefill_requests:
                self._running_queue.prefill_requests.remove(r)
            elif r in self._running_queue.decoding_requests:
                self._running_queue.decoding_requests.remove(r)

        if len(self._running_queue) == 0 and not self._waiting_queue:
            if (
                self._status == m.EngineStatus.DRAINING
                and self._shutdown_pending is not None
            ):
                return self._start_shutdown()

        if steps_taken > 0 or finished:
            return m.SimulationEvent(
                time=self._current_time,
                event_type=m.EventType.ENGINE_STEP_COMPLETE,
                payload={"engine_id": self._engine_id, "steps_taken": steps_taken},
            )
        else:
            return None
