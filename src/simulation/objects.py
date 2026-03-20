from __future__ import annotations
import global_vars as g
import random
from typing import List, Dict, Optional

from models import *
from request import RunningRequestsImpl


class AgentImpl(Agent):
    def __init__(self, agent_id: AgentId):
        self._agent_id = agent_id
        self._engines: List[LLMEngine] = []
        self._completed_requests: List[Request] = []
        self._dispatch_queue: List[Request] = []

    @property
    def agent_id(self) -> AgentId:
        return self._agent_id

    @property
    def engines(self) -> List[LLMEngine]:
        return self._engines

    @property
    def completed_requests(self) -> List[Request]:
        return self._completed_requests

    @property
    def dispatch_queue(self) -> List[Request]:
        return self._dispatch_queue

    def add_engine(self, engine: LLMEngine):
        self._engines.append(engine)

    def dispatch(self, request: Request, current_time: float) -> Optional[LLMEngine]:
        """
        Dispatches an incoming request to the best engine based on simple work-balance.
        Finds engines matching the requested model size, and picks the one with the smallest queue length.
        Sets completion_tokens based on the chosen engine's model before queuing the request.
        If no active engines exist, queues the request in the agent's waiting queue.
        """
        active_engines = [e for e in self.engines if e.status == EngineStatus.ACTIVE]
        if not active_engines:
            self._dispatch_queue.append(request)
            return None

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

    def process_waiting_queue(self, current_time: float) -> None:
        """Process queued requests if there are active engines."""
        while self._dispatch_queue:
            active_engines = [
                e for e in self.engines if e.status == EngineStatus.ACTIVE
            ]
            if not active_engines:
                break
            req = self._dispatch_queue.pop(0)
            self.dispatch(req, current_time)


class LLMEngineImpl(LLMEngine):
    @staticmethod
    def create(
        gpu: int,
        engine_id: str,
        owner: Agent,
        mig_profile: MIGProfile,
        current_time: float,
    ) -> "LLMEngineImpl":
        """Factory: create an LLMEngine with configuration loaded from SIM_CONFIG."""
        mname = g.SIM_CONFIG.get_model(owner.agent_id, mig_profile)
        return LLMEngineImpl(
            gpu=gpu,
            engine_id=engine_id,
            owner=owner,
            model_name=mname,
            mig_profile=mig_profile,
            max_batched_tokens=g.SIM_CONFIG.max_batched_tokens[mname],
            prefill_params=g.SIM_CONFIG.get_prefill_params(owner.agent_id, mig_profile),
            tpot_params=g.SIM_CONFIG.get_tpot_params(owner.agent_id, mig_profile),
            restart_time=g.SIM_CONFIG.get_restart_time(owner.agent_id, mig_profile),
            current_time=current_time,
        )

    def __init__(
        self,
        gpu: int,
        engine_id: str,
        owner: Agent,
        model_name: str,
        mig_profile: MIGProfile,
        max_batched_tokens: int,
        prefill_params: ParamDict,
        tpot_params: ParamDict,
        restart_time: float,
        current_time: float = 0.0,
    ):
        self._gpu = gpu
        self._engine_id = engine_id
        self._owner = owner
        self._model_name = model_name
        self._mig_profile = mig_profile

        # Regression params
        self.prefill_alpha = prefill_params["alpha"]
        self.prefill_beta = prefill_params["beta"]
        self.prefill_sigma = prefill_params["sigma"]
        self.tpot_alpha = tpot_params["alpha"]
        self.tpot_beta = tpot_params["beta"]
        self.tpot_sigma = tpot_params["sigma"]
        self.restart_time = restart_time
        self.max_batched_tokens = max_batched_tokens

        # Queues
        self._waiting_queue: List[Request] = []
        self._running_queue = RunningRequestsImpl()

        # State
        self._current_time: float = current_time
        self._status: EngineStatus = EngineStatus.ACTIVE
        self._shutdown_pending: Optional[ShutdownPayload] = None

    @property
    def gpu(self) -> int:
        return self._gpu

    @property
    def engine_id(self) -> str:
        return self._engine_id

    @property
    def owner(self) -> Agent:
        return self._owner

    @owner.setter
    def owner(self, value: Agent):
        self._owner = value

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def mig_profile(self) -> MIGProfile:
        return self._mig_profile

    @property
    def current_time(self) -> float:
        return self._current_time

    @property
    def status(self) -> EngineStatus:
        return self._status

    @property
    def waiting_queue(self) -> List[Request]:
        return self._waiting_queue

    @property
    def running_queue(self) -> RunningRequests:
        return self._running_queue

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
        self._model_name = model_name
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
            self._status == EngineStatus.ACTIVE
        ), f"Cannot add request to {self._engine_id} while {self._status}"
        self._waiting_queue.append(request)
        self._current_time = max(self._current_time, current_time)

    def trigger_shutdown(
        self, payload: ShutdownPayload, current_time: float
    ) -> Optional[SimulationEvent]:
        """Trigger engine shutdown for reallocation or MIG operations."""
        self._status = EngineStatus.DRAINING
        self._current_time = max(self._current_time, current_time)
        self._shutdown_pending = payload

        if len(self._running_queue) == 0 and not self._waiting_queue:
            return self._start_shutdown()
        return None

    def _start_shutdown(self) -> SimulationEvent:
        """Emit the stored shutdown payload as a SHUTDOWN_COMPLETE event."""
        assert self._shutdown_pending is not None
        payload = self._shutdown_pending
        self._shutdown_pending = None
        return SimulationEvent(
            time=self._current_time,
            event_type=EventType.ENGINE_SHUTDOWN_COMPLETE,
            payload=payload,
        )

    def trigger_boot(self, payload: BootPayload) -> SimulationEvent:
        """Move to BOOTING and schedule boot completion."""
        self._status = EngineStatus.BOOTING
        return SimulationEvent(
            time=self._current_time + self.restart_time,
            event_type=EventType.ENGINE_BOOT_COMPLETE,
            payload=payload,
        )

    def activate(self, current_time: float):
        """Move from BOOTING to ACTIVE."""
        self._status = EngineStatus.ACTIVE
        self._current_time = max(self._current_time, current_time)

    def step(
        self, current_time: float, next_arrival_time: Optional[float] = None
    ) -> Optional[SimulationEvent]:
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
            budget = self.max_batched_tokens - len(
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
                req = self._waiting_queue.pop(0)
                req.state = RequestState.PREFILLING
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
                duration = self.get_prefill_time(total_prefill_tokens)
                for req in self._running_queue.prefill_requests:
                    req.prefilled_tokens += req_prefill_tokens.get(req.id, 0)
                for r in self._running_queue.decoding_requests:
                    r.generated_tokens += 1
                    r.decode_time += duration
                    if r.first_token_time is None:
                        r.first_token_time = self._current_time + duration

                # Cleanup prefill_requests
                new_prefill: List[Request] = []
                for req in self._running_queue.prefill_requests:
                    if req.prefill_completed:
                        req.state = RequestState.DECODING
                        self._running_queue.decoding_requests.append(req)
                    else:
                        new_prefill.append(req)
                self._running_queue.prefill_requests = new_prefill

            else:
                num_decodes = len(self._running_queue.decoding_requests)
                duration = self.get_tpot(num_decodes)

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
            r.state = RequestState.COMPLETED
            r.finish_time = self._current_time
            self._owner.completed_requests.append(r)

            if r in self._running_queue.prefill_requests:
                self._running_queue.prefill_requests.remove(r)
            elif r in self._running_queue.decoding_requests:
                self._running_queue.decoding_requests.remove(r)

        if len(self._running_queue) == 0 and not self._waiting_queue:
            if (
                self._status == EngineStatus.DRAINING
                and self._shutdown_pending is not None
            ):
                return self._start_shutdown()

        if steps_taken > 0 or finished:
            return SimulationEvent(
                time=self._current_time,
                event_type=EventType.ENGINE_STEP_COMPLETE,
                payload={"engine_id": self._engine_id, "steps_taken": steps_taken},
            )
        else:
            return None
