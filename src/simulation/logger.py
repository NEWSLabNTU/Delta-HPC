import datetime
from pathlib import Path
from typing import Dict, List, Optional

from models import Agent, LLMEngine, AgentId, SimulationLogger as SimulationLoggerI

type LogMessage = str
type LogBuffer = List[LogMessage]


class SimulationLogger(SimulationLoggerI):
    def __init__(self, log_dir: str = "./logs", enabled: bool = True):
        self.log_dir = Path(log_dir)
        self.enabled = enabled
        self.buffer: LogBuffer = []
        self.buffer_size = 1000
        if not self.enabled:
            return
        self.log_dir.mkdir(parents=True, exist_ok=True)
        # Use datetime to create a unique file name
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.log_file = self.log_dir / f"simulation-{timestamp}.log"

    def log(self, message: LogMessage):
        if not self.enabled:
            return
        self.buffer.append(message)
        if len(self.buffer) >= self.buffer_size:
            self.flush()

    def flush(self):
        if not self.enabled or not self.buffer:
            return
        with open(self.log_file, "a") as f:
            for line in self.buffer:
                f.write(line + "\n")
        self.buffer.clear()

    def log_engine_step(
        self,
        current_time: float,
        agents: Dict[AgentId, Agent],
        stepping_engine: LLMEngine,
        next_arrival_time: Optional[float],
    ):
        """Logs the state of all engines for each agent with detailed progress."""
        if not self.enabled:
            return

        owner_id = None
        for aid, agent in agents.items():
            if stepping_engine in agent.engines:
                owner_id = aid
                break
        assert owner_id is not None

        lines = [
            f"[{current_time:.4f}] EVENT: ENGINE_STEP | "
            f"Stepping: {owner_id.value}-{stepping_engine.engine_id} | "
            f"Next Arrival: {next_arrival_time}"
        ]
        for aid, agent in agents.items():
            lines.append(f"  Agent: {aid.value}")
            for engine in agent.engines:
                # Log Owner + MIG
                is_stepping = " [STEPPING]" if engine is stepping_engine else ""
                lines.append(
                    f"    Engine: {aid.value}-{engine.engine_id} (Time: {engine.current_time:.4f}) [Status: {engine.status.value}]{is_stepping}"
                )
                # Log Waiting requests list
                if engine.waiting_queue:
                    req_ids = ", ".join([r.id for r in engine.waiting_queue])
                    lines.append(
                        f"      Waiting Requests: {len(engine.waiting_queue)} ({req_ids})"
                    )
                else:
                    lines.append(f"      Waiting Requests: 0")

                # Prefilling requests
                prefill = engine.running_queue.prefill_requests
                if prefill:
                    for req in prefill:
                        ftt_str = (
                            f"{req.first_token_time:.4f}"
                            if req.first_token_time is not None
                            else "None"
                        )
                        lines.append(
                            f"      Prefill: {req.id} | Progress: {req.prefilled_tokens}/{req.prompt_tokens} | FirstTokenTime: {ftt_str}"
                        )

                # Decoding requests
                decoding = engine.running_queue.decoding_requests
                if decoding:
                    for req in decoding:
                        ftt_str = (
                            f"{req.first_token_time:.4f}"
                            if req.first_token_time is not None
                            else "None"
                        )
                        lines.append(
                            f"      Decode: {req.id} | Gen: {req.generated_tokens}/{req.completion_tokens} | FirstTokenTime: {ftt_str}"
                        )

                if not prefill and not decoding:
                    lines.append(f"      No running requests")
        self.log("\n".join(lines))

    def log_request_arrival(
        self,
        current_time: float,
        req_id: str,
        target_agent: AgentId,
        assigned_engine: Optional[LLMEngine],
    ):
        """Logs a request arrival event."""
        if not self.enabled:
            return
        eng_str = (
            f"{target_agent.value}-{assigned_engine.engine_id}"
            if assigned_engine
            else "None"
        )
        msg = (
            f"[{current_time:.4f}] EVENT: REQUEST_ARRIVAL | "
            f"ReqId: {req_id} | Agent: {target_agent.value} | Engine: {eng_str}"
        )
        self.log(msg)

    def log_reallocation(
        self,
        current_time: float,
        giver_id: AgentId,
        receiver_id: AgentId,
        engine_id: str,
    ):
        """Logs a reallocation event."""
        if not self.enabled:
            return
        msg = (
            f"[{current_time:.4f}] EVENT: REALLOCATION | "
            f"Giver: {giver_id.value} | Receiver: {receiver_id.value} | "
            f"Engine: {engine_id}"
        )
        self.log(msg)

    def log_engine_restart_complete(
        self,
        current_time: float,
        engine_id: str,
        giver_id: AgentId,
        receiver_id: AgentId,
    ):
        """Logs an engine restart complete event with giver and receiver."""
        if not self.enabled:
            return
        msg = (
            f"[{current_time:.4f}] EVENT: ENGINE_RESTART_COMPLETE | "
            f"Engine: {engine_id} | Giver: {giver_id.value} | Receiver: {receiver_id.value}"
        )
        self.log(msg)
