import json
import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

from src.simulation.models import *


class SimulationLoggerImpl(SimulationLogger):
    def __init__(self, log_dir: str = "./logs", enabled: bool = True):
        self.log_dir = Path(log_dir)
        self.enabled = enabled
        self.buffer: List[str] = []
        self.buffer_size = 1000
        if not self.enabled:
            return
        self.log_dir.mkdir(parents=True, exist_ok=True)
        # Use datetime to create a unique file name
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.log_file = self.log_dir / f"simulation-{timestamp}.log"
        self.env_log_file = self.log_dir / f"env_state-{timestamp}.jsonl"

    def log_environment_state(self, current_time: float, state: EnvironmentStateData):
        if not self.enabled:
            return

        state_dict = {}
        for k, v in state.items():
            if k == "requests":
                continue
            if isinstance(v, dict):
                cleaned_v: Dict[str, Any] = {
                    str(key.value if hasattr(key, "value") else key): val  # type: ignore
                    for key, val in v.items()  # type: ignore
                }
                state_dict[k] = cleaned_v
            else:
                state_dict[k] = v

        record: Dict[str, Any] = {"time": current_time, "state": state_dict}
        with open(self.env_log_file, "a") as f:
            f.write(json.dumps(record) + "\n")

    def log(self, message: str):
        if not self.enabled:
            return
        # print(message)
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

        lines = [
            f"[{current_time:.4f}] EVENT: ENGINE_STEP | "
            f"Stepping: {stepping_engine.owner.agent_id.value}-{stepping_engine.engine_id} | "
            f"Next Arrival: {next_arrival_time:.4f}"
        ]
        for aid, agent in agents.items():
            lines.append(f"  Agent: {aid.value}")
            for engine in agent.engines:
                # Log Owner + MIG
                is_stepping = " [STEPPING]" if engine is stepping_engine else ""
                p_suffix = " [P]" if engine.is_permanent else ""
                lines.append(
                    f"    Engine: {engine.engine_id}{p_suffix} (Time: {engine.current_time:.4f}) [Status: {engine.status.value}]{is_stepping}"
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
                        lines.append(
                            f"      Prefill: {req.id} | Progress: {req.prefilled_tokens}/{req.prompt_tokens}"
                        )

                # Decoding requests
                decoding = engine.running_queue.decoding_requests
                if decoding:
                    for req in decoding:
                        lines.append(
                            f"      Decode: {req.id} | Gen: {req.generated_tokens}/{req.completion_tokens}"
                        )

                if not prefill and not decoding:
                    lines.append(f"      No running requests")
        self.log("\n".join(lines))

    def log_request_arrival(
        self,
        current_time: float,
        req: Request,
        eng: Optional[LLMEngine],
    ):
        """Logs a request arrival event."""
        if not self.enabled:
            return
        eng_str = f"{req.agent_id.value}-{eng.engine_id}" if eng else "None"
        msg = (
            f"[{current_time:.4f}] EVENT: REQUEST_ARRIVAL | "
            f"ReqId: {req.id} | Agent: {req.agent_id.value} | Engine: {eng_str}"
        )
        self.log(msg)

    def log_rag_search_complete(
        self, current_time: float, req: Request, eng: Optional[LLMEngine]
    ):
        if not self.enabled:
            return
        eng_str = f"{req.agent_id.value}-{eng.engine_id}" if eng else "None"
        msg = (
            f"[{current_time:.4f}] EVENT: RAG_SEARCH_COMPLETE | "
            f"ReqId: {req.id} | Agent: {req.agent_id.value} | Engine: {eng_str}"
        )
        self.log(msg)

    def log_vram_transfer(
        self,
        current_time: float,
        giver_id: AgentId,
        receiver_id: AgentId,
        amount: int,
        eids: List[str],
    ):
        """Logs a VRAM transfer event."""
        if not self.enabled:
            return
        msg = (
            f"[{current_time:.4f}] EVENT: VRAM_TRANSFER | "
            f"Giver: {giver_id.value} | Receiver: {receiver_id.value} | "
            f"Amount: {amount}GB | Source Engine: {", ".join(eids)}"
        )
        self.log(msg)

    def log_discard_vram_transfer(self, current_time: float, detail: TransferDetails):
        if not self.enabled:
            return
        msg = (
            f"[{current_time:.4f}] VRAM_TRANSFER_DROP | "
            f"GIVER: {detail.giver_id} | RECEIVER: {detail.receiver_id} | AMOUNT: {detail.amount}"
        )
        self.log(msg)

    def log_engine_boot_complete(
        self,
        current_time: float,
        engine_id: str,
    ):
        """Logs an engine boot complete event with giver and receiver."""
        if not self.enabled:
            return
        msg = f"[{current_time:.4f}] EVENT: ENGINE_BOOT_COMPLETE | Engine: {engine_id}"
        self.log(msg)

    def log_mig_merge_trigger(self, current_time: float, eids: List[str], gpu: int):
        if not self.enabled:
            return
        self.log(
            f"[{current_time:.4f}] EVENT: MIG_MERGE_TRIGGER | Engines: {", ".join(eids)} | GPU: {gpu}"
        )

    def log_mig_split_trigger(self, current_time: float, engine_id: str, gpu: int):
        if not self.enabled:
            return
        self.log(
            f"[{current_time:.4f}] EVENT: MIG_SPLIT_TRIGGER | Engine: {engine_id} | GPU: {gpu}"
        )

    def log_mig_merge_complete(self, current_time: float, new_engine_id: str):
        if not self.enabled:
            return
        self.log(
            f"[{current_time:.4f}] EVENT: MIG_MERGE_COMPLETE | New Engine: {new_engine_id}"
        )

    def log_mig_split_complete(self, current_time: float, engine_id: str):
        if not self.enabled:
            return
        self.log(
            f"[{current_time:.4f}] EVENT: MIG_SPLIT_COMPLETE | Original: {engine_id}"
        )
