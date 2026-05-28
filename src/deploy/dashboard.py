import logging
import time
from aiohttp import web
from typing import Any, Dict, Optional

import src.share.models as m
from src.deploy.system import SYSTEM_STATE
from src.deploy.obs import OBS_COLLECTOR
from src.training.config import TRAINING_CONFIG
import io
import contextlib
from src.deploy.report import print_benchmark_report

logger = logging.getLogger(__name__)

# ANSI Color Codes for Premium Terminal View
C_RESET = "\033[0m"
C_BOLD = ""
C_CYAN = "\033[36m"
C_GREEN = "\033[32m"
C_PURPLE = "\033[35m"
C_YELLOW = "\033[33m"
C_RED = "\033[31m"
C_WHITE = "\033[37m"
C_BLUE = "\033[34m"
C_GRAY = "\033[90m"


class DashboardServer:
    def __init__(self, publisher: Any, host: str = "0.0.0.0", port: int = 9000):
        self.publisher = publisher
        self.host = host
        self.port = port
        self.runner: Optional[web.AppRunner] = None
        self.site: Optional[web.TCPSite] = None
        self.start_time = time.time()
        self.live_slot_metrics: Dict[str, Dict[str, float]] = {}

    def record_live_slot_metrics(self, mig_uuid: str, metrics: Dict[str, float]):
        self.live_slot_metrics[mig_uuid] = metrics

    def get_live_slot_metrics(self, mig_uuid: str) -> Optional[Dict[str, float]]:
        return self.live_slot_metrics.get(mig_uuid)

    def clear_live_slot_metrics(self, mig_uuid: str):
        self.live_slot_metrics.pop(mig_uuid, None)

    async def start(self):
        app = web.Application()
        app.router.add_get("/", self.handle_index)
        app.router.add_get("/api/data", self.handle_api_data)
        app.router.add_get("/api/observation", self.handle_api_observation)
        app.router.add_get("/api/report", self.handle_api_report)

        self.runner = web.AppRunner(app, access_log=None)
        await self.runner.setup()
        self.site = web.TCPSite(self.runner, self.host, self.port)
        await self.site.start()
        logger.info(f"Dashboard Web Server running on http://{self.host}:{self.port}")

    async def stop(self):
        if self.site:
            await self.site.stop()
        if self.runner:
            await self.runner.cleanup()
        logger.info("Dashboard Web Server stopped.")

    async def handle_api_data(self, request: web.Request) -> web.Response:
        # Collect raw JSON data for other automation tools/scripts
        gpus_data = []
        for gpu_idx, gpu_state in SYSTEM_STATE.gpus.items():
            slots_data = []
            for slot in gpu_state.slots:
                slots_data.append({
                    "start_slice": slot.profile_placement.start_slice,
                    "size": slot.profile_placement.profile.size,
                    "vram": slot.profile_placement.profile.vram,
                    "profile_name": slot.profile_placement.profile.profile_type.name,
                    "agent_owner": slot.agent_id.name if slot.agent_id else None,
                    "is_ready": slot.is_ready,
                    "is_draining": slot.is_draining,
                    "port": slot.port,
                })
            gpus_data.append({
                "gpu_idx": gpu_idx,
                "is_simulated": gpu_state.is_simulated,
                "slots": slots_data,
            })

        agents_data = {}
        for agent_id in [m.AgentId.CODING, m.AgentId.RAG]:
            engines_list = []
            for gpu_idx, gpu_state in SYSTEM_STATE.gpus.items():
                for slot in gpu_state.slots:
                    if slot.agent_id == agent_id:
                        running = 0
                        waiting = 0
                        if not gpu_state.is_simulated:
                            # Attempt to use live per-slot metrics cache for accurate dashboard representation
                            live = self.get_live_slot_metrics(slot.mig_uuid)
                            if live is not None and slot.is_ready:
                                running = live["running"]
                                waiting = live["waiting"]
                            else:
                                running = 0
                                waiting = 0
                        else:
                            mig_uuid = slot.mig_uuid
                            running = len(
                                self.publisher.vllm_manager._active_reqs.get(
                                    mig_uuid, {}
                                )
                            )
                            waiting = self.publisher.vllm_manager.get_sim_waiting(
                                mig_uuid
                            )

                        engines_list.append({
                            "name": slot.profile_placement.profile.profile_type.name,
                            "gpu_idx": slot.gpu_idx,
                            "start_slice": slot.profile_placement.start_slice,
                            "running_requests": int(running),
                            "waiting_queue_length": int(waiting),
                            "is_ready": slot.is_ready,
                            "is_draining": slot.is_draining,
                            "is_permanent": slot.port is None,
                        })
            agents_data[agent_id.name] = engines_list

        data = {
            "budget": float(OBS_COLLECTOR._current_budget),
            "reconfig_history": OBS_COLLECTOR.reconfig_history,
            "gpus": gpus_data,
            "agents": agents_data,
            "system_time": time.time(),
            "completed_requests": getattr(self.publisher, "completed_requests", 0),
            "total_requests": getattr(self.publisher, "total_requests", 0),
        }
        return web.json_response(data)

    async def handle_api_observation(self, request: web.Request) -> web.Response:
        obs = OBS_COLLECTOR._last_observation
        if obs is None:
            obs = OBS_COLLECTOR.get_observation()

        def serialize_obs(val: Any) -> Any:
            if isinstance(val, dict):
                return {
                    (k.name if hasattr(k, "name") else str(k)): serialize_obs(v)
                    for k, v in val.items()
                }
            elif isinstance(val, (list, tuple)):
                return [serialize_obs(x) for x in val]
            elif hasattr(val, "name"):
                return val.name
            else:
                return val

        return web.json_response(serialize_obs(obs))

    async def handle_api_report(self, request: web.Request) -> web.Response:
        agent_metrics = getattr(self.publisher, "agent_metrics", None)
        if agent_metrics is None:
            return web.Response(
                text="No report data available yet.", content_type="text/plain"
            )
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            print_benchmark_report(agent_metrics)
        return web.Response(text=buf.getvalue(), content_type="text/plain")

    async def handle_index(self, request: web.Request) -> web.Response:
        lines = []
        # Main Title Header
        lines.append(f"{C_CYAN}{'=' * 80}{C_RESET}")
        lines.append(
            f"{C_WHITE}                    DELTA-HPC BENCHMARK RUNTIME MONITOR{C_RESET}"
        )
        lines.append(f"{C_CYAN}{'=' * 80}{C_RESET}")

        # Elapsed time and budget indicator
        elapsed_s = time.time() - self.start_time
        elapsed_h, remainder = divmod(int(elapsed_s), 3600)
        elapsed_m, elapsed_sec = divmod(remainder, 60)
        elapsed_str = f"{elapsed_h:02d}:{elapsed_m:02d}:{elapsed_sec:02d}"

        budget = float(OBS_COLLECTOR._current_budget)
        max_budget = TRAINING_CONFIG.reconfig_budget
        budget_color = C_RED if budget < (max_budget * 0.3) else C_GREEN
        completed = getattr(self.publisher, "completed_requests", 0)
        total = getattr(self.publisher, "total_requests", 0)
        pct = (completed / total * 100) if total > 0 else 0.0

        errors = sum(m.error_count for m in self.publisher.agent_metrics.values())
        error_color = C_RED if errors > 0 else C_GREEN
        lines.append(
            f"[Elapsed Time]: {elapsed_str} | [RL Action Budget]: {budget_color}{budget:.1f}s{C_RESET} / {max_budget:.1f}s | [Progress]: {completed}/{total} req ({pct:.1f}%) | [Errors]: {error_color}{errors}{C_RESET}"
        )
        lines.append("")

        # 1. GPU MIG Topology
        lines.append(f"{C_BLUE}{'-' * 80}{C_RESET}")
        lines.append(f"{C_BOLD}{C_WHITE}1. GPU MIG TOPOLOGY{C_RESET}")
        lines.append(f"{C_BLUE}{'-' * 80}{C_RESET}")
        for gpu_idx, gpu_state in SYSTEM_STATE.gpus.items():
            sim_str = (
                f" [{C_YELLOW}SIMULATED BACKUP{C_RESET}]"
                if gpu_state.is_simulated
                else f" [{C_GREEN}PHYSICAL MIG{C_RESET}]"
            )
            lines.append(f"{C_BOLD}GPU {gpu_idx}{C_RESET}{sim_str}:")
            if not gpu_state.slots:
                lines.append(f"  {C_GRAY}(No active partitions){C_RESET}")
            for slot in gpu_state.slots:
                owner = slot.agent_id.name if slot.agent_id else "FREE"
                owner_color = (
                    C_GREEN
                    if owner == "CODING"
                    else (C_PURPLE if owner == "RAG" else C_GRAY)
                )
                port_str = (
                    f"Port: {slot.port}" if slot.port is not None else "Port: SIM"
                )

                if slot.is_draining:
                    status = "DRAINING"
                    status_color = C_YELLOW
                elif not slot.is_ready:
                    status = "BOOTING"
                    status_color = C_YELLOW
                else:
                    status = "READY"
                    status_color = C_GREEN

                lines.append(
                    f"  Slice {slot.profile_placement.start_slice:<2} "
                    f"[{C_CYAN}{slot.profile_placement.profile.profile_type.name:<12}{C_RESET}] "
                    f"({owner_color}{owner:<6}{C_RESET}) | "
                    f"{C_BOLD}{port_str:<10}{C_RESET} | "
                    f"Status: {status_color}{status:<8}{C_RESET}"
                )
            lines.append("")

        # 2. Agent Engine Workloads
        lines.append(f"{C_BLUE}{'-' * 80}{C_RESET}")
        lines.append(f"{C_BOLD}{C_WHITE}2. AGENT ENGINE WORKLOADS{C_RESET}")
        lines.append(f"{C_BLUE}{'-' * 80}{C_RESET}")
        for agent_id in [m.AgentId.CODING, m.AgentId.RAG]:
            agent_color = C_GREEN if agent_id == m.AgentId.CODING else C_PURPLE
            lines.append(f"{C_BOLD}{agent_color}{agent_id.name} AGENT:{C_RESET}")

            engines_found = False
            for gpu_idx, gpu_state in SYSTEM_STATE.gpus.items():
                for slot in gpu_state.slots:
                    if slot.agent_id == agent_id:
                        engines_found = True

                        running = 0
                        waiting = 0
                        if not gpu_state.is_simulated:
                            # Attempt to use live per-slot metrics cache for accurate dashboard representation
                            live = self.get_live_slot_metrics(slot.mig_uuid)
                            if live is not None and slot.is_ready:
                                running = live["running"]
                                waiting = live["waiting"]
                            else:
                                running = 0
                                waiting = 0
                        else:
                            mig_uuid = slot.mig_uuid
                            running = len(
                                self.publisher.vllm_manager._active_reqs.get(
                                    mig_uuid, {}
                                )
                            )
                            waiting = self.publisher.vllm_manager.get_sim_waiting(
                                mig_uuid
                            )

                        type_str = (
                            f"GPU {slot.gpu_idx}, Slice {slot.profile_placement.start_slice:<2}"
                            if not gpu_state.is_simulated
                            else f"GPU {slot.gpu_idx}, Slice {slot.profile_placement.start_slice:<2} [SIM]"
                        )

                        run_color = C_GREEN if running > 0 else C_RESET
                        wait_color = C_RED if waiting > 0 else C_RESET

                        lines.append(
                            f"  - {C_CYAN}{slot.profile_placement.profile.profile_type.name:<12}{C_RESET} "
                            f"({type_str:<22}) | "
                            f"Concurrency: {run_color}{int(running):>3d}{C_RESET} | "
                            f"Queue: {wait_color}{int(waiting):>3d}{C_RESET}"
                        )
            if not engines_found:
                lines.append(f"  {C_GRAY}(No engines currently owned){C_RESET}")
            lines.append("")

        # 3. Reconfiguration History
        lines.append(f"{C_BLUE}{'-' * 80}{C_RESET}")
        lines.append(f"{C_BOLD}{C_WHITE}3. RECONFIGURATION HISTORY (LAST 10){C_RESET}")
        lines.append(f"{C_BLUE}{'-' * 80}{C_RESET}")
        if not OBS_COLLECTOR.reconfig_history:
            lines.append(f"  {C_GRAY}(No actions executed in this session){C_RESET}")
        else:
            for log in reversed(OBS_COLLECTOR.reconfig_history):
                time_str = time.strftime("%H:%M:%S", time.localtime(log["timestamp"]))
                action_name = log["action"]
                act_color = (
                    C_GREEN
                    if action_name == "SPLIT"
                    else (C_RED if action_name == "MERGE" else C_YELLOW)
                )
                lines.append(
                    f"[{C_GRAY}{time_str}{C_RESET}] "
                    f"Action: {act_color}{action_name:<8}{C_RESET} | "
                    f"Cost: {C_RED}{log['cost']:>5.1f}s{C_RESET}"
                )
                lines.append(f"  Details: {C_WHITE}{log['details']}{C_RESET}")
        lines.append(f"{C_CYAN}{'=' * 80}{C_RESET}")
        lines.append("")

        return web.Response(text="\n".join(lines), content_type="text/plain")
