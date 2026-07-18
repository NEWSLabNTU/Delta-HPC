import os

# =========================================================================
# PREVENT PYTORCH CUDA CONTEXT INITIALIZATION IN PARENT PROCESS
# Hides all GPUs from the python process so PyTorch/SB3 doesn't open driver file nodes
# =========================================================================
os.environ["CUDA_VISIBLE_DEVICES"] = ""
# =========================================================================

import asyncio
import argparse
import logging
from pathlib import Path
from typing import Optional
import subprocess

from src.deploy.req_pub import ReqPublisher
from src.deploy.vllm import VLLMManager
from src.deploy.rl import RLAgent
from src.deploy.act_controller import ActionController
from src.share.request_loader import RequestLoader
from src.deploy.cluster import DeployGPUSetup
from src.deploy.system import SYSTEM_STATE
from src.bench.models import Workload
from src.training.config import TRAINING_CONFIG
from src.share.logging_utils import setup_logging
from src.deploy.base_agent import BasePolicyAgent
from src.share.mig_matrix import STATE_DEFINITIONS
from src.share.models import MIGProfile
from src.deploy.heuristic_agent import HeuristicAgent
from src.deploy.qas_agent import QASAgent
from src.deploy.dashboard import DashboardServer
from src.deploy.config import DEPLOY_CONFIG
import src.simulation.utils as sim_utils
from src.simulation.config import SimulationConfig

logger = logging.getLogger(__name__)


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run RL-controlled Deploy Benchmark")
    parser.add_argument(
        "--policy",
        type=str,
        required=True,
        help="Which policy to run: 'heuristic', 'qas', 'static-7g', 'static-2g', or a path to a MaskablePPO checkpoint (.zip).",
    )
    parser.add_argument(
        "--duration", type=int, default=100, help="Duration of benchmark in seconds"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level",
    )
    args = parser.parse_args()

    setup_logging(getattr(logging, args.log_level))
    setup = DeployGPUSetup(seed=DEPLOY_CONFIG.seed)
    vllm_manager = None
    publisher = None
    dashboard = None
    policy_agent: Optional[BasePolicyAgent] = None

    try:
        # 1. Initialize vLLM Manager and Action Controller
        vllm_manager = VLLMManager()
        act_ctrl = ActionController(vllm_mgr=vllm_manager)

        # 2. Configure physical GPUs & Instantiate policy agent
        match args.policy:
            case "heuristic":
                logger.info(
                    "Initializing physical GPU configurations (random setup)..."
                )
                setup.apply_random()
                policy_agent = HeuristicAgent(act_ctrl)
            case "qas":
                logger.info(
                    "Initializing physical GPU configurations (random setup)..."
                )
                setup.apply_random()
                policy_agent = QASAgent(act_ctrl)
            case "static-7g":
                target_combo = (MIGProfile.MIG_7G,)
                state_id = next(
                    sid
                    for sid, combo in STATE_DEFINITIONS.items()
                    if combo == target_combo
                )
                logger.info("Initializing physical GPU configurations (static 7g)...")
                setup.apply_fixed(state_id)
                policy_agent = None
            case "static-2g":
                target_combo = (
                    MIGProfile.MIG_2G,
                    MIGProfile.MIG_2G,
                    MIGProfile.MIG_2G,
                    MIGProfile.MIG_1G_LARGE,
                )
                state_id = next(
                    sid
                    for sid, combo in STATE_DEFINITIONS.items()
                    if combo == target_combo
                )
                logger.info("Initializing physical GPU configurations (static 2g)...")
                setup.apply_fixed(state_id)
                policy_agent = None
            case _:
                # Treat as an RL checkpoint path
                ckpt_path = Path(args.policy)
                if not ckpt_path.exists():
                    raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
                logger.info(
                    "Initializing physical GPU configurations (random setup)..."
                )
                setup.apply_random()
                vecnorm_path = ckpt_path.with_name(f"{ckpt_path.stem}_vecnormalize.pkl")
                policy_agent = RLAgent(
                    act_ctrl=act_ctrl,
                    ckpt_path=ckpt_path,
                    vecnorm_path=vecnorm_path if vecnorm_path.exists() else None,
                )

        # 3. Register simulated permanent engines from deployment.yaml
        logger.info("Registering simulated permanent engines...")
        setup.register_simulated_gpus()

        # 3.5 Reload SIM_CONFIG and TOKENS_MAP with hardware detection explicitly enabled
        logger.info("Reloading SIM_CONFIG with physical hardware detection...")

        sim_utils.SIM_CONFIG = SimulationConfig.load(
            Path("configs/simulation_config.yaml"), use_hardware_detection=True
        )
        sim_utils.TOKENS_MAP = sim_utils.init_tokens_map(
            Path("."), sim_utils.SIM_CONFIG
        )

        # 4. Start vLLM servers
        logger.info("Starting vLLM servers on all active slots...")
        for gpu_state in SYSTEM_STATE.gpus.values():
            vllm_manager.start_all(gpu_state)
        for gpu_state in SYSTEM_STATE.gpus.values():
            for slot in gpu_state.slots:
                vllm_manager.wait_until_ready(slot)

        # 5. Build request publisher
        num_steps = int(args.duration / TRAINING_CONFIG.action_interval) + 10
        loader = RequestLoader(
            num_steps=num_steps,
            get_rate_range=lambda p, a: DEPLOY_CONFIG.get_rate_range(Workload(p), a),
            get_duration_range=lambda p: DEPLOY_CONFIG.get_duration_range(Workload(p)),
            seed=DEPLOY_CONFIG.seed,
            track_history=True,
            load_actual_prompt=True,
            dataset_paths=sim_utils.SIM_CONFIG.datasets,
        )

        publisher = ReqPublisher(vllm_manager, loader)

        # 5.5 Start the Web Dashboard Server
        dashboard = DashboardServer(
            publisher=publisher,
            host=DEPLOY_CONFIG.dashboard.host,
            port=DEPLOY_CONFIG.dashboard.port,
        )
        publisher.dashboard = dashboard
        await dashboard.start()

        # 6. Run request dispatch and policy control loop concurrently
        sending_future = publisher.start_sending(args.duration)
        if policy_agent is not None:
            policy_task = asyncio.create_task(policy_agent.run_loop(args.duration))
            await asyncio.gather(sending_future, policy_task, return_exceptions=False)
        else:
            await sending_future

    except (asyncio.CancelledError, KeyboardInterrupt):
        logger.info("Benchmark interrupted.")
    except Exception as e:
        logger.exception(f"Fatal error during benchmark: {e}")
        try:
            logger.info(
                "Attempting to dump docker compose logs to 'fatal_error_docker.log'..."
            )
            with open("fatal_error_docker.log", "w") as log_file:
                for gpu_state in SYSTEM_STATE.gpus.values():
                    for slot in gpu_state.slots:
                        if slot.port is not None and slot.mig_uuid:
                            try:
                                model_id = vllm_manager.model_for_slot(slot)
                                logger.info(
                                    f"Dumping logs for GPU {gpu_state.gpu_idx} slice {slot.profile_placement.start_slice}..."
                                )
                                log_file.write(
                                    f"\n\n{'=' * 80}\nLogs for GPU {gpu_state.gpu_idx} slice {slot.profile_placement.start_slice} (MIG {slot.mig_uuid}, Port {slot.port})\n{'=' * 80}\n"
                                )
                                log_file.flush()
                                subprocess.run(
                                    [
                                        str(DEPLOY_CONFIG.vllm.script),
                                        slot.mig_uuid,
                                        model_id,
                                        str(slot.port),
                                        "logs",
                                    ],
                                    stdout=log_file,
                                    stderr=subprocess.STDOUT,
                                )
                            except Exception as e:
                                logger.error(f"Error dumping logs for slot: {e}")
            logger.info("Dumped docker compose logs to fatal_error_docker.log")
        except Exception as log_e:
            logger.error(f"Failed to dump docker logs: {log_e}")
    finally:
        # Stop the Web Dashboard Server
        if dashboard:
            await dashboard.stop()

        # 7. Clean up in-flight requests and print metrics
        if publisher:
            await publisher.cleanup()

        # 8. Teardown vLLM and MIG
        if vllm_manager:
            logger.info("Shutting down vLLM servers...")
            for gpu_state in SYSTEM_STATE.gpus.values():
                vllm_manager.stop_all(gpu_state, graceful=False)

        logger.info("Cleaning up MIG instances...")
        setup.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
