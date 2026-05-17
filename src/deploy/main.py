import asyncio
import argparse
import logging
from pathlib import Path

from src.deploy.req_pub import ReqPublisher
from src.deploy.vllm import VLLMManager
from src.deploy.rl import RLAgent
from src.deploy.act_controller import ActionController
from src.share.request_loader import RequestLoader
from src.deploy.cluster import DeployGPUSetup
from src.deploy.system import SYSTEM_STATE
from src.bench.config import BENCH_CONFIG
from src.bench.models import Workload
from src.training.config import TRAINING_CONFIG
from src.share.logging_utils import setup_logging
from src.deploy.dashboard import DashboardServer
from src.deploy.config import DEPLOY_CONFIG
import src.simulation.utils as sim_utils
from src.simulation.config import SimulationConfig

logger = logging.getLogger(__name__)


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run RL-controlled Deploy Benchmark")
    parser.add_argument(
        "--ckpt",
        type=Path,
        help="Path to the MaskablePPO checkpoint (.zip)",
    )
    parser.add_argument(
        "--duration", type=int, default=100, help="Duration of benchmark in seconds"
    )
    args = parser.parse_args()

    ckpt_path: Path = args.ckpt
    vecnorm_path = ckpt_path.with_name(f"{ckpt_path.stem}_vecnormalize.pkl")

    setup_logging()
    setup = DeployGPUSetup()
    vllm_manager = None
    publisher = None
    dashboard = None

    try:
        # 1. Load RL policy first (CPU-only — no GPU memory consumed)
        rl_agent = RLAgent(
            ckpt_path=ckpt_path,
            vecnorm_path=vecnorm_path if vecnorm_path.exists() else None,
        )

        # 2. Configure physical GPUs with random initial MIG layout
        logger.info("Initializing physical GPU configurations (random setup)...")
        setup.apply_random()

        # Register simulated permanent engines from deployment.yaml
        logger.info("Registering simulated permanent engines...")
        setup.register_simulated_gpus()

        # 2.5 Reload SIM_CONFIG and TOKENS_MAP with hardware detection explicitly enabled
        logger.info("Reloading SIM_CONFIG with physical hardware detection...")

        sim_utils.SIM_CONFIG = SimulationConfig.load(
            Path("configs/simulation_config.yaml"), use_hardware_detection=True
        )
        sim_utils.TOKENS_MAP = sim_utils.init_tokens_map(
            Path("."), sim_utils.SIM_CONFIG
        )

        # 3. Start vLLM servers
        vllm_manager = VLLMManager()
        logger.info("Starting vLLM servers on all active slots...")
        for gpu_state in SYSTEM_STATE.gpus.values():
            vllm_manager.start_all(gpu_state)
        for gpu_state in SYSTEM_STATE.gpus.values():
            for slot in gpu_state.slots:
                vllm_manager.wait_until_ready(slot)

        # 4. Build RL action controller (MIGController is created internally per-action)
        act_ctrl = ActionController(vllm_mgr=vllm_manager)

        # 5. Build request publisher
        num_steps = int(args.duration / TRAINING_CONFIG.action_interval) + 10
        loader = RequestLoader(
            num_steps=num_steps,
            get_rate_range=lambda p, a: BENCH_CONFIG.get_rate_range(Workload(p), a),
            get_duration_range=lambda p: BENCH_CONFIG.get_duration_range(Workload(p)),
            seed=BENCH_CONFIG.seed,
            track_history=False,
            load_actual_prompt=True,
        )

        publisher = ReqPublisher(vllm_manager, loader)

        # 5.5 Start the Web Dashboard Server
        dashboard = DashboardServer(
            publisher=publisher,
            host=DEPLOY_CONFIG.dashboard.host,
            port=DEPLOY_CONFIG.dashboard.port,
        )
        await dashboard.start()

        # 6. Run request dispatch and RL control loop concurrently
        sending_future = publisher.start_sending(args.duration)
        rl_task = asyncio.create_task(rl_agent.run_loop(act_ctrl, args.duration))

        await asyncio.gather(sending_future, rl_task, return_exceptions=True)

    except (asyncio.CancelledError, KeyboardInterrupt):
        logger.info("Benchmark interrupted.")
    except Exception as e:
        logger.exception(f"Fatal error during benchmark: {e}")
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
