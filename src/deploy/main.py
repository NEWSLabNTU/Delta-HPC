import asyncio
import argparse
import logging

from src.deploy.req_pub import ReqPublisher
from src.deploy.vllm import VLLMManager
from src.share.request_loader import RequestLoader
from src.deploy.cluster import DeployGPUSetup
from src.deploy.system import SYSTEM_STATE
from src.bench.config import BENCH_CONFIG
from src.bench.models import Workload
from src.training.config import TRAINING_CONFIG
from src.share.logging_utils import setup_logging

logger = logging.getLogger(__name__)


async def main():
    parser = argparse.ArgumentParser(description="Run Deploy Benchmark")
    parser.add_argument(
        "--duration", type=int, default=100, help="Duration of benchmark in seconds"
    )
    args = parser.parse_args()

    setup_logging()
    setup = DeployGPUSetup()
    vllm_manager = None
    publisher = None

    try:
        # 1. Configure physical GPUs (e.g. 0, 1) with random setups
        logger.info("Initializing physical GPU configurations (random setup)...")
        setup.apply_random()

        # Register simulated permanent engines (e.g. GPU 2) from deployment.yaml
        logger.info("Registering simulated permanent engines...")
        setup.register_simulated_gpus()

        # 2. Start vLLM Servers
        vllm_manager = VLLMManager()
        logger.info("Starting vLLM servers on all active slots...")
        for gpu_idx, gpu_state in SYSTEM_STATE.gpus.items():
            vllm_manager.start_all(gpu_state)
        for gpu_idx, gpu_state in SYSTEM_STATE.gpus.items():
            # Wait for actual containers to be healthy
            for slot in gpu_state.slots:
                vllm_manager.wait_until_ready(slot)

        # 3. Run Benchmark
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
        await publisher.start_sending(args.duration)

    except (asyncio.CancelledError, KeyboardInterrupt):
        logger.info("Benchmark interrupted.")
    except Exception as e:
        logger.exception(f"Fatal error during benchmark: {e}")
    finally:
        # 4. Clean up in-flight requests and print metrics
        if publisher:
            await publisher.cleanup()

        # 5. Teardown
        if vllm_manager:
            logger.info("Shutting down vLLM servers...")
            for gpu_state in SYSTEM_STATE.gpus.values():
                vllm_manager.stop_all(gpu_state, graceful=False)

        logger.info("Cleaning up MIG instances...")
        setup.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
