import asyncio
import argparse

from src.deploy.req_pub import ReqPublisher
from src.deploy.vllm import VLLMManager
from src.share.request_loader import RequestLoader
from src.bench.config import BENCH_CONFIG
from src.bench.models import Workload
from src.training.config import TRAINING_CONFIG


async def main():
    parser = argparse.ArgumentParser(description="Run Deploy Benchmark")
    parser.add_argument(
        "--duration", type=float, default=60.0, help="Duration of benchmark in seconds"
    )
    args = parser.parse_args()

    vllm_manager = VLLMManager()

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
    await publisher.run_benchmark(args.duration)


if __name__ == "__main__":
    asyncio.run(main())
