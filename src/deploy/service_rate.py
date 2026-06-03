import yaml
import asyncio
from pathlib import Path
from typing import Dict, List

import src.share.models as m
import src.simulation.utils as utils
import src.simulation.config as config
from src.deploy.cluster import DeployGPUSetup
from src.deploy.vllm import VLLMManager
from src.deploy.system import SYSTEM_STATE, register_gpu
from src.deploy.models import ProfilePlacement, MIGSlotState, GPUState
from src.share.mig_matrix import STATE_DEFINITIONS, SLICE_MAPPING
from src.share.request_loader import RequestLoader


class EngineCrashedError(Exception):
    pass


async def check_rate(
    vllm_manager: VLLMManager,
    agent_id: m.AgentId,
    slot: MIGSlotState,
    sampled_reqs: List[m.Request],
    rate: float,
) -> bool:
    model_name = vllm_manager.model_for_slot(slot)
    req_map = utils.TOKENS_MAP[agent_id][model_name]
    is_simulated = SYSTEM_STATE.gpus[slot.gpu_idx].is_simulated
    assert not is_simulated

    ttfts = []

    async def worker(idx, req):
        arrival_time = idx / rate
        await asyncio.sleep(arrival_time)

        rid = req.original_id
        _, completion_tokens = req_map[rid]

        prompt = req.prompt
        if prompt is None:
            prompt = "Benchmark test request."

        messages = [{"role": "user", "content": prompt}]
        try:
            res = await vllm_manager.send_request(
                slot, messages, max_tokens=completion_tokens, data_id=rid
            )
            ttfts.append(res["ttft"])
        except Exception:
            import requests

            def probe():
                try:
                    r = requests.get(
                        f"http://localhost:{slot.port}/health", timeout=2.0
                    )
                    return r.status_code == 200
                except Exception:
                    return False

            is_healthy = await asyncio.to_thread(probe)

            if not is_healthy:
                print("FATAL: vLLM engine crashed (health check failed). Restarting...")
                raise EngineCrashedError()
            pass

    tasks = [asyncio.create_task(worker(i, req)) for i, req in enumerate(sampled_reqs)]
    try:
        await asyncio.gather(*tasks)
    except EngineCrashedError:
        for t in tasks:
            t.cancel()
        raise

    # Check stability by measuring if the wait time grows continuously over the test
    # This subtracts the base prefill time (which varies wildly between MIG profiles)
    first_10_percent_idx = max(1, int(len(ttfts) * 0.1))
    last_10_percent_idx = max(1, int(len(ttfts) * 0.9))

    first_ttfts = ttfts[:first_10_percent_idx]
    last_ttfts = ttfts[last_10_percent_idx:]

    if not first_ttfts or not last_ttfts:
        return False

    avg_ttft_first = sum(first_ttfts) / len(first_ttfts)
    avg_ttft_last = sum(last_ttfts) / len(last_ttfts)

    # If the TTFT grows by more than 1.0 seconds, the queue is backing up
    queue_growth = avg_ttft_last - avg_ttft_first
    print(
        f"Rate: {rate:.2f} | Base TTFT: {avg_ttft_first:.2f}s | Final TTFT: {avg_ttft_last:.2f}s | Growth: {queue_growth:.2f}s"
    )

    return queue_growth < 1.0


async def _run_benchmark():
    from src.share.logging_utils import setup_logging

    setup_logging()

    # Reload SIM_CONFIG and TOKENS_MAP with hardware detection explicitly enabled
    from src.simulation.config import SimulationConfig

    utils.SIM_CONFIG = SimulationConfig.load(
        Path("configs/simulation_config.yaml"), use_hardware_detection=True
    )
    utils.TOKENS_MAP = utils.init_tokens_map(Path("."), utils.SIM_CONFIG)

    num_requests = 1000

    loader = RequestLoader(
        num_steps=10,
        get_rate_range=lambda p, a: (150.0, 150.0),
        get_duration_range=lambda p: (100.0, 100.0),
        load_actual_prompt=True,
        dataset_paths=utils.SIM_CONFIG.datasets,
    )

    print("Generating requests...")
    agent_sampled_reqs: Dict[m.AgentId, List[m.Request]] = {}
    for agent_id in m.AgentId:
        reqs = loader.generate_requests(agent_id)
        agent_sampled_reqs[agent_id] = reqs[:num_requests]

    setup = DeployGPUSetup()

    unique_gpu_types: Dict[str, int] = {}
    for gpu_idx, info in setup.gpu_info.items():
        if info.model_name not in unique_gpu_types:
            unique_gpu_types[info.model_name] = gpu_idx

    results_dict = {}

    print(
        f"{'GPU Model':<15} | {'Agent':<15} | {'MIG Profile':<15} | {'Max Rate (req/s)':<15}"
    )
    print("-" * 65)

    for gpu_model, gpu_idx in unique_gpu_types.items():
        results_dict[gpu_model] = {aid.value: {} for aid in m.AgentId}
        info = setup.gpu_info[gpu_idx]
        mig_profile_cls = info.mig_profile_cls

        for hw_mig in mig_profile_cls:
            for agent_id in m.AgentId:
                if agent_id.value not in config.GPU_AGENTS_CONFIG.get(gpu_idx, {}):
                    continue
                if hw_mig.string not in config.GPU_AGENTS_CONFIG[gpu_idx][
                    agent_id.value
                ].get("mig", {}):
                    continue

                target_sid = -1
                target_slot_idx = -1
                for sid, defs in STATE_DEFINITIONS.items():
                    if hw_mig.profile_type in defs:
                        target_sid = sid
                        target_slot_idx = defs.index(hw_mig.profile_type)
                        break
                assert target_sid != -1, (
                    f"No state found containing {hw_mig.profile_type}"
                )

                slice_groups = SLICE_MAPPING[target_sid]
                placements = []
                for logical_prof, slice_group in zip(
                    STATE_DEFINITIONS[target_sid], slice_groups
                ):
                    hp = next(
                        p for p in mig_profile_cls if p.profile_type == logical_prof
                    )
                    placements.append(ProfilePlacement(hp, slice_group[0]))

                setup.mig_ctrl.apply_full_configuration(gpu_idx, placements)
                uuid_map = dict(setup.mig_ctrl.list_mig_device_uuids(gpu_idx))

                bench_placement = placements[target_slot_idx]
                slot = MIGSlotState(
                    gpu_idx=gpu_idx,
                    profile_placement=bench_placement,
                    mig_uuid=uuid_map[bench_placement.start_slice],
                    agent_id=agent_id,
                )

                gpu_state = GPUState(
                    gpu_idx=gpu_idx,
                    model_name=gpu_model,
                    mig_profile_cls=mig_profile_cls,
                    slots=[slot],
                )
                register_gpu(gpu_state)

                vllm_manager = VLLMManager()

                try:
                    vllm_manager.start(slot)
                    vllm_manager.wait_until_ready(slot)

                    sampled_reqs = agent_sampled_reqs[agent_id]
                    low = 0.1
                    high = 10.0
                    best_rate = 0.0

                    while high - low > 0.1:
                        mid = (low + high) / 2
                        try:
                            is_stable = await check_rate(
                                vllm_manager, agent_id, slot, sampled_reqs, mid
                            )
                        except EngineCrashedError:
                            print(
                                f"Engine crashed at rate {mid:.2f}. Restarting engine..."
                            )
                            vllm_manager.stop(slot, graceful=False)
                            vllm_manager.start(slot)
                            vllm_manager.wait_until_ready(slot)
                            is_stable = False

                        if is_stable:
                            best_rate = mid
                            low = mid
                        else:
                            high = mid

                    results_dict[gpu_model][agent_id.value][hw_mig.string] = float(
                        f"{best_rate:.2f}"
                    )
                    print(
                        f"{gpu_model:<15} | {agent_id.value:<15} | {hw_mig.string:<15} | {best_rate:>15.2f}"
                    )

                finally:
                    vllm_manager.stop(slot, graceful=False)

    setup.cleanup()

    config_path = Path("configs/deployment.yaml")
    with open(config_path, "r") as f:
        config_data = yaml.safe_load(f)

    service_rates = config_data.setdefault("heuristic", {}).setdefault(
        "service_rates", {}
    )

    for gpu_model, agents_dict in results_dict.items():
        model_rates = service_rates.setdefault(gpu_model, {})
        for agent_id, mig_dict in agents_dict.items():
            model_rates.setdefault(agent_id, {}).update(mig_dict)

    with open(config_path, "w") as f:
        yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)

    print("\nResults successfully saved to configs/deployment.yaml")


def measure_max_service_rate():
    asyncio.run(_run_benchmark())


if __name__ == "__main__":
    measure_max_service_rate()
