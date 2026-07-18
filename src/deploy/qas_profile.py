"""
Real-hardware counterpart of src/bench/qas_profile.py: builds the
TTFT-vs-arrival-rate profiling table QAS (Quality-Aware Scheduling, see
src/bench/qas.py) uses to predict per-engine TTFT under a candidate MIG
allocation, but measures it by sending real requests to real vLLM engines
on real MIG slices, instead of sweeping the single-engine mini-simulation
src/bench/qas_profile.py uses.

Mirrors src/deploy/service_rate.py's on-hardware harness (same GPU setup,
request sampling, and instability criterion) and reuses the max stable rate
`mu` it already measured and wrote into configs/deployment.yaml's
heuristic.service_rates -- this module only needs to run *after*
`python -m src.deploy.service_rate` has populated that table, since it
sweeps lambda as fractions of that already-known mu (same LAMBDA_FRACTIONS
grid as the simulated version, for comparability).

Response quality (Q_f) is NOT profiled or stored here -- it stays a fully
online computation at decision time in src/bench/qas.py, reusing the
existing TRAINING_CONFIG.qf(...) hyperparameter table.

Run standalone (does not start a full deployment):
    python -m src.deploy.qas_profile
"""

import yaml
import asyncio
from pathlib import Path
from typing import Dict, List, Optional

import src.share.models as m
import src.simulation.utils as utils
from src.deploy.cluster import DeployGPUSetup
from src.deploy.vllm import VLLMManager
from src.deploy.system import SYSTEM_STATE, register_gpu
from src.deploy.models import ProfilePlacement, MIGSlotState, GPUState
from src.share.mig_matrix import STATE_DEFINITIONS, SLICE_MAPPING
from src.share.request_loader import RequestLoader
from src.deploy.service_rate import EngineCrashedError

LAMBDA_FRACTIONS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
NUM_REQUESTS = 500


async def measure_ttft_at_rate(
    vllm_manager: VLLMManager,
    agent_id: m.AgentId,
    slot: MIGSlotState,
    sampled_reqs: List[m.Request],
    rate: float,
) -> Optional[float]:
    """Sends `sampled_reqs` to `slot` at Poisson-spaced arrival rate `rate`
    (same dispatch shape as service_rate.py's check_rate) and returns the
    steady-state average TTFT over the last 10% of completed requests, or
    None if TTFT is still growing (this rate exceeds the slice's stable
    capacity, matching check_rate's own instability criterion)."""
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

    first_10_percent_idx = max(1, int(len(ttfts) * 0.1))
    last_10_percent_idx = max(1, int(len(ttfts) * 0.9))

    first_ttfts = ttfts[:first_10_percent_idx]
    last_ttfts = ttfts[last_10_percent_idx:]

    if not first_ttfts or not last_ttfts:
        return None

    avg_ttft_first = sum(first_ttfts) / len(first_ttfts)
    avg_ttft_last = sum(last_ttfts) / len(last_ttfts)

    queue_growth = avg_ttft_last - avg_ttft_first
    print(
        f"Rate: {rate:.2f} | Base TTFT: {avg_ttft_first:.2f}s | "
        f"Final TTFT: {avg_ttft_last:.2f}s | Growth: {queue_growth:.2f}s"
    )

    if queue_growth > 1.0:
        return None
    return avg_ttft_last


async def build_curve(
    vllm_manager: VLLMManager,
    agent_id: m.AgentId,
    slot: MIGSlotState,
    mu: float,
    sampled_reqs: List[m.Request],
) -> List[List[float]]:
    """Sweeps lambda as fractions of `mu`, returning the sorted (lambda,
    TTFT) points that reached steady state. On an engine crash, restarts the
    engine and drops that rate (matching service_rate.py's own recovery)."""
    points: List[List[float]] = []
    for frac in LAMBDA_FRACTIONS:
        rate = frac * mu
        try:
            ttft = await measure_ttft_at_rate(
                vllm_manager, agent_id, slot, sampled_reqs, rate
            )
        except EngineCrashedError:
            print(f"Engine crashed at rate {rate:.2f}. Restarting engine...")
            vllm_manager.stop(slot, graceful=False)
            vllm_manager.start(slot)
            vllm_manager.wait_until_ready(slot)
            ttft = None

        if ttft is not None:
            points.append([rate, ttft])

    points.sort(key=lambda p: p[0])
    return points


async def _run_benchmark():
    from src.share.logging_utils import setup_logging

    setup_logging()

    from src.simulation.config import SimulationConfig

    utils.SIM_CONFIG = SimulationConfig.load(
        Path("configs/simulation_config.yaml"), use_hardware_detection=True
    )
    utils.TOKENS_MAP = utils.init_tokens_map(Path("."), utils.SIM_CONFIG)

    config_path = Path("configs/deployment.yaml")
    with open(config_path, "r") as f:
        config_data = yaml.safe_load(f)

    # Reuse the (gpu_model, agent, mig_profile) -> mu table already populated
    # by `python -m src.deploy.service_rate`, so both tables share keys.
    service_rates = config_data.get("heuristic", {}).get("service_rates", {})
    if not service_rates:
        raise RuntimeError(
            "configs/deployment.yaml has no heuristic.service_rates -- run "
            "`python -m src.deploy.service_rate` first."
        )

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
        agent_sampled_reqs[agent_id] = reqs[:NUM_REQUESTS]

    setup = DeployGPUSetup()

    gpu_model_to_idx: Dict[str, int] = {}
    for gpu_idx, info in setup.gpu_info.items():
        if info.model_name not in gpu_model_to_idx:
            gpu_model_to_idx[info.model_name] = gpu_idx

    profile_table: Dict[str, Dict[str, Dict[str, List[List[float]]]]] = {}

    print(
        f"{'GPU Model':<15} | {'Agent':<15} | {'MIG Profile':<15} | "
        f"{'μ (measured)':<15} | {'#λ points':<10}"
    )
    print("-" * 75)

    for gpu_model, agents_dict in service_rates.items():
        if gpu_model not in gpu_model_to_idx:
            continue
        gpu_idx = gpu_model_to_idx[gpu_model]
        info = setup.gpu_info[gpu_idx]
        mig_profile_cls = info.mig_profile_cls
        model_table = profile_table.setdefault(gpu_model, {})

        for agent_value, mig_dict in agents_dict.items():
            try:
                agent_id = m.AgentId(agent_value)
            except ValueError:
                continue
            agent_table = model_table.setdefault(agent_value, {})
            sampled_reqs = agent_sampled_reqs[agent_id]

            for mig_str, mu in mig_dict.items():
                mu = float(mu)
                if mu <= 0:
                    continue

                hw_mig = next((p for p in mig_profile_cls if p.string == mig_str), None)
                if hw_mig is None:
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

                    curve = await build_curve(
                        vllm_manager, agent_id, slot, mu, sampled_reqs
                    )
                    agent_table[mig_str] = curve

                    print(
                        f"{gpu_model:<15} | {agent_value:<15} | {mig_str:<15} | "
                        f"{mu:>15.2f} | {len(curve):>10}"
                    )
                finally:
                    vllm_manager.stop(slot, graceful=False)

    setup.cleanup()

    qas_cfg = config_data.setdefault("qas", {})
    qas_cfg["profile_table"] = profile_table

    with open(config_path, "w") as f:
        yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)

    print("\nResults successfully saved to configs/deployment.yaml")


def measure_ttft_profile():
    asyncio.run(_run_benchmark())


if __name__ == "__main__":
    measure_ttft_profile()
