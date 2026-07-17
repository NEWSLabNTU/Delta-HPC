import contextlib
import yaml
from pathlib import Path
import random
from typing import Dict, Iterator, List, Tuple, Type

from src.simulation.simulator import SimulatorImpl
from src.simulation.agent import AgentImpl
from src.simulation.engine import LLMEngineImpl
import src.share.models as m
from src.simulation.request import RequestImpl
import src.simulation.utils as utils
import src.simulation.config as config
from src.share.mig_matrix import STATE_DEFINITIONS


def _build_agents_and_engines(
    agent_id: m.AgentId, hw_mig: m.MIGProfileBase, gpu_id: int
) -> Tuple[Dict[m.AgentId, m.Agent], Dict[str, m.LLMEngine]]:
    """Constructs the agents/engines needed for a single-engine mini-simulation
    that isolates `hw_mig` on `gpu_id`, padding out the rest of the hardware
    state (co-located slices) with dummy-owned engines so the GPU state is
    physically valid."""
    # Find a valid state in STATE_DEFINITIONS that contains the target profile
    target_sid = -1
    for sid, defs in STATE_DEFINITIONS.items():
        if hw_mig.profile_type in defs:
            target_sid = sid
            break
    assert target_sid != -1, f"No state found containing {hw_mig.profile_type}"

    target_profiles = STATE_DEFINITIONS[target_sid]

    agents: Dict[m.AgentId, m.Agent] = {aid: AgentImpl(aid) for aid in m.AgentId}

    # Identify which index in the state will be our "measured" engine.
    # We use the first matching profile index.
    main_idx = -1
    for i, p in enumerate(target_profiles):
        if p == hw_mig.profile_type:
            main_idx = i
            break

    engines: Dict[str, m.LLMEngine] = {}

    for i, p in enumerate(target_profiles):
        if i == main_idx:
            curr_hw_mig = hw_mig
            owner_id = agent_id
        else:
            # Find any hardware profile for this logical profile
            curr_hw_mig = None
            for hp in config.GPU_MIG_PROFILE[gpu_id]:
                if hp.profile_type == p:
                    curr_hw_mig = hp
                    break
            assert curr_hw_mig is not None
            # Assign dummy owner (rotate agents to ensure they exist)
            owner_id = (
                m.AgentId.RAG if agent_id == m.AgentId.CODING else m.AgentId.CODING
            )

        eid = utils.generate_engine_id(gpu_id, curr_hw_mig.string)

        eng = LLMEngineImpl.create(
            gpu=gpu_id,
            engine_id=eid,
            owner=agents[owner_id],
            mig_profile=curr_hw_mig,
            current_time=0.0,
            mig_index=i,
            is_permanent=True,
        )
        engines[eid] = eng
        agents[owner_id].add_engine(eng)

    return agents, engines


@contextlib.contextmanager
def _isolated_gpu_cluster(gpu_id: int) -> Iterator[None]:
    """Temporarily overrides SIM_CONFIG.cluster to only include the target GPU,
    preventing SimulatorImpl from trying to identify states for other GPUs."""
    orig_cluster = utils.SIM_CONFIG.cluster
    utils.SIM_CONFIG.cluster = {gpu_id: orig_cluster[gpu_id]}
    try:
        yield
    finally:
        utils.SIM_CONFIG.cluster = orig_cluster


def build_rate_requests(
    agent_id: m.AgentId,
    hw_mig: m.MIGProfileBase,
    sampled_req_ids: List[str],
    rate: float,
    gpu_id: int,
) -> List[RequestImpl]:
    """Builds a list of requests with fixed-rate arrival times, sourcing
    prompt/completion tokens from the dataset for `agent_id` served by
    `hw_mig` on `gpu_id`."""
    model_name = utils.SIM_CONFIG.get_model(agent_id, hw_mig, gpu_id=gpu_id)
    req_map = utils.TOKENS_MAP[agent_id][model_name]

    requests = []
    for idx, rid in enumerate(sampled_req_ids):
        prompt_tokens, completion_tokens = req_map[rid]
        req = RequestImpl(
            id=f"{rid}_{agent_id.value}_{idx}",
            agent_id=agent_id,
            prompt_tokens=prompt_tokens,
            original_id=rid,
        )
        req.completion_tokens = completion_tokens
        req.arrival_time = idx / rate
        requests.append(req)
    return requests


def run_single_engine_sim(
    agent_id: m.AgentId,
    hw_mig: m.MIGProfileBase,
    gpu_id: int,
    requests: List[RequestImpl],
) -> List[RequestImpl]:
    """Runs a single-engine mini-simulation with the given pre-built requests
    to completion, returning the same requests (mutated in place with timing
    fields such as start_time/first_token_time/finish_time)."""
    agents, engines = _build_agents_and_engines(agent_id, hw_mig, gpu_id)

    with _isolated_gpu_cluster(gpu_id):
        sim = SimulatorImpl(agents=agents, engines=engines, no_log=True)
        sim.add_arrival_events(requests)
        for e in sim.engines.values():
            e.activate(sim.current_time)

        while sim.run():
            pass

    return requests


def check_rate(
    agent_id: m.AgentId,
    hw_mig: m.MIGProfileBase,
    sampled_req_ids: List[str],
    rate: float,
    gpu_id: int,
) -> bool:
    requests = build_rate_requests(agent_id, hw_mig, sampled_req_ids, rate, gpu_id)
    run_single_engine_sim(agent_id, hw_mig, gpu_id, requests)

    # Check stability by measuring if the wait time grows continuously over the test
    # This subtracts the base prefill time (which varies wildly between MIG profiles)
    ttfts = [
        r.start_time - r.arrival_time for r in requests if r.start_time is not None
    ]

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


def measure_max_service_rate():
    num_requests = 1000

    # 1. Prepare requests
    agent_sampled_reqs: Dict[m.AgentId, List[str]] = {}
    for agent_id in m.AgentId:
        first_model = next(iter(utils.TOKENS_MAP[agent_id]))
        all_rids = list(utils.TOKENS_MAP[agent_id][first_model].keys())
        agent_sampled_reqs[agent_id] = random.choices(all_rids, k=num_requests)

    # 2. Identify unique GPU models in the cluster
    unique_gpu_types: Dict[str, Tuple[int, Type[m.MIGProfileBase]]] = {}
    for gpu_id, hw_prof_cls in config.GPU_MIG_PROFILE.items():
        # Use first member to get the gpu_model name
        gpu_model_name = next(iter(hw_prof_cls)).gpu_model
        if gpu_model_name not in unique_gpu_types:
            unique_gpu_types[gpu_model_name] = (gpu_id, hw_prof_cls)

    print(
        f"{'GPU Model':<15} | {'Agent':<15} | {'MIG Profile':<15} | {'Max Rate (req/s)':<15}"
    )
    print("-" * 65)

    results_dict = {}

    for gpu_model, (gpu_id, hw_prof_proto) in unique_gpu_types.items():
        results_dict[gpu_model] = {aid.value: {} for aid in m.AgentId}

        # Iterate over all hardware profiles for this GPU model
        for hw_mig in hw_prof_proto:
            for agent_id in m.AgentId:
                # Check if this agent is configured for this GPU
                if agent_id.value not in config.GPU_AGENTS_CONFIG[gpu_id]:
                    continue

                # Check if this MIG profile is configured for this agent on this GPU
                if (
                    hw_mig.string
                    not in config.GPU_AGENTS_CONFIG[gpu_id][agent_id.value]["mig"]
                ):
                    continue

                sampled_req_ids = agent_sampled_reqs[agent_id]

                low = 0.1
                high = 20.0
                best_rate = 0.0

                while high - low > 0.1:
                    mid = (low + high) / 2
                    is_stable = check_rate(
                        agent_id, hw_mig, sampled_req_ids, mid, gpu_id
                    )
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

    # 3. Write to configs/bench_config.yaml
    config_path = Path("configs/bench_config.yaml")
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

    print("\nResults successfully saved to configs/bench_config.yaml")


if __name__ == "__main__":
    measure_max_service_rate()
