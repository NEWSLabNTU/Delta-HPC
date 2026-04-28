import yaml
from pathlib import Path
import random
from typing import Dict, List, Tuple, Type

from src.simulation.simulator import SimulatorImpl
from src.simulation.agent import AgentImpl
from src.simulation.engine import LLMEngineImpl
import src.simulation.models as m
from src.simulation.request import RequestImpl
import src.simulation.utils as utils
import src.simulation.config as config


def check_rate(
    agent_id: m.AgentId,
    hw_mig: m.MIGProfileBase,
    sampled_req_ids: List[str],
    rate: float,
    gpu_id: int,
) -> bool:
    requests = []
    model_name = utils.SIM_CONFIG.get_model(agent_id, hw_mig, gpu_id=gpu_id)
    req_map = utils.TOKENS_MAP[agent_id][model_name]

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

    agent = AgentImpl(agent_id)
    eid = utils.generate_engine_id(gpu_id, hw_mig.string)
    engine = LLMEngineImpl.create(
        gpu=gpu_id,
        engine_id=eid,
        owner=agent,
        mig_profile=hw_mig,
        current_time=0.0,
        mig_index=0,
        is_permanent=True,
    )
    agent.add_engine(engine)

    agents: Dict[m.AgentId, m.Agent] = {aid: AgentImpl(aid) for aid in m.AgentId}
    agents[agent_id] = agent

    sim = SimulatorImpl(agents=agents, engines={eid: engine}, no_log=True)
    sim.add_arrival_events(requests)
    for e in sim.engines.values():
        e.activate(sim.current_time)

    while sim.run():
        pass

    # We define "queuing happens" if the wait time in the queue grows continuously.
    last_10_percent = requests[-int(len(requests) * 0.1) :]
    valid_reqs = [r for r in last_10_percent if r.start_time is not None]
    if not valid_reqs:
        return False

    avg_delay = sum(max(0, r.start_time - r.arrival_time) for r in valid_reqs) / len(
        valid_reqs
    )
    return avg_delay < 2.0


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

    if "heuristic" not in config_data:
        config_data["heuristic"] = {}

    config_data["heuristic"]["service_rates"] = results_dict

    with open(config_path, "w") as f:
        yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)

    print("\nResults successfully saved to configs/bench_config.yaml")


if __name__ == "__main__":
    measure_max_service_rate()
