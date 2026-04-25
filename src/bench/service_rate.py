import yaml
from pathlib import Path
import random
from typing import Dict, List

from src.simulation.simulator import SimulatorImpl
from src.simulation.agent import AgentImpl
from src.simulation.engine import LLMEngineImpl
import src.simulation.models as m
from src.simulation.request import RequestImpl
import src.simulation.utils as utils


def check_rate(
    agent_id: m.AgentId,
    mig_profile: m.MIGProfile,
    sampled_req_ids: List[str],
    rate: float,
) -> bool:
    requests = []
    model_name = utils.SIM_CONFIG.get_model(agent_id, mig_profile)
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
    eid = utils.generate_engine_id(0, mig_profile.string)
    engine = LLMEngineImpl.create(
        gpu=0,
        engine_id=eid,
        owner=agent,
        mig_profile=mig_profile,
        current_time=0.0,
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
    # We check the average queuing delay of the last 10% of requests.
    last_10_percent = requests[-int(len(requests) * 0.1) :]

    # Delay is the time between arriving at the system and starting prefill.
    # For RAG, this includes search overhead (which is small, <0.3s).
    valid_reqs = [r for r in last_10_percent if r.start_time is not None]
    if not valid_reqs:
        return False

    avg_delay = sum(max(0, r.start_time - r.arrival_time) for r in valid_reqs) / len(
        valid_reqs
    )

    # A 2.0s threshold allows for normal micro-batching jitter,
    # but strictly rejects unstable rates where the queue grows indefinitely.
    return avg_delay < 2.0


def measure_max_service_rate():
    num_requests = 1000  # Enough to observe queue growth if rate is unsustainable

    # 1. Use the same set of requests among all MIGs to maintain fairness
    agent_sampled_reqs: Dict[m.AgentId, List[str]] = {}
    for agent_id in m.AgentId:
        first_model = next(iter(utils.TOKENS_MAP[agent_id]))
        all_rids = list(utils.TOKENS_MAP[agent_id][first_model].keys())
        agent_sampled_reqs[agent_id] = random.choices(all_rids, k=num_requests)

    print(f"{'Agent':<15} | {'MIG Profile':<15} | {'Max Rate (req/s)':<25}")
    print("-" * 55)

    results_dict = {
        m.AgentId.CODING.value: {},
        m.AgentId.RAG.value: {},
    }

    for agent_id in [m.AgentId.CODING, m.AgentId.RAG]:
        # 2. Use list(m.MIGProfile) instead of hardcoding
        for mig_profile in list(m.MIGProfile):
            sampled_req_ids = agent_sampled_reqs[agent_id]

            # Binary search for the maximum rate that doesn't cause queuing
            low = 0.1
            high = 20.0
            best_rate = 0.0

            while high - low > 0.05:
                mid = (low + high) / 2
                is_stable = check_rate(agent_id, mig_profile, sampled_req_ids, mid)
                if is_stable:
                    best_rate = mid
                    low = mid
                else:
                    high = mid

            results_dict[agent_id.value][mig_profile.string] = float(f"{best_rate:.2f}")
            print(
                f"{agent_id.value:<15} | {mig_profile.string:<15} | {best_rate:>15.2f}"
            )

    # Write to configs/bench_config.yaml
    config_path = Path("configs/bench_config.yaml")
    with open(config_path, "r") as f:
        config_data = yaml.safe_load(f)

    if "heuristic" not in config_data:
        config_data["heuristic"] = {}

    config_data["heuristic"]["utilization_factor"] = 0.8
    config_data["heuristic"]["high_threshold"] = 1.2
    config_data["heuristic"]["low_threshold"] = 0.8
    config_data["heuristic"]["service_rates"] = results_dict

    with open(config_path, "w") as f:
        yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)

    print("\nResults successfully saved to configs/bench_config.yaml")


if __name__ == "__main__":
    measure_max_service_rate()
