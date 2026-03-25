import random
import argparse
from typing import Dict, List

from src.simulation.models import *
import src.simulation.utils as utils
from src.simulation.request import RequestImpl
from src.simulation.simulator import SimulatorImpl
from src.simulation.engine import LLMEngineImpl
from src.simulation.agent import AgentImpl


def load_requests(
    arrival_interval_sec: float = 0.5, start_time: float = 0.0, turn: int = 0
) -> List[Request]:
    """
    Loads arriving Requests from the token map. Only prompt_tokens is set here;
    completion_tokens is determined at dispatch time based on the assigned engine's model.
    """
    requests: List[Request] = []

    for agent_id in AgentId:
        first_model = next(iter(utils.TOKENS_MAP[agent_id]))
        req_map = utils.TOKENS_MAP[agent_id][first_model]

        if agent_id == AgentId.CODING:
            # Pick 25,000 requests
            selected = random.sample(list(req_map.items()), 25000)
            for rid, (prompt_tokens, _) in selected:
                requests.append(
                    RequestImpl(
                        id=f"{rid}_t{turn}",
                        agent_id=agent_id,
                        prompt_tokens=prompt_tokens,
                        original_id=rid,
                    )
                )
        elif agent_id == AgentId.RAG:
            all_items = list(req_map.items())
            rag_requests: List[Request] = []

            # Duplicate each row
            for rid, (prompt_tokens, _) in all_items:
                rag_requests.append(
                    RequestImpl(
                        id=f"{rid}_dup1_t{turn}",
                        agent_id=agent_id,
                        prompt_tokens=prompt_tokens,
                        original_id=rid,
                    )
                )
            for rid, (prompt_tokens, _) in all_items:
                rag_requests.append(
                    RequestImpl(
                        id=f"{rid}_dup2_t{turn}",
                        agent_id=agent_id,
                        prompt_tokens=prompt_tokens,
                        original_id=rid,
                    )
                )

            # Fill to 25,000
            needed = 25000 - len(rag_requests)
            if needed > 0:
                extra = random.sample(all_items, needed)
                for i, (rid, (prompt_tokens, _)) in enumerate(extra):
                    rag_requests.append(
                        RequestImpl(
                            id=f"{rid}_extra_{i}_t{turn}",
                            agent_id=agent_id,
                            prompt_tokens=prompt_tokens,
                            original_id=rid,
                        )
                    )
            elif needed < 0:
                rag_requests = rag_requests[:25000]

            requests.extend(rag_requests)

    random.seed(42)
    random.shuffle(requests)

    for i, req in enumerate(requests):
        req.arrival_time = start_time + i * arrival_interval_sec

    return requests


def main():
    parser = argparse.ArgumentParser(description="Run simulation")
    parser.add_argument("--no-log", action="store_true", help="Disable logging")
    args = parser.parse_args()

    print("Loading config and datasets...")
    load_turn = 0
    requests = load_requests(turn=load_turn)
    print(f"Loaded {len(requests)} requests.")

    agents: Dict[AgentId, Agent] = {}
    engines: Dict[str, LLMEngine] = {}
    for aid in AgentId:
        agents[aid] = AgentImpl(aid)

    for eng_conf in utils.SIM_CONFIG.initial_state:
        mig = MIGProfile.from_string(eng_conf["mig"])
        gpu = int(eng_conf["gpu"])
        agent_name = eng_conf["agent"]
        agent = agents[AgentId(agent_name)]
        eid = utils.generate_engine_id(agent_name, gpu, mig.string)

        is_permanent = eng_conf.get("is-permanent", False)
        eng = LLMEngineImpl.create(
            gpu=gpu,
            engine_id=eid,
            owner=agent,
            mig_profile=mig,
            current_time=0.0,
            is_permanent=is_permanent,
        )

        agent.add_engine(eng)
        engines[eid] = eng

    sim = SimulatorImpl(
        agents=agents,
        engines=engines,
        no_log=args.no_log,
    )

    # For a quicker test we limit requests
    sim.add_arrival_events(requests[:1000])

    # Pre-populate triggering events upfront for step advanced mockup
    max_steps = 100
    sim.schedule_resource_manager_triggers(max_steps)

    print("Running mockup training loop...")
    for step in range(max_steps):
        # 1. Choose a random valid action
        mask = sim.get_action_mask()
        valid_actions = [a for a, m in zip(ResourceManagerAction, mask) if m]
        action = random.choice(valid_actions)

        if step > 0:
            sim.handle_resource_manager_trigger(action)
        sim.run()

        # 2. Print State
        state_data = sim.environment_state.get_state(
            sim.current_time, sim.agents, sim.engines
        )
        print(f"Step {step} (Time {sim.current_time:.2f}s) - Action: {action}")
        avg_q = state_data["avg_queue_length"]
        print(f"  Avg Queue Length: {sum(avg_q.values()):.2f}")

        # Count remaining arrival events to replenish proactively
        remain = sim.pending_arrival_count
        if remain < 1000:
            max_arr_time = sim.latest_arrival_time
            print(
                f"  [Replenish] Only {remain} arrivals left. Adding batch starting at {max_arr_time:.2f}s"
            )
            load_turn += 1
            new_requests = load_requests(start_time=max_arr_time, turn=load_turn)
            sim.add_arrival_events(new_requests[:1000])

    print("\n====== Simulation Finished ======")
    print(f"Total simulated time: {sim.current_time:.2f} seconds")

    for agent_id in AgentId:
        reqs = sim.agents[agent_id].completed_requests
        print(f"\nAgent {agent_id.value}: {len(reqs)} requests completed.")
        if reqs:
            # Latency
            avg_latency = sum(
                r.finish_time - r.arrival_time
                for r in reqs
                if r.finish_time is not None
            ) / len(reqs)

            # TTFT
            valid_ttfts = [
                r.first_token_time - r.arrival_time
                for r in reqs
                if r.first_token_time is not None
            ]
            avg_ttft = sum(valid_ttfts) / len(valid_ttfts) if valid_ttfts else 0.0

            # TPOT
            valid_tpots = [
                r.decode_time / r.generated_tokens
                for r in reqs
                if r.generated_tokens > 0
            ]
            avg_tpot = sum(valid_tpots) / len(valid_tpots) if valid_tpots else 0.0

            # Queuing Time
            valid_queueing = [
                r.start_time - r.arrival_time for r in reqs if r.start_time is not None
            ]
            avg_queueing = (
                sum(valid_queueing) / len(valid_queueing) if valid_queueing else 0.0
            )

            # Throughput: token per sec
            throughput_time = sum(
                r.finish_time - r.start_time
                for r in reqs
                if r.finish_time is not None and r.start_time is not None
            )
            avg_throughput = sum(r.completion_tokens for r in reqs) / throughput_time

            print(f"  Avg latency:        {avg_latency:.4f}s")
            print(f"  Avg TTFT:           {avg_ttft:.4f}s")
            print(f"  Avg Queuing Time:   {avg_queueing:.4f}s")
            print(f"  Avg TPOT:           {avg_tpot:.4f}s")
            print(f"  Avg Throughput:     {avg_throughput:.4f} tok/s")


if __name__ == "__main__":
    main()
