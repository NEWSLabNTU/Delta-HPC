import random
from models import Request, AgentId
from engine import LLMEngine
from agent import Agent
from simulator import Simulator
import global_vars as g
from typing import Dict, List
import argparse


def load_requests(arrival_interval_sec: float = 0.5) -> list[Request]:
    """
    Loads arriving Requests from the token map. Only prompt_tokens is set here;
    completion_tokens is determined at dispatch time based on the assigned engine's model.
    """
    requests: List[Request] = []

    for agent_id in AgentId:
        first_model = next(iter(g.TOKENS_MAP[agent_id]))
        req_map = g.TOKENS_MAP[agent_id][first_model]

        if agent_id == AgentId.CODING:
            # Pick 25,000 requests
            selected = random.sample(list(req_map.items()), 25000)
            for rid, (prompt_tokens, _) in selected:
                requests.append(
                    Request(
                        id=rid,
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
                    Request(
                        id=f"{rid}_dup1",
                        agent_id=agent_id,
                        prompt_tokens=prompt_tokens,
                        original_id=rid,
                    )
                )
            for rid, (prompt_tokens, _) in all_items:
                rag_requests.append(
                    Request(
                        id=f"{rid}_dup2",
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
                        Request(
                            id=f"{rid}_extra_{i}",
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
        req.arrival_time = i * arrival_interval_sec

    return requests


def main():
    parser = argparse.ArgumentParser(description="Run simulation")
    parser.add_argument("--no-log", action="store_true", help="Disable logging")
    args = parser.parse_args()

    print("Loading config and datasets...")

    requests = load_requests()
    print(f"Loaded {len(requests)} requests.")

    print("Initializing System Architecture...")

    coding_agent = Agent(AgentId.CODING)
    rag_agent = Agent(AgentId.RAG)

    engines: Dict[str, LLMEngine] = {}
    for eng_conf in g.SIM_CONFIG.initial_state:
        eid = str(eng_conf["id"])
        mig = eng_conf["mig"]
        agent_id = AgentId(eng_conf["agent"])
        mname = g.SIM_CONFIG.get_model(agent_id, mig)
        eng = LLMEngine(
            engine_id=eid,
            model_name=mname,
            mig_profile=mig,
            max_batched_tokens=g.SIM_CONFIG.max_batched_tokens[mname],
            prefill_params=g.SIM_CONFIG.get_prefill_params(agent_id, mig),
            tpot_params=g.SIM_CONFIG.get_tpot_params(agent_id, mig),
            restart_time=g.SIM_CONFIG.get_restart_time(agent_id, mig),
        )

        if agent_id == AgentId.CODING:
            coding_agent.add_engine(eng)
        else:
            rag_agent.add_engine(eng)

        engines[eid] = eng

    print("Initializing Simulator...")
    sim = Simulator(
        agents={AgentId.CODING: coding_agent, AgentId.RAG: rag_agent},
        engines=engines,
        no_log=args.no_log,
    )

    print("Feeding events...")
    # For a quicker test we limit requests
    sim.add_arrival_events(requests[:50000])

    print("Running simulation...")
    sim.run()

    print("\nSimulation Finished!")
    print(f"Total simulated time: {sim.current_time:.2f} seconds")

    for agent_id in AgentId:
        reqs = sim.agents[agent_id].completed_requests
        print(f"\nAgent {agent_id.value}: {len(reqs)} requests completed.")
        if reqs:
            avg_latency = sum(
                r.finish_time - r.arrival_time
                for r in reqs
                if r.finish_time is not None
            ) / len(reqs)
            valid_ttfts = [
                r.first_token_time - r.arrival_time
                for r in reqs
                if r.first_token_time is not None
            ]
            avg_ttft = sum(valid_ttfts) / len(valid_ttfts) if valid_ttfts else 0.0

            valid_tpots = [
                r.decode_time / r.generated_tokens
                for r in reqs
                if r.generated_tokens > 0
            ]
            avg_tpot = sum(valid_tpots) / len(valid_tpots) if valid_tpots else 0.0

            valid_queueing = [
                r.start_time - r.arrival_time
                for r in reqs
                if r.start_time is not None
            ]
            avg_queueing = sum(valid_queueing) / len(valid_queueing) if valid_queueing else 0.0

            print(f"  Avg latency: {avg_latency:.4f}s")
            print(f"  Avg Queue:   {avg_queueing:.4f}s")
            print(f"  Avg TTFT:    {avg_ttft:.4f}s")
            print(f"  Avg TPOT:    {avg_tpot:.4f}s")


if __name__ == "__main__":
    main()
