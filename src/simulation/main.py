import random
from pathlib import Path
from models import Request, AgentId
from engine import LLMEngine
from agent import Agent
from simulator import Simulator
import global_vars as g


def load_requests(arrival_interval_sec: float = 0.5) -> list[Request]:
    """
    Loads arriving Requests from the token map. Only prompt_tokens is set here;
    completion_tokens is determined at dispatch time based on the assigned engine's model.
    """
    requests = []

    for agent_id in AgentId:
        # Use the first model's token counts as the canonical prompt_tokens source
        first_model = next(iter(g.TOKENS_MAP[agent_id]))
        req_map = g.TOKENS_MAP[agent_id][first_model]

        for rid, (prompt_tokens, _) in req_map.items():
            req = Request(
                id=rid,
                agent_id=agent_id,
                prompt_tokens=prompt_tokens,
            )
            requests.append(req)

    random.seed(42)
    random.shuffle(requests)

    for i, req in enumerate(requests):
        req.arrival_time = i * arrival_interval_sec

    return requests


def main():
    base_dir = Path(".")

    print("Loading config and datasets...")
    g.init(base_dir)

    requests = load_requests()
    print(f"Loaded {len(requests)} requests.")

    print("Initializing System Architecture...")

    coding_agent = Agent(AgentId.CODING)
    rag_agent = Agent(AgentId.RAG)

    engines = {}
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
    )

    print("Feeding events...")
    # For a quicker test we limit requests
    sim.add_arrival_events(requests[:50000])

    print("Running simulation...")
    sim.run()

    print("\nSimulation Finished!")
    print(f"Total simulated time: {sim.current_time:.2f} seconds")

    for agent_id in AgentId:
        reqs = sim.resource_manager.agent_completed[agent_id]
        print(f"\nAgent {agent_id.value}: {len(reqs)} requests completed.")
        if reqs:
            avg_latency = sum(r.finish_time - r.arrival_time for r in reqs) / len(reqs)
            valid_ttfts = [
                r.first_token_time - r.arrival_time
                for r in reqs
                if r.first_token_time is not None
            ]
            avg_ttft = sum(valid_ttfts) / len(valid_ttfts) if valid_ttfts else 0.0
            print(f"  Avg latency: {avg_latency:.4f}s")
            print(f"  Avg TTFT:    {avg_ttft:.4f}s")


if __name__ == "__main__":
    main()
