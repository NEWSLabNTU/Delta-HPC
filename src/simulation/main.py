import random
import argparse
from typing import Dict, List

import src.simulation.models as m
import src.simulation.utils as utils
from src.simulation.request_loader import RequestLoader
from src.simulation.simulator import SimulatorImpl
from src.simulation.engine import LLMEngineImpl
from src.simulation.agent import AgentImpl

from src.training.rewards import compute_reward


def main():
    parser = argparse.ArgumentParser(description="Run simulation")
    parser.add_argument("--no-log", action="store_true", help="Disable logging")
    parser.add_argument(
        "--phase", type=int, default=1, choices=[1, 2], help="Training phase (1 or 2)"
    )
    args = parser.parse_args()

    phase = m.TrainingPhase(args.phase)
    print(f"Starting simulation in {phase.name}...")

    print("Loading config and datasets...")
    load_turn = 0
    request_loader = RequestLoader()
    requests: List[m.Request] = []
    for aid in m.AgentId:
        requests.extend(request_loader.generate_requests(agent_id=aid, turn=load_turn))
    print(f"Loaded {len(requests)} requests.")

    agents: Dict[m.AgentId, m.Agent] = {}
    engines: Dict[str, m.LLMEngine] = {}
    for aid in m.AgentId:
        agents[aid] = AgentImpl(aid)

    for eng_conf in utils.SIM_CONFIG.initial_state:
        mig = m.MIGProfile.from_string(eng_conf["mig"])
        gpu = int(eng_conf["gpu"])
        agent_name = eng_conf["agent"]
        agent = agents[m.AgentId(agent_name)]
        eid = utils.generate_engine_id(gpu, mig.string)

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
    max_steps = 200
    sim.reset()
    sim.init_simulator(requests, max_steps)
    sim.run()  # advance to the first action interval

    print("Running mockup training loop...")
    for step in range(max_steps):
        # 1. Choose a random valid action
        mask = sim.get_action_mask()

        # Apply phase-based masking
        if phase == m.TrainingPhase.PHASE_1:
            # Mask MIG Split/Merge
            for act_id, action in enumerate(m.ResourceManagerAction):
                if action != m.ResourceManagerAction.NO_ACTION and isinstance(
                    action.value, m.MigAction
                ):
                    mask[act_id] = False
        elif phase == m.TrainingPhase.PHASE_2:
            pass

        valid_actions = [a for a, msk in zip(m.ResourceManagerAction, mask) if msk]
        action = random.choice(valid_actions)
        # action = m.ResourceManagerAction.NO_ACTION
        print(f"Step {step} (Time {sim.current_time:.2f}s) - Action: {action}")
        for aid in m.AgentId:
            print(
                f" Agent {aid.value}: {[e.engine_id for e in sim.agents[aid].engines]}"
            )
        for i, msk in enumerate(mask):
            print(f"  {list(m.ResourceManagerAction)[i].name}: {msk}")

        sim.handle_resource_manager_trigger(action)
        sim.run()

        # 2. Print State
        state_data = sim.get_state(step + 1)
        avg_q = state_data["avg_queue_length"]
        total_avg_q = sum(sum(v) for v in avg_q.values())
        print(f"  Avg Queue Length: {total_avg_q:.2f}")

        reward = compute_reward(state_data["requests"], action, sim.current_time, agents=sim.agents)
        print(f"  Reward: {reward:.4f}")

        # Count remaining arrival events to replenish proactively
        for agent_id in sim.need_requests_replenish():
            max_arr_time = sim.latest_arrival_time(agent_id)
            print(
                f"  [Replenish] Agent {agent_id.value} adding batch starting at {max_arr_time:.2f}s"
            )
            load_turn += 1
            new_requests = request_loader.generate_requests(
                agent_id=agent_id, start_time=max_arr_time, turn=load_turn
            )
            sim.add_arrival_events(new_requests)

    print("\n====== Simulation Finished ======")
    print(f"Total simulated time: {sim.current_time:.2f} seconds")

    for agent_id in m.AgentId:
        reqs = sim.agents[
            agent_id
        ].completed_requests  # this now only has last 500 requests
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
            p90_ttft = list(sorted(valid_ttfts))[int(0.9 * len(valid_ttfts))]

            # TPOT
            valid_tpots = [
                r.decode_time / r.generated_tokens
                for r in reqs
                if r.generated_tokens > 0
            ]
            avg_tpot = sum(valid_tpots) / len(valid_tpots) if valid_tpots else 0.0
            p90_tpot = list(sorted(valid_tpots))[int(0.9 * len(valid_tpots))]

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
            print(f"  P90 TTFT:           {p90_ttft:.4f}s")
            print(f"  Avg Queuing Time:   {avg_queueing:.4f}s")
            print(f"  Avg TPOT:           {avg_tpot:.4f}s")
            print(f"  P90 TPOT:           {p90_tpot:.4f}s")
            print(f"  Avg Throughput:     {avg_throughput:.4f} tok/s")


if __name__ == "__main__":
    main()
