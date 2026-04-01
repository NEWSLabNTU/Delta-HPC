import argparse
from typing import Dict

import src.simulation.models as m
import src.simulation.utils as utils
from src.simulation.request_loader import RequestLoader
from src.training.config import TRAINING_CONFIG
from src.simulation.simulator import SimulatorImpl
from src.simulation.engine import LLMEngineImpl
from src.simulation.agent import AgentImpl

from src.training.rewards import compute_reward


def main():
    parser = argparse.ArgumentParser(description="Run simulation")
    parser.add_argument("--no-log", action="store_true", help="Disable logging")
    args = parser.parse_args()

    print("Loading config and datasets...")
    load_turn = 0
    request_loader = RequestLoader(phase=TRAINING_CONFIG.phase)
    requests = request_loader.generate_requests(turn=load_turn)
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
    sim.init_simulator(requests, max_steps)
    sim.run()  # advance to the first action interal

    print("Running mockup training loop...")
    for step in range(max_steps):
        # 1. Choose a random valid action
        mask = sim.get_action_mask()
        if TRAINING_CONFIG.phase == 1:
            for i in range(1, 5):
                mask[i] = False

        valid_actions = [a for a, m in zip(m.ResourceManagerAction, mask) if m]
        # action = random.choice(valid_actions)
        action = m.ResourceManagerAction.NO_ACTION
        print(f"Step {step} (Time {sim.current_time:.2f}s) - Action: {action}")
        print(f"Action mask: {mask}")

        sim.handle_resource_manager_trigger(action)
        sim.run()

        # 2. Print State
        state_data = sim.environment_state.get_state(
            sim.current_time, sim.agents, sim.engines
        )
        avg_q = state_data["avg_queue_length"]
        print(f"  Avg Queue Length: {sum(avg_q.values()):.2f}")

        reward = compute_reward(state_data["requests"], action)
        print(f"  Reward: {reward:.4f}")

        # Count remaining arrival events to replenish proactively
        remain = sim.pending_arrival_count
        if remain < 1000:
            max_arr_time = sim.latest_arrival_time
            print(
                f"  [Replenish] Only {remain} arrivals left. Adding batch starting at {max_arr_time:.2f}s"
            )
            load_turn += 1
            new_requests = request_loader.generate_requests(
                start_time=max_arr_time, turn=load_turn
            )
            sim.add_arrival_events(new_requests)

    print("\n====== Simulation Finished ======")
    print(f"Total simulated time: {sim.current_time:.2f} seconds")

    for agent_id in m.AgentId:
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
