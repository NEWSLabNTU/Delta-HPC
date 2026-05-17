import random
from typing import Dict, List, cast

import src.share.models as m
import src.simulation.models as sm
from src.share.request_loader import RequestLoader
from src.simulation.simulator import SimulatorImpl
from src.training.config import TRAINING_CONFIG
from src.training.models import AgentPattern
from src.training.rewards import compute_reward


def main():
    print("Starting simulation...")

    print("Loading config and datasets...")
    load_turn = 0
    request_loader = RequestLoader(
        num_steps=TRAINING_CONFIG.episode_length,
        get_rate_range=lambda p, a: TRAINING_CONFIG.pattern_rate(AgentPattern(p), a),
        get_duration_range=lambda p: TRAINING_CONFIG.pattern_duration(AgentPattern(p)),
    )
    requests: List[m.Request] = []
    for aid in m.AgentId:
        requests.extend(request_loader.generate_requests(agent_id=aid, turn=load_turn))
    print(f"Loaded {len(requests)} requests.")

    agents: Dict[m.AgentId, m.Agent] = {}
    engines: Dict[str, m.LLMEngine] = {}
    sim = SimulatorImpl(
        agents=agents,
        engines=engines,
        no_log=False,
    )

    # For a quicker test we limit requests
    max_steps = 1024
    sim.reset()
    sim.init_simulator(requests, max_steps)
    sim.run()  # advance to the first action interval

    print("Running mockup training loop...")
    for step in range(max_steps):
        # 1. Choose a random valid action
        mask = sim.get_action_mask()

        valid_actions = [a for a, msk in zip(m.ResourceManagerAction, mask) if msk]
        action = random.choice(valid_actions)
        # action = sm.ResourceManagerAction.NO_ACTION
        print(f"Step {step} (Time {sim.current_time:.2f}s) - Action: {action}")
        print("  GPU Geometry:")
        for gpu_id, gpu_engines in sim.gpu_engines.items():
            print(f"    GPU {gpu_id}: {[e.engine_id for e in gpu_engines]}")
        for aid in m.AgentId:
            print(
                f" Agent {aid.value}: {[e.engine_id for e in sim.agents[aid].engines]}"
            )
        state_data = sim.get_state()
        avg_q = state_data["avg_queue_length"]
        total_avg_q = sum(sum(v) for v in avg_q.values())
        print(f"  Avg Queue Length: {total_avg_q:.2f}")
        print(f"  Current Budget:   {sim._environment_state.current_budget:.2f}s")
        for aid in m.AgentId:
            cooldowns = {
                k: sim._environment_state.get_steps_since(
                    aid, cast(sm.ActionHistoryKey, k)
                )
                for k in ["split", "merge", "give", "receive"]
            }
            print(f"    {aid.value} Cooldowns: {cooldowns}")
        for i, msk in enumerate(mask):
            print(f"  {list(m.ResourceManagerAction)[i].name}: {msk}")

        sim_action = sim.map_to_action(action)
        sim.handle_resource_manager_trigger(sim_action)
        sim.run()

        reward = compute_reward(
            state_data["requests"], action, sim.current_time, sim.gpu_engines
        )
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
