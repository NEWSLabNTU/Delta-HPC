"""Standalone counter for how many requests `just bench` generates per agent.

Replicates the exact control flow of RequestLoader.generate_requests
(src/share/request_loader.py) using only configs/bench_config.yaml,
configs/training_config.yaml and configs/simulation_config.yaml -- no
dataset/model loading required. Because it drives the same `random` calls
in the same order with the same seed, the per-agent counts match a real
`just bench` run exactly (workload_sequence is not set in bench_config.yaml,
so phases are picked via random.choice, same as the real loader).

Usage: python scripts/count_bench_requests.py
"""

import random
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
PATTERNS = ["idle", "even", "busy", "burst"]


def load_yaml(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def active_agents(sim_cfg: dict) -> list[str]:
    agent_conf = sim_cfg.get("agent", {})
    return [name for name, active in agent_conf.items() if active]


def count_requests_for_agent(
    agent_idx: int,
    agent_name: str,
    seed: int,
    num_steps: int,
    action_interval: float,
    workloads: dict,
) -> tuple[int, list[tuple[str, int]]]:
    random.seed(seed ^ (agent_idx * 0x9E3779B9))

    current_time = 0.0
    max_time = num_steps * action_interval

    count = 0
    phase_log: list[tuple[str, int]] = []

    while current_time < max_time:
        pattern = random.choice(PATTERNS)

        rate_cfg = workloads[pattern]["rate"][agent_name]
        min_rate, max_rate = float(rate_cfg[0]), float(rate_cfg[1])
        dur_cfg = workloads[pattern]["duration"]
        min_dur, max_dur = float(dur_cfg[0]), float(dur_cfg[1])

        duration = random.uniform(min_dur, max_dur)
        phase_end = current_time + duration

        phase_count = 0
        while current_time < phase_end and current_time < max_time:
            rate = random.uniform(min_rate, max_rate)
            current_time += random.expovariate(rate)
            if current_time >= max_time:
                break
            count += 1
            phase_count += 1

        phase_log.append((pattern, phase_count))

    return count, phase_log


def main():
    bench_cfg = load_yaml(ROOT / "configs" / "bench_config.yaml")
    training_cfg = load_yaml(ROOT / "configs" / "training_config.yaml")
    sim_cfg = load_yaml(ROOT / "configs" / "simulation_config.yaml")

    seed = int(bench_cfg.get("seed", 42))
    num_steps = int(bench_cfg["benchmark-length"])
    action_interval = float(training_cfg["training"]["action-interval"])
    workloads = bench_cfg["workloads"]

    agents = active_agents(sim_cfg)

    print(f"seed={seed} benchmark-length={num_steps} action-interval={action_interval}")
    print(f"active agents (enum order): {agents}")
    print(f"max simulated time per agent: {num_steps * action_interval:.1f}s\n")

    total = 0
    for agent_idx, agent_name in enumerate(agents):
        n, phases = count_requests_for_agent(
            agent_idx, agent_name, seed, num_steps, action_interval, workloads
        )
        total += n
        print(f"{agent_name}: {n} requests ({len(phases)} phases)")

    print(f"\nTotal requests: {total}")


if __name__ == "__main__":
    main()
