"""
Builds the offline TTFT-vs-arrival-rate profiling table QAS (Quality-Aware
Scheduling, see src/bench/qas.py) uses to predict per-engine TTFT under a
candidate MIG allocation.

Earlier versions of this module tried to derive TTFT(lambda) in closed form
from the existing prefill/tpot regression parameters (configs/
simulation_config.yaml). Every closed-form variant required an assumption
about how concurrent load translates into per-step batch size, and each one
either extrapolated the regression far past its fitted domain or required
progressively more elaborate self-consistent equations to patch. Rather than
keep patching the math, this module instead sweeps arrival rate directly
through the same single-engine mini-simulation harness
src/bench/service_rate.py's measure_max_service_rate() already uses -- the
simulator already implements the real chunked-prefill/decode-interleaved
dynamics, so no analytical model of that mechanism is needed.

For each (agent, MIG profile, gpu), for a grid of arrival rates lambda
(fractions of the already-measured max stable rate mu, so each profile is
probed across a range meaningful to its own real capacity), a single-engine
simulation is run and the steady-state average TTFT (first_token_time -
arrival_time, over the tail window of completed requests, discarding the
warm-up period) is recorded. Rates that never reach steady state (the queue
is still growing) are dropped -- that lambda exceeds what this profile can
sustain.

Response quality (Q_f) is NOT profiled or stored here -- it stays a fully
online computation at decision time in src/bench/qas.py, reusing the existing
TRAINING_CONFIG.qf(...) hyperparameter table.
"""

import random
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import src.share.models as m
import src.simulation.utils as utils
import src.simulation.config as config
from src.bench.service_rate import build_rate_requests, run_single_engine_sim

LAMBDA_FRACTIONS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
NUM_REQUESTS = 1000


def measure_ttft_at_rate(
    agent_id: m.AgentId,
    hw_mig: m.MIGProfileBase,
    sampled_req_ids: List[str],
    rate: float,
    gpu_id: int,
) -> Optional[float]:
    """Runs the single-engine mini-simulation at arrival rate `rate` and
    returns the steady-state average TTFT (over the last 10% of completed
    requests), or None if the queue is still growing (this rate exceeds the
    profile's stable capacity)."""
    requests = build_rate_requests(agent_id, hw_mig, sampled_req_ids, rate, gpu_id)
    run_single_engine_sim(agent_id, hw_mig, gpu_id, requests)

    ttfts = [
        r.first_token_time - r.arrival_time
        for r in requests
        if r.first_token_time is not None
    ]

    first_10_percent_idx = max(1, int(len(ttfts) * 0.1))
    last_10_percent_idx = max(1, int(len(ttfts) * 0.9))

    first_ttfts = ttfts[:first_10_percent_idx]
    last_ttfts = ttfts[last_10_percent_idx:]

    if not first_ttfts or not last_ttfts:
        return None

    avg_ttft_first = sum(first_ttfts) / len(first_ttfts)
    avg_ttft_last = sum(last_ttfts) / len(last_ttfts)

    # Same instability criterion as service_rate.py's check_rate: if TTFT is
    # still growing by more than 1.0s between the start and end of the run,
    # the queue is backing up and this rate isn't sustainable.
    if avg_ttft_last - avg_ttft_first > 1.0:
        return None

    return avg_ttft_last


def build_curve(
    agent_id: m.AgentId,
    hw_mig: m.MIGProfileBase,
    gpu_id: int,
    mu: float,
    sampled_req_ids: List[str],
) -> List[Tuple[float, float]]:
    """Sweeps lambda as fractions of `mu`, returning the sorted (lambda,
    TTFT) points that reached steady state."""
    points: List[Tuple[float, float]] = []
    for frac in LAMBDA_FRACTIONS:
        rate = frac * mu
        ttft = measure_ttft_at_rate(agent_id, hw_mig, sampled_req_ids, rate, gpu_id)
        if ttft is not None:
            points.append((rate, ttft))

    points.sort(key=lambda p: p[0])
    return points


def measure_ttft_profile():
    # Sample once per agent, reused across every (profile, lambda) run for
    # that agent -- mirrors measure_max_service_rate()'s own sampling.
    agent_sampled_reqs: Dict[m.AgentId, List[str]] = {}
    for agent_id in m.AgentId:
        first_model = next(iter(utils.TOKENS_MAP[agent_id]))
        all_rids = list(utils.TOKENS_MAP[agent_id][first_model].keys())
        agent_sampled_reqs[agent_id] = random.choices(all_rids, k=NUM_REQUESTS)

    config_path = Path("configs/bench_config.yaml")
    with open(config_path, "r") as f:
        config_data = yaml.safe_load(f)

    # Reuse the (gpu_model, agent, mig_profile) nesting already populated by
    # service_rate.py's measure_max_service_rate(), so both tables share keys.
    service_rates = config_data.get("heuristic", {}).get("service_rates", {})

    profile_table: Dict[str, Dict[str, Dict[str, List[List[float]]]]] = {}

    print(
        f"{'GPU Model':<15} | {'Agent':<15} | {'MIG Profile':<15} | "
        f"{'μ (measured)':<15} | {'#λ points':<10}"
    )
    print("-" * 75)

    for gpu_model, agents_dict in service_rates.items():
        gpu_id = next(
            gid
            for gid, hw_cls in config.GPU_MIG_PROFILE.items()
            if next(iter(hw_cls)).gpu_model == gpu_model
        )
        model_table = profile_table.setdefault(gpu_model, {})

        for agent_value, mig_dict in agents_dict.items():
            # heuristic.service_rates may retain entries for agents disabled
            # in the current configs/simulation_config.yaml (e.g. ChatAgent in
            # 2-agent mode) -- skip those, matching what the active AgentId
            # enum actually supports.
            try:
                agent_id = m.AgentId(agent_value)
            except ValueError:
                continue
            agent_table = model_table.setdefault(agent_value, {})
            sampled_req_ids = agent_sampled_reqs[agent_id]

            for mig_str, mu in mig_dict.items():
                hw_mig = next(
                    p for p in config.GPU_MIG_PROFILE[gpu_id] if p.string == mig_str
                )
                curve = build_curve(
                    agent_id, hw_mig, gpu_id, float(mu), sampled_req_ids
                )
                agent_table[mig_str] = [[lam, ttft] for lam, ttft in curve]

                print(
                    f"{gpu_model:<15} | {agent_value:<15} | {mig_str:<15} | "
                    f"{float(mu):>15.2f} | {len(curve):>10}"
                )

    qas_cfg = config_data.setdefault("qas", {})
    qas_cfg.setdefault("L_target", 0.45)
    qas_cfg["profile_table"] = profile_table

    with open(config_path, "w") as f:
        yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)

    print("\nResults successfully saved to configs/bench_config.yaml")


if __name__ == "__main__":
    measure_ttft_profile()
