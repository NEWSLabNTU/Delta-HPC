import argparse
from pathlib import Path
import numpy as np
import gymnasium as gym
from tqdm import tqdm
from typing import List
import tabulate

import src.simulation.models as m
from src.simulation.agent import AgentImpl
from src.simulation.simulator import SimulatorImpl
import src.simulation.utils as utils

from src.training.train import MIGResourceEnv

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from sb3_contrib import MaskablePPO

from src.bench.models import BenchMode, Workload
from src.bench.config import BENCH_CONFIG
from src.bench.request_loader import BenchRequestLoader


class BenchMIGResourceEnv(MIGResourceEnv):
    def __init__(
        self,
        simulator: m.Simulator,
        workload: Workload,
        baseline_mode: BenchMode = None,
    ):
        super().__init__(simulator, enable_log=False)
        self.workload = workload
        self.baseline_mode = baseline_mode
        self.phase_history = {}
        self.enable_replenish = False

        # Overwrite episode length
        self.max_steps = BENCH_CONFIG.benchmark_length

    def reset(self, *, seed=None, options=None):
        gym.Env.reset(self, seed=seed)
        self.current_step = 0
        self.load_turn = 0
        self.episode_count += 1

        # Override initial state for baselines if needed
        original_initial_state = utils.SIM_CONFIG.initial_state
        if self.baseline_mode == BenchMode.BASELINE_2_2_2_1:
            new_state = []
            for conf in original_initial_state:
                if conf.get("is-permanent", False):
                    new_state.append(conf)
                else:
                    new_state.append(
                        {
                            "gpu": conf["gpu"],
                            "mig": "2g.10gb",
                            "agent": conf["agent"],
                            "is-permanent": False,
                        }
                    )
                    new_state.append(
                        {
                            "gpu": conf["gpu"],
                            "mig": "2g.10gb",
                            "agent": conf["agent"],
                            "is-permanent": False,
                        }
                    )
                    new_state.append(
                        {
                            "gpu": conf["gpu"],
                            "mig": "2g.10gb",
                            "agent": conf["agent"],
                            "is-permanent": False,
                        }
                    )
                    new_state.append(
                        {
                            "gpu": conf["gpu"],
                            "mig": "1g.10gb",
                            "agent": conf["agent"],
                            "is-permanent": False,
                        }
                    )
            utils.SIM_CONFIG.initial_state = new_state

        try:
            self.sim.reset()

            self.request_loader = BenchRequestLoader(self.workload)
            requests = []
            for aid in m.AgentId:
                requests.extend(
                    self.request_loader.generate_requests(
                        agent_id=aid, turn=self.load_turn
                    )
                )

            self.sim.init_simulator(requests, BENCH_CONFIG.benchmark_length)
            self.sim.run()  # advance to the first action interval
            self.phase_history = self.request_loader.phase_history

            state_data = self.sim.get_state(self.current_step)
            obs = self._get_obs(state_data)
        finally:
            utils.SIM_CONFIG.initial_state = original_initial_state

        return obs, {}


class BenchRunner:
    def __init__(self, workload: Workload, mode: BenchMode, ckpt_path: Path):
        self.workload = workload
        self.mode = mode  # "RL", "7g", "2_2_2_1"
        self.ckpt_path = ckpt_path
        self.results = {}

    def run(self):
        agents = {}
        engines = {}  # Sim reset will rebuild these
        for aid in m.AgentId:
            agents[aid] = AgentImpl(aid)

        sim = SimulatorImpl(agents=agents, engines=engines, no_log=True)
        env = BenchMIGResourceEnv(
            sim,
            self.workload,
            baseline_mode=self.mode if self.mode != BenchMode.RL else None,
        )

        model = None
        venv = None
        if self.mode == BenchMode.RL:
            vec_env = DummyVecEnv([lambda: env])
            norm_path = self.ckpt_path.with_name(
                f"{self.ckpt_path.stem}_vecnormalize.pkl"
            )
            if norm_path.exists():
                venv = VecNormalize.load(str(norm_path), vec_env)
                venv.training = False
                venv.norm_reward = False
            else:
                venv = vec_env

            custom_objects = {
                "learning_rate": 0.0,
                "clip_range": 0.2,
                "lr_schedule": lambda _: 0.0,
            }
            model = MaskablePPO.load(
                self.ckpt_path,
                env=venv,
                device="cuda",
                custom_objects=custom_objects,
                verbose=1,
            )
            obs = venv.reset()
        else:
            obs, _ = env.reset()

        # Metrics tracking
        total_steps = BENCH_CONFIG.benchmark_length
        queue_lengths_sum = {aid: 0 for aid in m.AgentId}
        split_count = {aid: 0 for aid in m.AgentId}
        merge_count = {aid: 0 for aid in m.AgentId}
        last_merge_split_time = {aid: 0.0 for aid in m.AgentId}
        action_durations = {aid: [] for aid in m.AgentId}
        completed_reqs_map = {aid: {} for aid in m.AgentId}

        for step in tqdm(
            range(total_steps),
            desc=f"{self.mode.name:<5} | {self.workload.name:<8}",
            leave=True,
            ncols=100,
        ):
            if self.mode == BenchMode.RL:
                assert model is not None
                action_masks = env.action_masks()
                action_np, _ = model.predict(
                    obs, action_masks=action_masks, deterministic=True
                )
                action = int(action_np[0])
            else:
                action = list(m.ResourceManagerAction).index(
                    m.ResourceManagerAction.NO_ACTION
                )

            enum_action = list(m.ResourceManagerAction)[action]

            # Record merge/split counts and durations
            if enum_action != m.ResourceManagerAction.NO_ACTION and isinstance(
                enum_action.value, m.MigAction
            ):
                act_val = enum_action.value
                aid = act_val.victim

                if last_merge_split_time[aid] > 0:
                    action_durations[aid].append(
                        env.sim.current_time - last_merge_split_time[aid]
                    )
                last_merge_split_time[aid] = env.sim.current_time

                if act_val.action == "split":
                    split_count[aid] += 1
                elif act_val.action == "merge":
                    merge_count[aid] += 1

            if self.mode == BenchMode.RL:
                obs, _, _, _ = venv.step([action])
            else:
                obs, _, _, _, _ = env.step(action)

            for aid, agent in env.sim.agents.items():
                ql_sum = sum(
                    len(e.waiting_queue)
                    for e in agent.engines
                    if e.status != m.EngineStatus.BOOTING
                )
                queue_lengths_sum[aid] += ql_sum

            # Accumulate completed requests before potential environment resets clear them
            for aid, reqs in env.sim.interval_requests.items():
                for req in reqs:
                    if req.is_finished and req.serving_engine is not None:
                        completed_reqs_map[aid][req.id] = req

        # Extract Episode Metrics
        ttft_list = {aid: [] for aid in m.AgentId}
        tpot_list = {aid: [] for aid in m.AgentId}
        tokens_by_mig = {aid: {prof: 0 for prof in m.MIGProfile} for aid in m.AgentId}

        for aid, req_map in completed_reqs_map.items():
            for req in req_map.values():
                if req.first_token_time is not None:
                    ttft = req.first_token_time - req.arrival_time
                else:
                    ttft = (
                        req.finish_time - req.arrival_time if req.finish_time else 0.0
                    )
                ttft_list[aid].append(ttft)

                if req.generated_tokens > 0 and req.decode_time > 0:
                    tpot = req.decode_time / req.generated_tokens
                else:
                    tpot = 0.0
                tpot_list[aid].append(tpot)

                # Check serving engine MIG profile
                tokens_by_mig[aid][req.serving_engine.mig_profile] += (
                    req.generated_tokens
                )

        # Synthesize results
        res = {}
        for aid in m.AgentId:
            total_tokens = sum(tokens_by_mig[aid].values())
            res[aid.value] = {
                "ttft_percentiles": np.percentile(
                    ttft_list[aid], [25, 50, 75, 99]
                ).tolist()
                if ttft_list[aid]
                else [0, 0, 0, 0],
                "tpot_quartiles": np.percentile(tpot_list[aid], [25, 50, 75]).tolist()
                if tpot_list[aid]
                else [0, 0, 0],
                "avg_waiting_queue": queue_lengths_sum[aid] / total_steps,
                "split_count": split_count[aid],
                "merge_count": merge_count[aid],
                "avg_duration_between_actions": np.mean(action_durations[aid])
                if action_durations[aid]
                else 0.0,
                "token_mig_percentages": {
                    k: (v / total_tokens * 100 if total_tokens > 0 else 0)
                    for k, v in tokens_by_mig[aid].items()
                },
            }

        # Workload summary aggregation
        workload_summary = {}
        for aid, phases in env.phase_history.items():
            summary = {}
            for p in phases:
                pat = p["pattern"]
                if pat not in summary:
                    summary[pat] = {"total_duration": 0, "weighted_rate": 0}
                summary[pat]["total_duration"] += p["duration"]
                summary[pat]["weighted_rate"] += p["avg_rate"] * p["duration"]

            final_phases = []
            total_ben_dur = sum(s["total_duration"] for s in summary.values())
            for pat in sorted(summary.keys()):
                s = summary[pat]
                final_phases.append(
                    {
                        "pattern": pat,
                        "avg_rate": s["weighted_rate"] / s["total_duration"]
                        if s["total_duration"] > 0
                        else 0,
                        "duration": s["total_duration"],
                        "proportion": (s["total_duration"] / total_ben_dur * 100)
                        if total_ben_dur > 0
                        else 0,
                    }
                )
            workload_summary[aid.value] = final_phases

        res["workload_summary"] = workload_summary
        self.results = res


def print_metrics_table(mode_name: str, workload_name: str, results: dict):
    print(f"\nMode: {mode_name} | Workload: {workload_name}")
    headers = [
        "Agent",
        "TTFT (P25/50/75/99)",
        "TPOT (P25/50/75)",
        "Avg Q",
        "S/M",
        "Tokens By MIG (%)",
    ]
    table_data = []

    # Filter out workload_summary and then iterate
    agents = [k for k in results.keys() if k != "workload_summary"]
    for i, aid in enumerate(agents):
        metrics = results[aid]
        if i > 0:
            table_data.append(tabulate.SEPARATING_LINE)

        ttft_str = "/".join([f"{x:.3f}" for x in metrics["ttft_percentiles"]])
        tpot_str = "/".join([f"{x:.3f}" for x in metrics["tpot_quartiles"]])
        avg_q_str = f"{metrics['avg_waiting_queue']:.3f}"
        sm_str = f"{metrics['split_count']}/{metrics['merge_count']}"

        mig_tokens = []
        for mig in sorted(
            metrics["token_mig_percentages"].keys(), key=lambda m: m.size
        ):
            val = metrics["token_mig_percentages"][mig]
            if val > 0:
                mig_tokens.append(f"{mig.size}g:{val:.1f}%")
        mig_str = ", ".join(mig_tokens)

        table_data.append([aid, ttft_str, tpot_str, avg_q_str, sm_str, mig_str])

    print(tabulate.tabulate(table_data, headers=headers, tablefmt="fancy_outline"))


def print_workload_summary_table(results: dict):
    summary = results.get("workload_summary", {})
    if not summary:
        return

    print("\nWorkload Summary (Actual Patterns Encountered)")
    headers = ["Agent", "Pattern", "Avg Rate (req/s)", "Duration (s)", "Proportion (%)"]
    table_data = []

    for j, aid in enumerate(sorted(summary.keys())):
        if j > 0:
            table_data.append(tabulate.SEPARATING_LINE)

        phases = summary[aid]
        for i, p in enumerate(phases):
            agent_str = aid if i == 0 else ""
            table_data.append(
                [
                    agent_str,
                    p["pattern"],
                    f"{p['avg_rate']:.3f}",
                    f"{p['duration']:.3f}",
                    f"{p['proportion']:.1f}%",
                ]
            )

    print(tabulate.tabulate(table_data, headers=headers, tablefmt="fancy_outline"))


def main():
    parser = argparse.ArgumentParser(description="Run Performance Benchmarks")
    parser.add_argument(
        "--ckpt", type=Path, default=None, help="Path to RL Checkpoint Zip"
    )
    parser.add_argument(
        "--bl",
        action="store_true",
        help="Run the baseline benchmarks",
    )
    args = parser.parse_args()

    modes: List[BenchMode] = []
    if args.ckpt:
        modes.append(BenchMode.RL)
    if args.bl:
        modes.extend([BenchMode.BASELINE_7G, BenchMode.BASELINE_2_2_2_1])
    if not modes:
        parser.error("No benchmark to run. Use --ckpt for RL and --bl for baselines.")

    print("\n" + "=" * 60)
    print("STARTING HYBRID BENCHMARKS")
    print("=" * 60)

    # Run each mode sequentially with Hybrid workload
    for mode in modes:
        r = BenchRunner(workload=Workload.HYBRID, mode=mode, ckpt_path=args.ckpt)
        r.run()
        print_metrics_table(mode.name, r.workload.name, r.results)
        print_workload_summary_table(r.results)

    print("\nBenchmark Suite Completed.")


if __name__ == "__main__":
    main()
