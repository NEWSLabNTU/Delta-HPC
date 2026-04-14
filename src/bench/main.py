import argparse
import tabulate
import numpy as np
from tqdm import tqdm
from pathlib import Path
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from typing import Any, Dict, List

import src.simulation.models as m
from src.simulation.agent import AgentImpl
from src.simulation.simulator import SimulatorImpl
from src.bench.models import BenchMode, Workload
from src.bench.config import BENCH_CONFIG
from src.bench.request_loader import BenchRequestLoader
from src.bench.prints import print_metrics, print_workloads
from src.bench.env import BenchMIGResourceEnv
import src.simulation.utils as utils


class BenchRunner:
    def __init__(
        self,
        workload: Workload,
        mode: BenchMode,
        ckpt_path: Path,
        requests: List[m.Request],
    ):
        self.workload = workload
        self.mode = mode  # "RL", "7g", "2_2_2_1"
        self.ckpt_path = ckpt_path
        self.requests = requests
        self.results: Dict[str, Any] = {}

    def _display_initial_state(self):
        # Display Initial State
        print(f"\n[Initial State: {self.mode.name}]")
        state_info: List[List[str]] = [
            [
                e["gpu"],
                e["agent"],
                e["mig"],
                "✓" if e.get("is-permanent", False) else " ",
            ]
            for e in utils.SIM_CONFIG.initial_state
        ]
        print(
            tabulate.tabulate(
                state_info,
                headers=["GPU", "Agent", "MIG", "Perm"],
                tablefmt="fancy_outline",
            )
        )

    def run(self):
        agents: Dict[m.AgentId, m.Agent] = {}
        engines: Dict[str, m.LLMEngine] = {}  # Sim reset will rebuild these
        for aid in m.AgentId:
            agents[aid] = AgentImpl(aid)

        sim = SimulatorImpl(agents=agents, engines=engines, no_log=True)
        env = BenchMIGResourceEnv(
            sim,
            self.workload,
            baseline_mode=self.mode,
            requests=self.requests,
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

            custom_objects: Dict[str, Any] = {
                "learning_rate": 0.0,
                "clip_range": 0.2,
                "lr_schedule": lambda: 0.0,
            }
            model = MaskablePPO.load(  # type: ignore
                self.ckpt_path,
                env=venv,
                device="cuda",
                custom_objects=custom_objects,
                verbose=1,
            )
            obs = venv.reset()  # type: ignore
        else:
            obs, _ = env.reset()

        self._display_initial_state()

        # Metrics tracking
        total_steps = BENCH_CONFIG.benchmark_length
        queue_lengths_sum = {aid: 0 for aid in m.AgentId}
        split_count = {aid: 0 for aid in m.AgentId}
        merge_count = {aid: 0 for aid in m.AgentId}
        transfer_count = {aid: 0 for aid in m.AgentId}
        completed_reqs_map: Dict[m.AgentId, Dict[str, m.Request]] = {
            aid: {} for aid in m.AgentId
        }
        presence_by_mig = {aid: {prof: 0 for prof in m.MIGProfile} for aid in m.AgentId}

        for _ in tqdm(
            range(total_steps),
            desc=f"{self.mode.name:<5} | {self.workload.name:<8}",
            leave=True,
            ncols=100,
        ):
            if self.mode == BenchMode.RL:
                assert model is not None
                action_masks = env.action_masks()
                action_np, _ = model.predict(  # type: ignore
                    obs, action_masks=action_masks, deterministic=True
                )
                action = int(action_np[0])
            else:
                action = list(m.ResourceManagerAction).index(
                    m.ResourceManagerAction.NO_ACTION
                )

            enum_action = list(m.ResourceManagerAction)[action]

            # Record merge/split/transfer counts and durations
            if enum_action != m.ResourceManagerAction.NO_ACTION:
                act_val = enum_action.value
                if isinstance(act_val, m.MigAction):
                    if act_val.action == "split":
                        split_count[act_val.victim] += 1
                    elif act_val.action == "merge":
                        merge_count[act_val.victim] += 1
                else:
                    # VRAMTransfer
                    transfer_count[act_val.giver] += 1

            if self.mode == BenchMode.RL:
                obs, _, _, _ = venv.step([action])  # type: ignore
            else:
                obs, _, _, _, _ = env.step(action)

            for aid, agent in env.sim.agents.items():
                ql_sum = sum(
                    len(e.waiting_queue)
                    for e in agent.engines
                    if e.status != m.EngineStatus.BOOTING
                )
                queue_lengths_sum[aid] += ql_sum
                for engine in agent.engines:
                    if not engine.is_permanent:
                        presence_by_mig[aid][engine.mig_profile] += 1

            # Accumulate completed requests before potential environment resets clear them
            for aid, reqs in env.sim.interval_requests.items():
                for req in reqs:
                    if req.is_finished and req.serving_engine is not None:
                        completed_reqs_map[aid][req.id] = req

        # Extract Episode Metrics
        ttft_list: Dict[m.AgentId, List[float]] = {aid: [] for aid in m.AgentId}
        tpot_list: Dict[m.AgentId, List[float]] = {aid: [] for aid in m.AgentId}
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
                assert req.serving_engine is not None
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
                "transfer_count": transfer_count[aid],
                "token_mig_percentages": {
                    k: (v / total_tokens * 100 if total_tokens > 0 else 0)
                    for k, v in tokens_by_mig[aid].items()
                },
                "mig_existence_percentages": {
                    prof: (count / total_steps * 100)
                    for prof, count in presence_by_mig[aid].items()
                },
            }

        self.results = res


def _get_workload_summary(
    phase_history: Dict[m.AgentId, List[Dict[str, Any]]],
) -> Dict[m.AgentId, List[Dict[str, Any]]]:
    # Workload summary aggregation
    workload_summary = {}
    for aid, phases in phase_history.items():
        summary: Dict[str, Dict[str, float]] = {}
        for p in phases:
            pat = p["pattern"]
            if pat not in summary:
                summary[pat] = {"total_duration": 0, "weighted_rate": 0}
            summary[pat]["total_duration"] += p["duration"]
            summary[pat]["weighted_rate"] += p["avg_rate"] * p["duration"]

        final_phases: List[Dict[str, Any]] = []
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
        workload_summary[aid] = final_phases
    return workload_summary


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
    print(f"Using seed: {BENCH_CONFIG.seed}")
    print("=" * 60)

    # Pre-generate the shared workload once so all modes see identical request patterns
    print("Pre-generating workload requests...")
    shared_loader = BenchRequestLoader(Workload.HYBRID, seed=BENCH_CONFIG.seed)
    shared_requests: List[m.Request] = []
    for aid in m.AgentId:
        shared_requests.extend(shared_loader.generate_requests(agent_id=aid, turn=0))
    print(f"Generated {len(shared_requests)} requests.")
    print_workloads(_get_workload_summary(shared_loader.phase_history))

    # Run each mode sequentially with the same pre-built workload
    for mode in modes:
        r = BenchRunner(
            workload=Workload.HYBRID,
            mode=mode,
            ckpt_path=args.ckpt,
            requests=shared_requests,
        )
        r.run()
        print_metrics(mode.name, r.workload.name, r.results)

    print("\nBenchmark Suite Completed.")


if __name__ == "__main__":
    main()
