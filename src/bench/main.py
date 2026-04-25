from typing import Any, Dict, List, Optional, Tuple, cast
import os
import datetime
import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import src.simulation.models as m
from src.simulation.agent import AgentImpl
from src.simulation.simulator import SimulatorImpl
from src.training.models import TrainingPhase
from src.bench.env import BenchMIGResourceEnv
from src.bench.models import BenchMode, Workload, PhaseHistoryType
from src.bench.config import BENCH_CONFIG
from src.bench.request_loader import BenchRequestLoader
from src.bench.prints import (
    print_banner,
    print_metrics,
    print_workloads,
    print_matrix_metrics,
    print_initial_state,
)
from src.bench.heuristic import RuleBasedHeuristic


class BenchRunner:
    def __init__(
        self,
        ckpt: Optional[Path],
        workload: Workload,
        mode: BenchMode,
        requests: List[m.Request],
        phase_history: Dict[m.AgentId, List[PhaseHistoryType]],
        init_mode: m.InitialMIGCombination
        | Tuple[
            m.InitialMIGCombination, m.InitialMIGCombination
        ] = m.InitialMIGCombination.RANDOM,
    ):
        self.workload = workload
        self.mode = mode
        self.requests = requests
        self.phase_history = phase_history
        self._init_mode = init_mode
        self.model: Optional[MaskablePPO] = None
        self.env: BenchMIGResourceEnv
        self.venv: Optional[VecNormalize] = None
        self.obs: Any = None
        self.ckpt = ckpt

        self._setup_execution(ckpt)

    def run(self) -> Dict[str, Dict[str, Any]]:
        """Executes a complete benchmark run including thermal-up and flush phases."""
        # 1. Metrics Tracking Initialization
        stats = self._init_stats_tracking()

        # 2. Display Context
        print_initial_state(self._init_mode)

        # 3. Main Simulation Loop
        self._run_benchmark_loop(stats=stats)

        # 4. Flush Phase
        completed_reqs = self._flush_simulation(stats=stats)

        # 5. Plot Timeline
        self._plot_timeline(stats, self.ckpt)

        # 6. Synthesis
        return self._synthesize_results(completed_reqs=completed_reqs, stats=stats)

    def _setup_execution(self, ckpt: Optional[Path]):
        agents: Dict[m.AgentId, m.Agent] = {aid: AgentImpl(aid) for aid in m.AgentId}
        sim = SimulatorImpl(agents=agents, engines={}, no_log=True)
        self.env = BenchMIGResourceEnv(
            sim,
            bench_mode=self.mode,
            requests=self.requests,
            init_mode=self._init_mode,
        )

        model, venv = None, None
        if self.mode == BenchMode.RL:
            vec_env = DummyVecEnv([lambda: self.env])
            assert ckpt is not None
            norm_path = ckpt.with_name(f"{ckpt.stem}_vecnormalize.pkl")
            assert norm_path.exists()
            venv = VecNormalize.load(str(norm_path), vec_env)

            venv.training = venv.norm_reward = False  # type: ignore

            custom_objects: Dict[str, Any] = {
                "learning_rate": 0.0,
                "clip_range": 0.2,
                "lr_schedule": 0.0,
            }
            model = MaskablePPO.load(  # type: ignore
                ckpt, env=venv, device="cuda", custom_objects=custom_objects, verbose=1
            )
            self.obs = venv.reset()  # type: ignore
        else:
            self.obs, _ = self.env.reset()

        self.model = model
        self.venv = venv

    def _init_stats_tracking(self) -> Dict[str, Any]:
        patterns = [w.value for w in Workload if w != Workload.HYBRID]
        return {
            "queue_lengths_sum": {aid: 0 for aid in m.AgentId},
            "split_count": {aid: 0 for aid in m.AgentId},
            "merge_count": {aid: 0 for aid in m.AgentId},
            "transfer_count": {aid: 0 for aid in m.AgentId},
            "completed_reqs_map": {aid: {} for aid in m.AgentId},
            "presence_by_mig": {
                aid: {prof: 0 for prof in m.MIGProfile} for aid in m.AgentId
            },
            "joint_phase_ticks": {pc: {pr: 0 for pr in patterns} for pc in patterns},
            "patterns": patterns,
            "timeline_time": [],
            "timeline_pattern_coding": [],
            "timeline_pattern_rag": [],
            "timeline_actions": [],
        }

    def _run_benchmark_loop(
        self,
        stats: Dict[str, Any],
    ):
        heuristic = (
            RuleBasedHeuristic() if self.mode == BenchMode.BASELINE_HEURISTIC else None
        )

        for _ in tqdm(
            range(BENCH_CONFIG.benchmark_length),
            desc=f"{self.mode.name:<5} | {self.workload.name:<8}",
            leave=True,
            ncols=100,
        ):
            # 1. Action Selection
            action, enum_action = self._select_action(heuristic)

            # 2. Record Structural Actions (Merge/Split/Transfer)
            self._record_structural_actions(enum_action, stats)

            # 3. Environment Step
            if self.mode == BenchMode.RL and self.venv:
                self.obs, _, _, _ = self.venv.step([action])  # type: ignore
            else:
                assert self.env is not None
                self.obs, _, _, _, _ = self.env.step(action)

            # 4. Tick Statistics
            self._tick_step_stats(stats)

    def _select_action(
        self,
        heuristic: Optional[RuleBasedHeuristic],
    ) -> Tuple[int, m.ResourceManagerAction]:
        assert self.env is not None
        mask = self.env.action_masks()
        if self.mode == BenchMode.RL:
            assert self.model is not None
            assert self.venv is not None
            act_np, _ = self.model.predict(  # type: ignore
                self.obs, action_masks=mask, deterministic=True
            )
            action = int(act_np[0])
            return action, list(m.ResourceManagerAction)[action]
        elif self.mode == BenchMode.BASELINE_HEURISTIC and heuristic:
            enum_act = heuristic.decide_action(self.env.sim)
            return list(m.ResourceManagerAction).index(enum_act), enum_act

        no_action = m.ResourceManagerAction.NO_ACTION
        return list(m.ResourceManagerAction).index(no_action), no_action

    def _record_structural_actions(
        self, enum_action: m.ResourceManagerAction, stats: Dict[str, Any]
    ):
        if enum_action != m.ResourceManagerAction.NO_ACTION:
            stats["timeline_actions"].append((self.env.sim.current_time, enum_action))
            act_val = enum_action.value
            match act_val:
                case m.MigAction:
                    if act_val.action == "split":
                        stats["split_count"][act_val.victim] += 1
                    elif act_val.action == "merge":
                        stats["merge_count"][act_val.victim] += 1

                    # If it's a combined action, also count the transfer
                    if act_val.receiver is not None:
                        stats["transfer_count"][act_val.victim] += 1
                case m.VramTransferAction:
                    stats["transfer_count"][act_val.giver] += 1
                case _:
                    raise ValueError(f"Unknown action type: {enum_action}")

    def _tick_step_stats(self, stats: Dict[str, Any]):
        assert self.env is not None
        curr_time = self.env.sim.current_time
        pat_c = self._get_active_pattern(m.AgentId.CODING, curr_time)
        pat_r = self._get_active_pattern(m.AgentId.RAG, curr_time)
        stats["joint_phase_ticks"][pat_c][pat_r] += 1
        stats["timeline_time"].append(curr_time)
        stats["timeline_pattern_coding"].append(pat_c)
        stats["timeline_pattern_rag"].append(pat_r)

        for aid, agent in self.env.sim.agents.items():
            ql_sum = sum(
                len(e.waiting_queue)
                for e in agent.engines
                if e.status != m.EngineStatus.BOOTING
            )
            stats["queue_lengths_sum"][aid] += ql_sum
            for engine in agent.engines:
                if not engine.is_permanent:
                    stats["presence_by_mig"][aid][engine.mig_profile] += 1

        # Accumulate completed requests
        for aid, reqs in self.env.sim.interval_requests.items():
            for req in reqs:
                if req.is_finished and req.serving_engine is not None:
                    stats["completed_reqs_map"][aid][req.id] = req

    def _flush_simulation(
        self,
        stats: Dict[str, Any],
    ) -> Dict[m.AgentId, Dict[str, m.Request]]:
        assert self.env is not None
        print(
            "\nBenchmark steps exhausted. Entering flush period to finish remaining requests..."
        )
        no_action = list(m.ResourceManagerAction).index(
            m.ResourceManagerAction.NO_ACTION
        )
        flush_steps = 0

        with tqdm(
            unit="step",
            leave=True,
            ncols=100,
            bar_format="{desc} | {n_fmt} steps [{elapsed}, {rate_fmt}]",
        ) as pbar:
            while self.env.sim.has_active_work():
                inflight = sum(
                    1
                    for e in self.env.sim.events
                    if e.event_type
                    in [m.EventType.REQUEST_ARRIVAL, m.EventType.RAG_SEARCH_COMPLETE]
                )
                queue_q = sum(
                    len(e.waiting_queue) + len(e.running_queue)
                    for a in self.env.sim.agents.values()
                    for e in a.engines
                )

                pbar.set_description(
                    f"{self.mode.name:<5} | FLUSHING (P: {inflight + queue_q:<4})"
                )

                if self.mode == BenchMode.RL and self.venv:
                    self.obs, _, _, _ = self.venv.step([no_action])  # type: ignore
                else:
                    self.obs, _, _, _, _ = self.env.step(no_action)

                # Record existence during flush
                for aid, agent in self.env.sim.agents.items():
                    for eng in agent.engines:
                        if not eng.is_permanent:
                            stats["presence_by_mig"][aid][eng.mig_profile] += 1

                # Accumulate completions
                for aid, reqs in self.env.sim.interval_requests.items():
                    for req in reqs:
                        if req.is_finished and req.serving_engine is not None:
                            stats["completed_reqs_map"][aid][req.id] = req

                flush_steps += 1
                pbar.update(1)
                if flush_steps > 1000:
                    print(
                        f"\n[Warning] Flush period timed out after {flush_steps} steps."
                    )
                    break

        return stats["completed_reqs_map"]

    def _synthesize_results(
        self,
        completed_reqs: Dict[m.AgentId, Dict[str, m.Request]],
        stats: Dict[str, Any],
    ) -> Dict[str, Dict[str, Any]]:
        presence = stats["presence_by_mig"]
        ql_sum = stats["queue_lengths_sum"]
        patterns = stats["patterns"]
        total_steps = BENCH_CONFIG.benchmark_length

        ttft_list: Dict[m.AgentId, List[float]] = {aid: [] for aid in m.AgentId}
        tpot_list: Dict[m.AgentId, List[float]] = {aid: [] for aid in m.AgentId}
        tokens_by_mig = {
            aid: {
                pc: {pr: {prof: 0 for prof in m.MIGProfile} for pr in patterns}
                for pc in patterns
            }
            for aid in m.AgentId
        }

        for aid, req_map in completed_reqs.items():
            for req in req_map.values():
                ttft = (
                    (req.first_token_time - req.arrival_time)
                    if req.first_token_time
                    else (
                        req.finish_time - req.arrival_time if req.finish_time else 0.0
                    )
                )
                ttft_list[aid].append(ttft)

                if req.generated_tokens > 0 and req.decode_time > 0:
                    tpot_list[aid].append(req.decode_time / req.generated_tokens)

                # Track token migration attribution
                r_pat_c = self._get_active_pattern(m.AgentId.CODING, req.arrival_time)
                r_pat_r = self._get_active_pattern(m.AgentId.RAG, req.arrival_time)
                assert req.serving_engine is not None
                tokens_by_mig[aid][r_pat_c][r_pat_r][
                    req.serving_engine.mig_profile
                ] += req.generated_tokens

        res: Dict[str, Dict[str, Any]] = {}
        total_ticks = max(
            1, sum(sum(stats["joint_phase_ticks"][pc].values()) for pc in patterns)
        )

        for aid in m.AgentId:
            # Token distribution math
            token_mig_percentages = {
                pc: {
                    pr: {
                        k: (
                            v / sum(tokens_by_mig[aid][pc][pr].values()) * 100
                            if sum(tokens_by_mig[aid][pc][pr].values()) > 0
                            else 0
                        )
                        for k, v in tokens_by_mig[aid][pc][pr].items()
                    }
                    for pr in patterns
                }
                for pc in patterns
            }

            overall_tokens = {
                prof: sum(
                    tokens_by_mig[aid][pc][pr][prof]
                    for pc in patterns
                    for pr in patterns
                )
                for prof in m.MIGProfile
            }
            total_gen = sum(overall_tokens.values())
            overall_token_mig_percentages = {
                prof: (v / total_gen * 100)
                for prof, v in overall_tokens.items()
                if total_gen > 0
            }

            res[aid.value] = {
                "ttft_percentiles": np.percentile(
                    ttft_list[aid], [25, 50, 75, 99]
                ).tolist()
                if ttft_list[aid]
                else [0, 0, 0, 0],
                "tpot_quartiles": np.percentile(tpot_list[aid], [25, 50, 75]).tolist()
                if tpot_list[aid]
                else [0, 0, 0],
                "avg_waiting_queue": ql_sum[aid] / total_steps,
                "split_count": stats["split_count"][aid],
                "merge_count": stats["merge_count"][aid],
                "transfer_count": stats["transfer_count"][aid],
                "token_mig_percentages": token_mig_percentages,
                "overall_token_mig_percentages": overall_token_mig_percentages,
                "joint_occurrences": {
                    pc: {
                        pr: (stats["joint_phase_ticks"][pc][pr] / total_ticks * 100)
                        for pr in patterns
                    }
                    for pc in patterns
                },
                "mig_existence_percentages": {
                    prof: (count / total_steps * 100)
                    for prof, count in presence[aid].items()
                },
            }

        return res

    def _plot_timeline(self, stats: Dict[str, Any], ckpt: Optional[Path]):
        run_name = ckpt.stem if ckpt else self.mode.name
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = f"{timestamp}_{run_name}"
        save_dir = Path("fig") / folder_name
        os.makedirs(save_dir, exist_ok=True)

        times = stats["timeline_time"]
        if not times:
            return

        coding_patterns = stats["timeline_pattern_coding"]
        rag_patterns = stats["timeline_pattern_rag"]

        mapping = {"idle": 0, "even": 1, "busy": 2}
        y_coding = [mapping[p] for p in coding_patterns]
        y_rag = [mapping[p] for p in rag_patterns]

        for target_action in ["split", "merge", "transfer"]:
            plt.figure(figsize=(15, 5))
            sns.set_style("whitegrid")

            plt.step(
                times,
                y_coding,
                label="CodingAgent",
                where="post",
                alpha=0.8,
                linewidth=3,
            )
            plt.step(
                times, y_rag, label="RAGAgent", where="post", alpha=0.8, linewidth=3
            )

            for t, enum_action in stats["timeline_actions"]:
                val = enum_action.value
                if isinstance(val, m.MigAction):
                    if val.action == "split" and target_action == "split":
                        plt.axvline(
                            x=t,
                            color="red",
                            linestyle="--",
                            alpha=0.9,
                            linewidth=2,
                            label="Split",
                        )
                    elif val.action == "merge" and target_action == "merge":
                        plt.axvline(
                            x=t,
                            color="blueviolet",
                            linestyle="--",
                            alpha=0.9,
                            linewidth=2,
                            label="Merge",
                        )
                elif isinstance(val, m.VramTransferAction):
                    if target_action == "transfer":
                        plt.axvline(
                            x=t,
                            color="forestgreen",
                            linestyle="--",
                            alpha=0.9,
                            linewidth=2,
                            label="Transfer",
                        )

            plt.yticks(list(mapping.values()), list(mapping.keys()))
            plt.xlabel("Simulation Time (sec)")
            plt.ylabel("Workload Pattern")
            plt.title(
                f"Workload Pattern Timeline ({target_action.capitalize()}) - {run_name}",
                fontweight="bold",
            )

            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            plt.legend(
                by_label.values(),
                by_label.keys(),
                loc="center left",
                bbox_to_anchor=(1, 0.5),
            )
            plt.tight_layout()

            fig_path = save_dir / f"{target_action}.png"
            plt.savefig(fig_path, dpi=300)
            plt.close()

    def _get_active_pattern(self, agent_id: m.AgentId, t: float) -> str:
        for ph in self.phase_history[agent_id]:
            if ph["start_time"] <= t <= ph["start_time"] + ph["duration"] + 1e-5:
                return ph["pattern"]
        last_ph = self.phase_history[agent_id][-1]
        if t > last_ph["start_time"] + last_ph["duration"]:
            return last_ph["pattern"]
        assert False, f"No pattern found for {agent_id} at t={t:.2f}s"

    @staticmethod
    def get_workload_summary(
        phase_history: Dict[m.AgentId, List[PhaseHistoryType]],
    ) -> Dict[m.AgentId, List[Dict[str, Any]]]:
        workload_summary: Dict[m.AgentId, List[Dict[str, Any]]] = {}
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

    @classmethod
    def run_suite(cls):
        args = cls._parse_args()
        bench_modes = cls._resolve_bench_modes(args)

        print(
            "\n" + "=" * 60 + "\nSTARTING HYBRID BENCHMARKS\nUsing seed:",
            BENCH_CONFIG.seed,
            "\n" + "=" * 60,
        )

        # 1. Prepare Workload
        print("Pre-generating workload requests...")
        shared_loader = BenchRequestLoader(Workload.HYBRID, seed=BENCH_CONFIG.seed)
        shared_requests: List[m.Request] = []
        for aid in m.AgentId:
            shared_requests.extend(
                shared_loader.generate_requests(agent_id=aid, turn=0)
            )
        print(f"Generated {len(shared_requests)} requests.")
        print_workloads(cls.get_workload_summary(shared_loader.phase_history))

        # 2. Run sequential trials
        for mode in bench_modes:
            ckpt = Path(args.ckpts.pop()) if mode == BenchMode.RL else None
            print_banner(mode, ckpt.parent.name if ckpt else "")

            if BENCH_CONFIG.phase == TrainingPhase.PHASE_1:
                cls._run_phase_1_matrix(
                    mode, shared_requests, shared_loader.phase_history, ckpt
                )
            else:
                cls._run_phase_2_trial(
                    mode, shared_requests, shared_loader.phase_history, ckpt
                )

        print("\nBenchmark Suite Completed.")

    @classmethod
    def _parse_args(cls) -> argparse.Namespace:
        parser = argparse.ArgumentParser(description="Run Performance Benchmarks")
        parser.add_argument(
            "--ckpts",
            type=Path,
            nargs="+",
            default=None,
            help="Path to RL Checkpoint Zip",
        )
        parser.add_argument(
            "--bl",
            nargs="+",
            default=[],
            help="Baselines to run (e.g., 7g, 2_2_2_1, static, heuristic)",
        )
        args = parser.parse_args()
        args.ckpts = cast(List[str], args.ckpts) if args.ckpts else []
        args.ckpts.reverse()
        return args

    @classmethod
    def _resolve_bench_modes(cls, args: argparse.Namespace) -> List[BenchMode]:
        modes = [BenchMode.RL] * len(args.ckpts)
        mapping = {m.value: m for m in BenchMode}
        for bl in args.bl:
            if bl == "all":
                modes.extend(
                    [BenchMode.BASELINE_STATIC]
                    if BENCH_CONFIG.phase == TrainingPhase.PHASE_1
                    else [
                        BenchMode.BASELINE_7G,
                        BenchMode.BASELINE_2_2_2_1,
                        BenchMode.BASELINE_HEURISTIC,
                    ]
                )
            elif bl in mapping:
                modes.append(mapping[bl])
            else:
                print(f"Unknown baseline: {bl}")
        if not modes:
            raise SystemExit(
                "No benchmark to run. Use --ckpts for RL and --bl for baselines."
            )
        return modes

    @classmethod
    def _run_phase_1_matrix(
        cls,
        mode: BenchMode,
        requests: List[m.Request],
        phase_history: Dict[m.AgentId, List[PhaseHistoryType]],
        ckpt: Optional[Path],
    ):
        init_modes = [
            im for im in m.InitialMIGCombination if im != m.InitialMIGCombination.RANDOM
        ]
        matrix_results: Dict[
            m.InitialMIGCombination,
            Dict[m.InitialMIGCombination, Dict[str, Dict[str, Any]]],
        ] = {}
        for im_coding in init_modes:
            matrix_results[im_coding] = {}
            for im_rag in init_modes:
                runner = cls(
                    ckpt,
                    Workload.HYBRID,
                    mode,
                    requests,
                    phase_history,
                    (im_coding, im_rag),
                )
                matrix_results[im_coding][im_rag] = runner.run()
        print_matrix_metrics(matrix_results)

    @classmethod
    def _run_phase_2_trial(
        cls,
        mode: BenchMode,
        requests: List[m.Request],
        phase_history: Dict[m.AgentId, List[PhaseHistoryType]],
        ckpt: Optional[Path],
    ):
        mapping = {
            BenchMode.BASELINE_7G: m.InitialMIGCombination.C7,
            BenchMode.BASELINE_2_2_2_1: m.InitialMIGCombination.C2_2_2_1,
        }
        init_mode = mapping.get(mode, m.InitialMIGCombination.RANDOM)
        runner = cls(ckpt, Workload.HYBRID, mode, requests, phase_history, init_mode)
        results = runner.run()
        print_metrics(results)


def main():
    BenchRunner.run_suite()


if __name__ == "__main__":
    main()
