"""Evaluate a GO-CART training run's checkpoints and plot performance over training.

Given a training run id and a resolution k, benchmarks every checkpoint whose step
count is a multiple of 5120*k (the trainer saves every 5120 steps) and plots one
performance figure across the run.

Performance is delivered quality over log tail latency:

    performance = model-tier quality score (%)  /  log10(1 + P99 TTFT (s))

Both terms are pooled over every agent's requests rather than averaged per agent.
A per-agent mean would let a low-traffic agent count as much as a busy one; pooling
asks what the system as a whole delivered. The per-agent figures are still printed,
because a ratio hides which of its two terms moved.

Checkpoints are evaluated in parallel, one process each, on the same workload the
other benchmarks use. Each is a full benchmark run, so wall time is roughly
(number of checkpoints / workers) x single-run time.

Usage:
    python -m src.bench.training_curve 20260711-233808-449 -k 10
    python -m src.bench.training_curve <run-id> -k 4 --workers 8
"""

import argparse
import contextlib
import math
import multiprocessing as mp
import os
import random
import re
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tabulate
import torch

import src.share.models as m
from src.bench.main import BenchRunner
from src.bench.models import BenchMode
from src.bench.recovery_compare import (
    POLICY_COLORS,
    SERIF_STACK,
    build_shared_workload,
    request_tier_pct,
)

# The trainer's checkpoint callback fires on this stride; the resolution argument
# is a multiplier on it.
CKPT_INTERVAL = 5120
# Ten concurrent full benchmark runs. The simulator is pure-Python and GIL-bound,
# so this is processes, not threads. Higher is possible on this box (80 cores) but
# each worker also holds its own copy of the ~1M-request workload, and the
# checkpoint's policy has to share the GPUs.
DEFAULT_WORKERS = 10
# The tail statistic. P99 matches what the benchmark tables report, so the curve
# stays commensurable with them.
TAIL_PCTL = 99
# Seeds the global RNG immediately before the runner is built, which pins two
# things the simulator would otherwise draw from wherever the stream happened to
# stop: the starting MIG layout (SimulationConfig.generate_initial_state picks it
# with random.choice) and the per-token latency noise (engine.py's random.gauss).
# Without this every checkpoint starts from a different cluster -- ChatAgent might
# open with two large B200 slices in one run and six tiny ones in the next -- and
# the curve would measure the draw rather than the policy. Pinned after the
# workload is generated, since RequestLoader seeds the same global RNG itself and
# reseeding earlier would change the request stream.
LAYOUT_SEED = 77

POLL_INTERVAL = 2.0
# A non-TTY stdout cannot overwrite a line, so status there is appended rarely.
QUIET_STATUS_INTERVAL = 30.0
# One hue per agent, from the same all-pairs colourblind-safe triple the policy
# comparison uses (worst CVD dE 9.2). Cycles if a deployment ever carries more
# than three agents, which would need a wider validated set.
AGENT_PALETTE = list(POLICY_COLORS.values())
# Agent lines are context, so they recede; the pooled line is the headline. The
# legend carries identity, so the hues only have to stay separable from each
# other, not compete with the aggregate for attention.
AGENT_ALPHA = 0.38
# Magenta, validated all-pairs against the three agent hues (worst CVD dE 9.2, the
# pre-existing green/orange pair; normal-vision floor 20.6). A violet was the first
# choice and had to be rejected -- dE 4.6 against CodingAgent's blue under
# deuteranopia, and 13.4 even with normal colour vision.
POOLED_COLOUR = "#c4399b"
# Marker area on the pooled line, in points^2, spanning the run's action range.
# The smallest still clears the 8px minimum for a readable marker.
MARKER_AREA = (60.0, 300.0)
# Labels are text and wear text ink, not the series colour.
LABEL_INK = "#3a3a38"


class CkptSpec(NamedTuple):
    steps: int
    ckpt: Path
    log_path: Path
    gpu: int
    layout_seed: int


# (gpu, agent, mig profile) for every engine the run starts with.
Layout = Tuple[Tuple[int, str, str], ...]


class Point(NamedTuple):
    """One checkpoint's evaluation."""

    steps: int
    quality: float  # pooled token-weighted model-tier score, %
    tail: float  # pooled P99 TTFT, seconds
    performance: float  # quality / tail
    per_agent: Dict[str, Tuple[float, float]]  # agent -> (quality, tail)
    n_actions: int  # reconfigurations the policy emitted over the run
    layout: Layout  # starting cluster, for cross-checkpoint verification


def find_checkpoints(run_id: str, k: int) -> List[Tuple[int, Path]]:
    """Checkpoints whose step count is a multiple of CKPT_INTERVAL * k.

    Multiples of the stride rather than every k-th file on disk: a resumed run
    starts at whatever step it left off, so counting files would put the sample
    points at arbitrary step numbers and two runs could not be compared.
    """
    ckpt_dir = Path("results") / run_id / "ckpts" / run_id
    if not ckpt_dir.is_dir():
        raise SystemExit(f"No checkpoint directory at {ckpt_dir}")

    stride = CKPT_INTERVAL * k
    found: List[Tuple[int, Path]] = []
    skipped = 0
    for path in ckpt_dir.glob("rl_model_*_steps.zip"):
        match = re.fullmatch(r"rl_model_(\d+)_steps", path.stem)
        if not match or int(match.group(1)) % stride:
            continue
        steps = int(match.group(1))
        # BenchRunner asserts on this file rather than degrading, so a checkpoint
        # missing its normaliser would abort the whole sweep at load time.
        if not path.with_name(f"{path.stem}_vecnormalize.pkl").exists():
            skipped += 1
            continue
        found.append((steps, path))

    if skipped:
        print(f"Skipped {skipped} checkpoint(s) with no _vecnormalize.pkl sibling.")
    if not found:
        raise SystemExit(
            f"No checkpoint in {ckpt_dir} has a step count divisible by {stride:,}. "
            f"Try a smaller -k."
        )
    return sorted(found)


def performance_of(quality: float, tail: float) -> float:
    """Quality score over the log of tail latency.

    The log is on the denominator rather than on the axis: P99 TTFT spans two
    orders of magnitude across checkpoints while the quality score stays inside a
    narrow band, so a plain quality/tail ratio is effectively a picture of 1/TTFT
    with the quality term invisible. Taking the log first puts the two terms on
    comparable footing.

    log1p, not log. The denominator must stay positive and finite for every value
    the benchmark can produce: per-agent P99 TTFTs below one second are routine
    (0.7 s and 0.9 s both appear in this run), where log10 is negative and would
    silently flip the sign of performance, and log10(1.0) is exactly zero.
    """
    denom = math.log10(1.0 + tail)
    return quality / denom if denom > 0 else 0.0


def ttft_of(r: m.Request) -> float:
    """Time to first token, matching how the benchmark tables compute it."""
    if r.first_token_time:
        return r.first_token_time - r.arrival_time
    return r.finish_time - r.arrival_time if r.finish_time else 0.0


def measure(steps: int, runner: BenchRunner, layout: Layout = ()) -> Point:
    """Pooled quality and tail latency for one evaluated checkpoint.

    Quality is Equation 5.1 evaluated over the whole run: the token-weighted mean
    of each request's model tier as a share of the platform's top tier.
    """
    pooled_ttft: List[float] = []
    pooled_num = pooled_den = 0.0
    per_agent: Dict[str, Tuple[float, float]] = {}

    for aid, req_map in runner.completed_reqs.items():
        ttfts: List[float] = []
        num = den = 0.0
        for r in req_map.values():
            ttfts.append(ttft_of(r))
            pct = request_tier_pct(r)
            if pct is not None:
                num += pct * r.generated_tokens
                den += r.generated_tokens
        pooled_ttft.extend(ttfts)
        pooled_num += num
        pooled_den += den
        per_agent[aid.value] = (
            num / den if den else 0.0,
            float(np.percentile(ttfts, TAIL_PCTL)) if ttfts else 0.0,
        )

    quality = pooled_num / pooled_den if pooled_den else 0.0
    tail = float(np.percentile(pooled_ttft, TAIL_PCTL)) if pooled_ttft else 0.0
    return Point(
        steps=steps,
        quality=quality,
        tail=tail,
        performance=performance_of(quality, tail),
        per_agent=per_agent,
        n_actions=len(runner.stats["timeline_actions"]),
        layout=layout,
    )


def evaluate_checkpoint(spec: CkptSpec) -> Point:
    """Benchmark one checkpoint end to end. Runs in its own process."""
    # Pin the GPU before torch creates a CUDA context. CUDA_VISIBLE_DEVICES is
    # read at first context creation, which has not happened yet in a freshly
    # spawned process -- without this every worker lands on cuda:0 and stacks ten
    # contexts on one card.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(spec.gpu)

    def note(msg: str) -> None:
        with open(spec.log_path, "a") as fh:
            fh.write(msg + "\n")

    import src.simulation.utils as u

    note("loading config")
    u.SIM_CONFIG = u.init_config(Path("."))
    u.TOKENS_MAP = u.init_tokens_map(Path("."), u.SIM_CONFIG)
    torch.distributions.Distribution.set_default_validate_args(False)

    # Regenerated from the shared seed rather than shipped in: the stream is
    # deterministic, so every checkpoint faces an identical workload without
    # pickling ~1M request objects to each of ten workers.
    note("building workload")
    requests, phase_history = build_shared_workload()

    # Pin here, not earlier: RequestLoader seeds this same global RNG per agent
    # while generating the stream above, so seeding before it would change the
    # workload. Constructing the runner resets the env, which is what draws the
    # starting layout.
    note("pinning layout")
    random.seed(spec.layout_seed)
    runner = BenchRunner(spec.ckpt, BenchMode.RL, requests, phase_history)
    layout: Layout = tuple(
        sorted((e["gpu"], e["agent"], e["mig"]) for e in u.SIM_CONFIG.initial_state)
    )

    # stderr as well as stdout: tqdm writes there, and ten concurrent bars would
    # be unreadable. The parent tails these logs for progress instead.
    with open(spec.log_path, "a", buffering=1) as log:
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            runner.run()

    return measure(spec.steps, runner, layout)


def step_progress(log_path: Path) -> Optional[Tuple[int, int]]:
    """Parse a worker's newest tqdm frame into (done, total), or None."""
    try:
        with open(log_path, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            fh.seek(max(0, fh.tell() - 4096))
            tail = fh.read().decode("utf-8", "replace")
    except OSError:
        return None
    frame = next(
        (c.strip() for c in reversed(re.split(r"[\r\n]", tail)) if c.strip()), ""
    )
    match = re.search(r"(\d+)/(\d+) \[", frame)
    return (int(match.group(1)), int(match.group(2))) if match else None


def run_parallel(specs: List[CkptSpec], workers: int) -> List[Point]:
    """Evaluate every checkpoint, at most `workers` at a time, reporting progress."""
    points: List[Point] = []
    failed: Dict[int, str] = {}
    live = sys.stdout.isatty()
    t0 = time.perf_counter()
    last_status = 0.0
    running: Dict[int, Path] = {}

    def render(final: bool = False) -> None:
        nonlocal last_status
        now = time.perf_counter()
        if not final and not live and now - last_status < QUIET_STATUS_INTERVAL:
            return
        last_status = now
        # Ten per-worker bars will not fit a terminal line, so report the
        # aggregate and the straggler -- the straggler is what sets the wall time.
        seen = [step_progress(p) for p in running.values()]
        done_steps = [s for s in seen if s]
        slowest = (
            f"slowest {min(d for d, _ in done_steps)}/{done_steps[0][1]}"
            if done_steps
            else "starting"
        )
        el = now - t0
        line = (
            f"  [{int(el // 60):02d}:{el % 60:04.1f}]  "
            f"{len(points) + len(failed)}/{len(specs)} evaluated | "
            f"{len(running)} running, {slowest}"
        )
        print(
            "\r" + line.ljust(96)[:96] if live else line,
            end="" if live else "\n",
            flush=True,
        )

    def clear() -> None:
        if live:
            print("\r" + " " * 96 + "\r", end="", flush=True)

    # spawn rather than fork: torch is imported in the parent, and forking a
    # process with CUDA state initialised is unsafe.
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
        futures = {pool.submit(evaluate_checkpoint, s): s for s in specs}
        running = {s.steps: s.log_path for s in specs}
        pending = set(futures)
        while pending:
            finished, pending = wait(
                pending, timeout=POLL_INTERVAL, return_when=FIRST_COMPLETED
            )
            for fut in finished:
                spec = futures[fut]
                running.pop(spec.steps, None)
                clear()
                try:
                    point = fut.result()
                except Exception as exc:  # noqa: BLE001 - one bad ckpt must not sink the sweep
                    failed[spec.steps] = f"{type(exc).__name__}: {exc}"
                    print(
                        f"  {spec.steps:,} FAILED: {failed[spec.steps]}  "
                        f"(see {spec.log_path})"
                    )
                    continue
                points.append(point)
                print(
                    f"  {point.steps:>10,} steps  "
                    f"quality {point.quality:6.2f}%  "
                    f"P{TAIL_PCTL} TTFT {point.tail:8.2f}s  "
                    f"performance {point.performance:7.3f}"
                )
            render(final=bool(finished))

    clear()
    print(
        f"  {len(points)}/{len(specs)} checkpoints evaluated in "
        f"{time.perf_counter() - t0:.0f}s"
    )
    # Explicit key: Point's later fields include a dict, which has no ordering.
    return sorted(points, key=lambda p: p.steps)


def formal_name(agent: str) -> str:
    """CodingAgent -> Coding Agent, RAGAgent -> RAG Agent.

    The enum values are code identifiers; a figure caption should not be.
    """
    return re.sub(r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", " ", agent)


def plot_curve(points: List[Point], save_path: Path) -> None:
    """Performance against training progress."""
    sns.set_style("whitegrid")
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = SERIF_STACK
    plt.rcParams["mathtext.fontset"] = "stix"

    fig, ax = plt.subplots(figsize=(11, 6), layout="constrained")
    # Millions on the axis: raw step counts run to seven digits and would either
    # overlap or force scientific notation. Exact values are in the table.
    xs = [p.steps / 1e6 for p in points]

    agents = [a.value for a in m.AgentId if a.value in points[0].per_agent]
    for i, agent in enumerate(agents):
        ax.plot(
            xs,
            [performance_of(*p.per_agent[agent]) for p in points],
            color=AGENT_PALETTE[i % len(AGENT_PALETTE)],
            alpha=AGENT_ALPHA,
            linewidth=1.6,
            marker="o",
            markersize=3,
            label=formal_name(agent),
        )

    pooled = [p.performance for p in points]
    ax.plot(
        xs, pooled, color=POOLED_COLOUR, linewidth=3.0, zorder=5,
        label="All Agents (pooled)",
    )
    # Marker area carries the action count as well as the printed number --
    # redundant encoding, so the figure still reads if the labels are too small to
    # resolve at print size. Scaled linearly across this run's own range; a run
    # where every checkpoint emits the same count falls back to one mid size.
    counts = [p.n_actions for p in points]
    lo, hi = min(counts), max(counts)
    smin, smax = MARKER_AREA
    sizes = (
        [smin + (c - lo) / (hi - lo) * (smax - smin) for c in counts]
        if hi > lo
        else [(smin + smax) / 2] * len(counts)
    )
    ax.scatter(
        xs, pooled, s=sizes, color=POOLED_COLOUR, zorder=6,
        edgecolors="white", linewidths=0.8,
    )
    ax.legend(loc="best", frameon=True)

    # Linear axis: the log lives in the metric itself (see performance_of), so
    # log-scaling the axis on top of it would compress the range twice.
    # Mathtext for the log, so the subscript sets properly in the serif face
    # rather than reading as the string "log10".
    ax.set_ylabel(
        "Model-Tier Quality Score (%) / "
        rf"$\log_{{10}}(1 + \mathrm{{P{TAIL_PCTL}}}\ \mathrm{{TTFT}})$"
    )
    ax.set_ylim(bottom=0)
    # Anchor at zero so the curve sits against the full extent of training and
    # figures drawn at different -k values share one x range.
    ax.set_xlim(left=0)

    # Action counts ride under their own point on the pooled line, so each number
    # sits next to the value it explains. Offset in points rather than data units,
    # so the gap stays constant regardless of where the log axis puts the point.
    for x, p in zip(xs, points):
        ax.annotate(
            f"{p.n_actions}",
            xy=(x, p.performance),
            xytext=(0, -13),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=10,
            fontweight="bold",
            color=LABEL_INK,
            # The pooled line often runs inside the bundle of agent lines, so the
            # label lands on top of one of them; a backing box keeps it readable
            # without moving it away from the point it belongs to.
            bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none",
                  "alpha": 0.75},
        )
    ax.set_xlabel(
        "Training Timesteps (millions)\n"
        "Marker size and label on the aggregate series denote the total number "
        "of reconfiguration actions emitted"
    )
    fig.suptitle("Quality-Latency Performance Across Training")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"\nSaved training curve to {save_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot GO-CART performance across a training run's checkpoints"
    )
    parser.add_argument(
        "run_id",
        help="Training run id under results/, e.g. 20260711-233808-449",
    )
    parser.add_argument(
        "-k",
        "--resolution",
        type=int,
        default=10,
        help=f"Evaluate checkpoints at multiples of {CKPT_INTERVAL} * k steps "
        f"(default: 10)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Checkpoints evaluated concurrently (default: {DEFAULT_WORKERS})",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output figure path (default: results/<run-id>/bench/training_curve.png)",
    )
    parser.add_argument(
        "--layout-seed",
        type=int,
        default=LAYOUT_SEED,
        help=f"Seed fixing the starting MIG layout and latency noise every "
        f"checkpoint is evaluated against (default: {LAYOUT_SEED}). Change it to "
        f"re-run the same sweep against a different starting cluster.",
    )
    return parser.parse_args()


def main() -> None:
    torch.distributions.Distribution.set_default_validate_args(False)
    args = parse_args()
    if args.resolution < 1:
        raise SystemExit("-k must be at least 1.")

    checkpoints = find_checkpoints(args.run_id, args.resolution)
    out = args.out or Path("results") / args.run_id / "bench" / "training_curve.png"
    log_dir = out.parent / "training_curve_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    n_gpus = max(torch.cuda.device_count(), 1)
    specs = [
        CkptSpec(
            steps=steps,
            ckpt=path,
            log_path=log_dir / f"{steps}.log",
            gpu=i % n_gpus,
            layout_seed=args.layout_seed,
        )
        for i, (steps, path) in enumerate(checkpoints)
    ]
    for spec in specs:
        spec.log_path.unlink(missing_ok=True)

    workers = min(args.workers, len(specs))
    print(
        f"Evaluating {len(specs)} checkpoint(s) from run {args.run_id} "
        f"(every {CKPT_INTERVAL * args.resolution:,} steps, "
        f"{checkpoints[0][0]:,} to {checkpoints[-1][0]:,}).\n"
        f"{workers} at a time across {n_gpus} GPU(s), "
        f"starting layout pinned by seed {args.layout_seed}; "
        f"per-checkpoint output: {log_dir}/<steps>.log"
    )

    points = run_parallel(specs, workers)
    if not points:
        raise SystemExit("Every checkpoint failed; nothing to plot.")

    # The pin is the premise of the whole comparison, so verify rather than assume.
    # Any disagreement means checkpoints faced different hardware and their
    # performance numbers are not comparable.
    layouts = {p.layout for p in points}
    if len(layouts) > 1:
        raise SystemExit(
            f"Starting layout differed across checkpoints ({len(layouts)} distinct); "
            "the comparison would be meaningless. This should not happen with a "
            "pinned seed -- please report it."
        )
    layout = points[0].layout
    by_agent: Dict[str, List[str]] = {}
    for _, agent, mig in layout:
        by_agent.setdefault(agent, []).append(mig)
    print(
        f"\n● Starting Cluster (seed {args.layout_seed}, identical for all "
        f"{len(points)} checkpoints)"
    )
    for agent, migs in sorted(by_agent.items()):
        counts = ", ".join(
            f"{migs.count(p)}x {p}" for p in sorted(set(migs), reverse=True)
        )
        print(f"  {agent:<12} {counts}")

    agents = [aid.value for aid in m.AgentId]
    table = [
        [
            f"{p.steps:,}",
            f"{p.quality:.2f}",
            f"{p.tail:.2f}",
            f"{p.performance:.3f}",
            p.n_actions,
            *[
                f"{p.per_agent[a][0]:.1f} / {p.per_agent[a][1]:.1f}"
                if a in p.per_agent
                else "n/a"
                for a in agents
            ],
        ]
        for p in points
    ]
    print(
        f"\n● Checkpoint Evaluation"
        f"\n  Performance = model-tier quality score / log10(1 + P{TAIL_PCTL} TTFT), "
        f"pooled over all agents."
    )
    print(
        tabulate.tabulate(
            table,
            headers=[
                "Steps",
                "Quality (%)",
                f"P{TAIL_PCTL} TTFT (s)",
                "Performance",
                "Actions",
                *[f"{a}\nquality / tail" for a in agents],
            ],
            tablefmt="fancy_grid",
        )
    )

    best = max(points, key=lambda p: p.performance)
    print(
        f"\nBest checkpoint: {best.steps:,} steps (performance {best.performance:.3f})"
    )

    plot_curve(points, out)


if __name__ == "__main__":
    main()
