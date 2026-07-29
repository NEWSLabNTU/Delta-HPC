"""Compare reconfiguration impact across policies on one shared workload.

Runs GO-CART, the HPA rule-based heuristic and QAS over an identical request
stream, then draws three figures per policy showing how the waiting-request
backlog, latency and model-tier quality respond in a window around each
reconfiguration action: one over every attributable episode, and one for each
direction of capacity change (see EPISODE_CATEGORIES). Nine figures per run, all
on shared latency and backlog axes. Episode accounting is printed to the
terminal.

The policies run concurrently, one process each. Threads would not help -- the
simulator is a pure-Python event loop and holds the GIL throughout.

The table and the figures run two separate passes over the same episodes, and
they exclude on different terms. Only the table's set is fully attributable:

  - The table (BenchRunner._compute_reconfig_recovery) bounds each episode at the
    next trigger for that agent, and discards episodes spanning a
    workload-pattern change -- the queue there moves for reasons unrelated to the
    reconfiguration -- as well as those overlapping another one. Both discard
    counts are printed so the volume stays visible. Episodes that never cleared
    inside the horizon are a *result*, not a measurement failure, so they are
    reported as each policy's failure rate rather than dropped silently.
  - The figures (action_windows) discard far less: only episodes interrupted
    before their boot completed, and those with no completed request in the
    window. Everything else inside the window is drawn as it happened, including
    a later action on the same agent, the tail of the previous one, and any
    pattern change. At a 120s decision interval the 300s post-window spans two
    further decision points, so a frequently-acting policy's line carries its own
    follow-up actions. Read the first decision interval as one action's cost;
    past that the trace is how a policy behaves after acting, not how one action
    recovers.

Usage:
    python -m src.bench.recovery_compare --ckpt results/<run>/ckpt/<model>.zip
    python -m src.bench.recovery_compare                     # baselines only
"""

import argparse
import bisect
import contextlib
import multiprocessing as mp
import os
import re
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
import seaborn as sns
import tabulate
import torch

import src.share.models as m
import src.simulation.utils as sim_utils
from src.bench.config import BENCH_CONFIG
from src.bench.main import RECOVERY_HORIZON, BenchRunner
from src.bench.models import BenchMode, Workload
from src.share.request_loader import RequestLoader

# One validated categorical hue per policy. Fixed by name so a colour keeps its
# meaning when a policy is absent from a run. Verified all-pairs colourblind-safe
# (worst CVD dE 9.2) -- matplotlib's default blue/orange/green is not, its orange
# and green being dE 0.7 apart under protanopia.
POLICY_COLORS = {
    "GO-CART": "#2a78d6",
    "HPA": "#eb6834",
    "QAS": "#1baf7a",
}

# Serif fallbacks after Times New Roman are metric-compatible clones, so the
# figure still renders in the intended face on a machine without it rather than
# silently dropping to matplotlib's sans-serif default.
SERIF_STACK = ["Times New Roman", "Liberation Serif", "Nimbus Roman", "DejaVu Serif"]

# Each worker's stdout/stderr goes to its own file here. Three concurrent tqdm
# bars written straight to the terminal interleave into noise, but silencing them
# outright leaves the run with no sign of life for the seven minutes a policy
# takes. The parent tails these instead and renders one consolidated line.
LOG_DIR = Path("results/recovery_compare/logs")
POLL_INTERVAL = 2.0
# A non-TTY stdout (redirected to a file, or under nohup) cannot overwrite a line,
# so status there is appended at a much coarser cadence.
QUIET_STATUS_INTERVAL = 30.0

# --- Action-impact window (per-policy TTFT / quality traces) ---
IMPACT_PRE = 60.0  # seconds of context before the action fires
IMPACT_POST = 500.0  # seconds tracked afterwards
IMPACT_BIN = 15.0  # bin width for the per-window series
# Below this the TTFT axis is linear, above it logarithmic. 0.1s puts the dense
# sub-second baseline into the log region instead of squashing it flat against
# zero, while the congested bins stay on screen.
IMPACT_LINTHRESH = 0.1
# A single reconfiguration's line is a per-bin statistic over whatever requests
# happened to arrive in those 15 seconds. Below this count the bin is left as a
# gap rather than drawn from two or three requests.
IMPACT_MIN_BIN_REQ = 5
# The percentile both TTFT lines report. The median is robust to a single
# congested reconfiguration dominating a bin -- at ~4,500 requests per bin the P99
# is only the ~45th largest value, few enough that one bad episode can supply all
# of them and set the line for the whole panel. The cost is sensitivity: most
# requests are served promptly whatever the load, so the median moves far less
# than the disruption does. Raising this to 90 keeps the robustness (the ~450th
# largest value cannot come from one episode) while restoring tail signal.
IMPACT_PCTL = 50
PCTL_LABEL = "Median" if IMPACT_PCTL == 50 else f"P{IMPACT_PCTL}"
# Extra pooled lines up the tail, so the robust trend and the tail behaviour sit
# in one panel. Pooled only, never per-episode: over a single reconfiguration's
# ~180 requests in a bin a "P99" is its top two values, whereas pooled over ~4,500
# it is a real percentile. Style, not hue, separates them -- the colour already
# encodes the policy.
IMPACT_POOLED = (
    # percentile, linestyle, linewidth
    (IMPACT_PCTL, "-", 2.5),
    (75, "--", 1.8),
    (90, ":", 1.8),
)

# The queue panel is sampled state, not per-request observations: the simulator
# records one (time, waiting requests) point per agent roughly every second, so a
# bin holds ~15 samples of a step function rather than a noisy statistic over
# arrivals. One sample is therefore already a valid reading of the backlog, and
# bins are only left empty when the trace genuinely has nothing there.
QUEUE_MIN_BIN_SAMPLES = 1
# Backlogs span zero to thousands across policies on a shared axis, so the queue
# panel is symlog like the TTFT one. Linear below one request, where the only
# meaningful value is an empty queue.
QUEUE_LINTHRESH = 1.0

# Model-tier assignment for the quality score (thesis Table 5.1). Profiles hosting
# the same model share a tier; the score is the tier-weighted token share,
# normalised by the platform's top tier (Equation 5.1):
#     Quality Score = sum_t (t * s_t) / T * 100%
# where s_t is the share of an agent's tokens generated on tier-t profiles.
# This is deliberately not Q_f: Q_f is a training hyperparameter shaping policy
# behaviour, whereas this score is the reported experimental quality measure, so
# using it keeps these plots commensurable with the latency-quality scatters.
MODEL_TIERS: Dict[str, Dict[str, int]] = {
    "A100_40GB": {
        "1g.10gb": 1,
        "2g.10gb": 1,  # 3B
        "3g.20gb": 2,
        "4g.20gb": 2,  # 7B
        "7g.40gb": 3,  # 14B
    },
    "B200": {
        "1g.23gb": 1,  # 7B
        "1g.45gb": 2,
        "2g.45gb": 2,  # 14B
        "3g.90gb": 3,
        "4g.90gb": 3,  # 80B FP8
        "7g.180gb": 4,  # 80B
    },
}
MAX_TIER: Dict[str, int] = {"A100_40GB": 3, "B200": 4}

# Episodes are also plotted split by the direction of the capacity change the
# action makes *for the agent the episode belongs to*. Splitting a MIG or
# receiving one from the other agent adds serving capacity, which is how a policy
# answers rising demand; merging or giving one away removes it, which is what it
# does when demand falls. The two directions cost different things -- a split
# pays boot time to gain throughput, a merge gives throughput up for tier -- so
# pooling them hides both.
#
# A transfer lands in both categories, once per side: the giver's episode is a
# decrease and the receiver's an increase. They are different agents with
# different queues, so this is two observations, not one counted twice.
# (key, filename suffix, title clause)
EPISODE_CATEGORIES = (
    ("up", "load_up", "Capacity-Adding Actions: Split / Receive"),
    ("down", "load_down", "Capacity-Releasing Actions: Merge / Give"),
)

PolicySpec = Tuple[str, BenchMode, Optional[Path]]
RecoveryByAgent = Dict[m.AgentId, Dict[str, Any]]


def episode_category(ep: Any) -> Optional[str]:
    """Which EPISODE_CATEGORIES key an episode belongs to, or None if neither.

    None is unreachable for the three action types the simulator emits today; it
    exists so a new one shows up as an uncategorised count rather than being
    silently filed under a direction it does not have.
    """
    match ep.action_type:
        case "split":
            return "up"
        case "merge":
            return "down"
        case "transfer":
            return "up" if ep.is_receiver else "down"
    return None


def build_shared_workload() -> Tuple[List[m.Request], Dict[m.AgentId, Any]]:
    """Generate one request stream for every policy to be measured against."""
    loader = RequestLoader(
        num_steps=BENCH_CONFIG.benchmark_length,
        get_rate_range=lambda p, a: BENCH_CONFIG.get_rate_range(Workload(p), a),
        get_duration_range=lambda p: BENCH_CONFIG.get_duration_range(Workload(p)),
        dataset_paths=sim_utils.SIM_CONFIG.datasets,
        seed=BENCH_CONFIG.seed,
        track_history=True,
        workload_sequence=BENCH_CONFIG.workload_sequence,
    )
    requests: List[m.Request] = []
    for aid in m.AgentId:
        requests.extend(loader.generate_requests(agent_id=aid, turn=0))
    return requests, loader.phase_history


def run_policy(spec: PolicySpec) -> Tuple[str, RecoveryByAgent, "ImpactData"]:
    """Run one policy end to end. Executed in its own process, one per policy.

    The workload is regenerated here from the shared seed rather than passed in:
    the stream is deterministic, and shipping ~800k request objects through a
    pickle to each worker would cost more than the parallelism saves. Each process
    also gets its own simulator-config globals, so the reload the sequential path
    needed to stop profiling state leaking between trials is automatic here.

    Only picklable results cross the process boundary -- the statistics dict and
    the per-request arrays. The runner itself, holding the simulator and engines,
    stays in the worker and is discarded with it.
    """
    name, mode, ckpt = spec

    # Setup runs for the best part of a minute before the first progress bar
    # appears -- generating ~800k requests dominates. Breadcrumb each stage into
    # the log so the parent can name what is happening instead of showing a
    # motionless "starting" for the whole warm-up.
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{name.lower()}.log"

    def note(msg: str) -> None:
        with open(log_path, "a") as fh:
            fh.write(msg + "\n")

    import src.simulation.utils as u

    note("loading config")
    u.SIM_CONFIG = u.init_config(Path("."))
    u.TOKENS_MAP = u.init_tokens_map(Path("."), u.SIM_CONFIG)
    torch.distributions.Distribution.set_default_validate_args(False)

    note("building workload")
    requests, phase_history = build_shared_workload()
    note("starting runner")
    runner = BenchRunner(ckpt, mode, requests, phase_history)

    # stderr as well as stdout: tqdm writes there. Both go to this policy's own
    # log so the bars stay readable and the parent can tail them for progress.
    with open(log_path, "a", buffering=1) as log:
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            results = runner.run()

    recovery = {aid: results[aid.value]["reconfig_recovery"] for aid in m.AgentId}
    return name, recovery, action_windows(runner)


def request_tier_pct(r: m.Request) -> Optional[float]:
    """A single request's tier as a percentage of the platform's top tier.

    The thesis quality score (Eq. 5.1) is the token-weighted mean of this
    quantity: sum_t t*s_t / T is exactly sum_requests (tier_r/T)*(tokens_r/total).
    Keeping it per-request lets the figure scatter the raw observations and draw
    the score itself as their weighted mean, rather than plotting a summary whose
    relationship to the underlying data the reader has to take on trust.
    """
    eng = r.serving_engine
    if eng is None or r.generated_tokens <= 0:
        return None
    profile = eng.mig_profile
    tier = MODEL_TIERS.get(profile.gpu_model, {}).get(profile.string)
    if tier is None:
        return None  # profile outside the tier table (e.g. A100 1g.5gb)
    return tier / MAX_TIER[profile.gpu_model] * 100.0


class ImpactData(NamedTuple):
    """Per-request observations gathered around a policy's reconfigurations."""

    centres: List[float]  # bin centres for the trend lines
    rel: np.ndarray  # per-request offset from its action, pooled
    ttft: np.ndarray  # per-request TTFT, aligned with `rel`
    ep: np.ndarray  # which reconfiguration each request belongs to
    # Quality arrays cover only requests that generated tokens on a tiered
    # profile, so they are shorter than `rel` and carry their own offsets.
    q_rel: np.ndarray
    q_pct: np.ndarray  # tier / T * 100
    q_tokens: np.ndarray  # weights for the score (Eq. 5.1 is token-weighted)
    q_ep: np.ndarray
    # Queue arrays are samples of the agent's backlog rather than per-request
    # values, so they too carry their own offsets and episode index.
    qlen_rel: np.ndarray
    qlen: np.ndarray  # waiting requests, summed over the agent's engines
    qlen_ep: np.ndarray
    # One EPISODE_CATEGORIES key (or "") per episode, indexed by episode number,
    # so the per-category figures can be cut from this without a second pass over
    # the simulator.
    ep_cat: np.ndarray
    n_episodes: int


def action_windows(runner: BenchRunner) -> ImpactData:
    """Collect per-request TTFT, quality and backlog traces around each trigger.

    The panels are indexed by different clocks, deliberately. Latency is a
    property of a request's whole journey, so it belongs at its arrival. Quality
    is fixed by the engine that served it, so it belongs at the moment service
    began -- binning it by arrival credits a post-reconfiguration tier to a
    pre-action bin whenever the request sat in the queue across the trigger, and
    under HPA (pre-action P99 TTFT ~10^3 s) that reaches back tens of
    seconds and makes quality appear to fall before the action that caused it.
    Queue length has no such ambiguity: it is agent state sampled by the
    simulator, already stamped with the time it was observed.

    Only two things are discarded here, both below: an episode interrupted before
    it finished booting, and one with no completed request in its window. Neither
    the next-trigger bound nor the pattern-shift check the accounting table
    applies is repeated -- see the module docstring for what that leaves in the
    traces.
    """
    edges = np.arange(-IMPACT_PRE, IMPACT_POST + IMPACT_BIN, IMPACT_BIN)
    centres = ((edges[:-1] + edges[1:]) / 2).tolist()

    by_agent: Dict[m.AgentId, List[m.Request]] = {}
    for aid, req_map in runner.completed_reqs.items():
        reqs = [r for r in req_map.values() if r.serving_engine is not None]
        reqs.sort(key=lambda r: r.arrival_time)
        by_agent[aid] = reqs
    arrivals = {aid: [r.arrival_time for r in rs] for aid, rs in by_agent.items()}

    # A second index, ordered by first-token time, for the quality panel. Widening
    # the arrival scan and re-filtering would be the obvious alternative, but the
    # queue wait has no bound to widen by: a request arriving well before
    # t_trigger - IMPACT_PRE can still take its first token inside the window, and
    # those are precisely the ones a reconfiguration delayed.
    served: Dict[m.AgentId, List[m.Request]] = {}
    for aid, reqs in by_agent.items():
        with_ftt = [r for r in reqs if r.first_token_time is not None]
        with_ftt.sort(key=lambda r: r.first_token_time)
        served[aid] = with_ftt
    first_tokens = {aid: [r.first_token_time for r in rs] for aid, rs in served.items()}

    # Waiting-request samples, one series per agent, already time-ordered by the
    # simulator. Sliced by the same bisect scheme as the request indices above.
    queue_trace = runner.env.sim.queue_trace
    queue_times = {aid: [t for t, _ in queue_trace.get(aid, [])] for aid in m.AgentId}

    def ttft_of(r: m.Request) -> float:
        if r.first_token_time:
            return r.first_token_time - r.arrival_time
        return r.finish_time - r.arrival_time if r.finish_time else 0.0

    rel_all: List[float] = []
    ttft_all: List[float] = []
    ep_all: List[int] = []
    q_rel: List[float] = []
    q_pct: List[float] = []
    q_tok: List[float] = []
    q_ep: List[int] = []
    ql_rel: List[float] = []
    ql_val: List[float] = []
    ql_ep: List[int] = []
    cats: List[str] = []
    n_episodes = 0

    for ep in runner.env.sim.reconfig_episodes:
        # A second action arriving mid-reconfiguration puts two disturbances in
        # one trace before this one has even finished. Note the narrow scope: the
        # flag is set only while the episode is open, so it catches nothing once
        # boot completes, ~60-70s into a 360s window. t_boot_done is None covers
        # episodes still open at the end of the run, which includes those whose
        # action never reached the agent at all (see seen_activity).
        if ep.interrupted or ep.t_boot_done is None:
            continue

        aid = ep.agent_id
        lo = bisect.bisect_left(arrivals[aid], ep.t_trigger - IMPACT_PRE)
        hi = bisect.bisect_right(arrivals[aid], ep.t_trigger + IMPACT_POST)
        window = by_agent[aid][lo:hi]
        if not window:
            # No completed request arrived in the window, so there is no latency
            # trace to draw. This drops the episode's quality and queue samples
            # too, which is the intended reading: an agent nothing arrived at was
            # not measurably disturbed by the action.
            continue

        for r in window:
            rel_all.append(r.arrival_time - ep.t_trigger)
            ttft_all.append(ttft_of(r))
            ep_all.append(n_episodes)

        q_lo = bisect.bisect_left(first_tokens[aid], ep.t_trigger - IMPACT_PRE)
        q_hi = bisect.bisect_right(first_tokens[aid], ep.t_trigger + IMPACT_POST)
        for r in served[aid][q_lo:q_hi]:
            pct = request_tier_pct(r)
            if pct is not None:
                q_rel.append(r.first_token_time - ep.t_trigger)
                q_pct.append(pct)
                q_tok.append(float(r.generated_tokens))
                q_ep.append(n_episodes)

        ql_lo = bisect.bisect_left(queue_times[aid], ep.t_trigger - IMPACT_PRE)
        ql_hi = bisect.bisect_right(queue_times[aid], ep.t_trigger + IMPACT_POST)
        for t, q in queue_trace.get(aid, [])[ql_lo:ql_hi]:
            ql_rel.append(t - ep.t_trigger)
            ql_val.append(float(q))
            ql_ep.append(n_episodes)

        cats.append(episode_category(ep) or "")
        n_episodes += 1

    return ImpactData(
        centres=centres,
        rel=np.asarray(rel_all),
        ttft=np.asarray(ttft_all),
        ep=np.asarray(ep_all, dtype=np.int64),
        q_rel=np.asarray(q_rel),
        q_pct=np.asarray(q_pct),
        q_tokens=np.asarray(q_tok),
        q_ep=np.asarray(q_ep, dtype=np.int64),
        qlen_rel=np.asarray(ql_rel),
        qlen=np.asarray(ql_val),
        qlen_ep=np.asarray(ql_ep, dtype=np.int64),
        ep_cat=np.asarray(cats, dtype=object),
        n_episodes=n_episodes,
    )


def category_subset(data: ImpactData, category: str) -> ImpactData:
    """The episodes of one category, with episode numbering closed up.

    Every per-episode row in the plots is indexed by position, so the surviving
    episodes have to be renumbered 0..k-1 rather than keeping their original
    numbers -- otherwise the per-episode matrices would be mostly empty rows and
    `faint` would fade the lines for episodes that are no longer there.
    """
    keep_ep = data.ep_cat == category
    n_kept = int(keep_ep.sum())
    renumber = np.full(data.n_episodes, -1, dtype=np.int64)
    renumber[keep_ep] = np.arange(n_kept)

    def take(ep: np.ndarray, *arrays: np.ndarray) -> Tuple[np.ndarray, ...]:
        if ep.size == 0:
            return (*arrays, ep)
        sel = keep_ep[ep]
        return (*(a[sel] for a in arrays), renumber[ep[sel]])

    rel, ttft, ep = take(data.ep, data.rel, data.ttft)
    q_rel, q_pct, q_tokens, q_ep = take(
        data.q_ep, data.q_rel, data.q_pct, data.q_tokens
    )
    qlen_rel, qlen, qlen_ep = take(data.qlen_ep, data.qlen_rel, data.qlen)

    return ImpactData(
        centres=data.centres,
        rel=rel,
        ttft=ttft,
        ep=ep,
        q_rel=q_rel,
        q_pct=q_pct,
        q_tokens=q_tokens,
        q_ep=q_ep,
        qlen_rel=qlen_rel,
        qlen=qlen,
        qlen_ep=qlen_ep,
        ep_cat=data.ep_cat[keep_ep],
        n_episodes=n_kept,
    )


def summarize(collected: Dict[str, RecoveryByAgent]) -> List[List[Any]]:
    """Per-agent episode accounting: recovered, failed, and why episodes dropped."""
    audit: List[List[Any]] = []
    for policy, by_agent in collected.items():
        for aid in m.AgentId:
            rec = by_agent[aid]
            cleared = rec["n_analyzed"]
            failed = rec["n_uncleared"]
            audit.append([
                policy,
                aid.value,
                cleared,
                failed,
                f"{failed / (cleared + failed) * 100:.1f}%"
                if cleared + failed
                else "n/a",
                rec["n_pattern_shift"],
                rec["n_overlapping"],
                rec["n_episodes"],
            ])
    return audit


def episode_percentile(
    ep: np.ndarray,
    idx: np.ndarray,
    values: np.ndarray,
    n_ep: int,
    n_bins: int,
    q: float,
) -> np.ndarray:
    """Per-episode, per-bin percentile as an (n_ep, n_bins) array; NaN where thin.

    Grouping by a combined key and splitting once keeps this linear in the number
    of requests. Masking each (episode, bin) pair in turn would be ~2,000 passes
    over an 800k-element array.
    """
    out = np.full((n_ep, n_bins), np.nan)
    keep = (idx >= 0) & (idx < n_bins)
    if not keep.any():
        return out

    key = ep[keep] * n_bins + idx[keep]
    order = np.argsort(key, kind="stable")
    key, vals = key[order], values[keep][order]
    starts = np.concatenate(([0], np.flatnonzero(np.diff(key)) + 1))
    for group, start in zip(np.split(vals, starts[1:]), starts):
        if group.size >= IMPACT_MIN_BIN_REQ:
            k = key[start]
            out[k // n_bins, k % n_bins] = np.percentile(group, q)
    return out


def bin_edges(centres: List[float]) -> np.ndarray:
    """Bin boundaries for the window's centres, one longer than the centres."""
    edges = np.asarray(centres) - IMPACT_BIN / 2
    return np.append(edges, centres[-1] + IMPACT_BIN / 2)


def episode_mean(
    ep: np.ndarray,
    idx: np.ndarray,
    values: np.ndarray,
    n_ep: int,
    n_bins: int,
    min_count: int,
) -> np.ndarray:
    """Per-episode, per-bin mean as an (n_ep, n_bins) array; NaN where thin.

    A mean is a ratio of two sums, so unlike the percentile version this needs no
    sorting -- two bincount passes over the whole array give every cell.
    """
    out = np.full(n_ep * n_bins, np.nan)
    keep = (idx >= 0) & (idx < n_bins)
    if not keep.any():
        return out.reshape(n_ep, n_bins)

    key = ep[keep] * n_bins + idx[keep]
    size = n_ep * n_bins
    total = np.bincount(key, weights=values[keep], minlength=size)
    count = np.bincount(key, minlength=size)
    enough = count >= min_count
    out[enough] = total[enough] / count[enough]
    return out.reshape(n_ep, n_bins)


class PolicySeries(NamedTuple):
    """One policy's binned lines, for the panels drawn on a shared y-axis."""

    ttft_per_ep: np.ndarray  # (n_episodes, n_bins)
    ttft_pooled: Dict[int, List[Optional[float]]]  # percentile -> line
    qlen_per_ep: np.ndarray  # (n_episodes, n_bins)
    qlen_pooled: List[Optional[float]]  # mean backlog across all episodes


def impact_series(data: ImpactData) -> PolicySeries:
    """Binned latency and backlog lines: per-episode matrix, then pooled lines.

    Split out of the plotting so the three policy figures can share one y-axis
    per panel: the limits must be known before any figure is drawn, and they
    depend on these binned values, not on the raw observations. A single request
    can reach 10^3 s while no plotted line goes near it.
    """
    centres = data.centres
    n_bins = len(centres)
    edges = bin_edges(centres)
    idx = np.digitize(data.rel, edges) - 1

    per_ep = episode_percentile(
        data.ep, idx, data.ttft, data.n_episodes, n_bins, IMPACT_PCTL
    )
    pooled = {
        pct: [
            float(np.percentile(data.ttft[idx == k], pct)) if np.any(idx == k) else None
            for k in range(n_bins)
        ]
        for pct, _, _ in IMPACT_POOLED
    }

    # Per-episode backlog lines are bin means: over ~15 samples of a step
    # function a mean is the time-average queue length, which is the quantity the
    # panel is about, and it stays defined in bins the trace sampled only once.
    q_idx = np.digitize(data.qlen_rel, edges) - 1
    qlen_per_ep = episode_mean(
        data.qlen_ep, q_idx, data.qlen, data.n_episodes, n_bins, QUEUE_MIN_BIN_SAMPLES
    )
    # The mean, not a percentile: a backlog is a total, and the actions that pile
    # up thousands of requests are the cost the panel exists to show. A median
    # would report the typical action's queue and hide exactly those.
    qlen_pooled = [
        float(np.mean(data.qlen[q_idx == k])) if np.any(q_idx == k) else None
        for k in range(n_bins)
    ]
    return PolicySeries(per_ep, pooled, qlen_per_ep, qlen_pooled)


def shared_top(
    per_ep: List[np.ndarray], lines: List[List[Optional[float]]]
) -> Optional[float]:
    """Highest value any policy plots on a panel, with headroom for the legend.

    Multiplicative headroom, not additive: both panels using this are logarithmic
    above their linear threshold, so a fixed margin would be invisible at the top
    and enormous at the bottom.
    """
    highest: List[float] = []
    for matrix in per_ep:
        if np.isfinite(matrix).any():
            highest.append(float(np.nanmax(matrix)))
    for line in lines:
        drawn = [v for v in line if v is not None]
        if drawn:
            highest.append(max(drawn))
    return max(highest) * 1.6 if highest else None


def shared_ttft_top(series: Dict[str, PolicySeries]) -> Optional[float]:
    """Latency limit shared by every policy's figure."""
    return shared_top(
        [s.ttft_per_ep for s in series.values()],
        [line for s in series.values() for line in s.ttft_pooled.values()],
    )


def shared_queue_top(series: Dict[str, PolicySeries]) -> Optional[float]:
    """Backlog limit shared by every policy's figure."""
    return shared_top(
        [s.qlen_per_ep for s in series.values()],
        [s.qlen_pooled for s in series.values()],
    )


def plot_action_impact(
    policy: str,
    data: ImpactData,
    series: PolicySeries,
    save_path: Path,
    ttft_top: Optional[float] = None,
    queue_top: Optional[float] = None,
    category_label: Optional[str] = None,
) -> None:
    """Stacked panels per policy: backlog, TTFT, quality score, shared time axis.

    Each reconfiguration is drawn as its own faint line; the value pooled over all
    of them carries the solid colour. Separate panels rather than twin y-axes: the
    measures have no common scale, so overlaying them would imply crossings and
    gaps that are artefacts of the axis limits rather than properties of the data.

    The panels are ordered by causation: an action's immediate cost is a queue
    that builds and drains, the latency panel below it is what that queue does to
    the requests sitting in it, and quality is what the new configuration serves
    them on.

    ttft_top and queue_top are the limits shared by every figure this run draws,
    per-category ones included. Left to autoscale, each would get its own and none
    could be read against another -- HPA's ~10^3 s recovery would occupy the same
    height as GO-CART's few seconds, and a policy's split figure would look like
    its merge figure at a different scale.

    category_label names the episode subset in the title; None means every
    attributable episode.
    """
    if data.n_episodes == 0 or data.rel.size == 0:
        what = f"{policy} ({category_label})" if category_label else policy
        print(f"{what}: no attributable reconfigurations; skipping impact plot.")
        return

    centres = data.centres
    colour = POLICY_COLORS[policy]

    sns.set_style("whitegrid")
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = SERIF_STACK
    plt.rcParams["mathtext.fontset"] = "stix"

    fig, (ax_l, ax_t, ax_q) = plt.subplots(
        3, 1, sharex=True, figsize=(10, 11), layout="constrained"
    )

    n_bins = len(centres)
    edges = bin_edges(centres)

    # Fainter as episodes accumulate, so a policy that reconfigures often does not
    # render as a solid block that hides the pooled line.
    faint = float(np.clip(2.5 / max(data.n_episodes, 1), 0.05, 0.45))

    # --- Queue: waiting requests, the backlog the action creates and drains. ---
    if data.qlen_rel.size:
        for row in series.qlen_per_ep:
            ax_l.plot(centres, row, color=colour, alpha=faint, linewidth=0.9, zorder=2)
        ax_l.plot(
            centres,
            series.qlen_pooled,
            color=colour,
            linewidth=2.5,
            marker="o",
            markersize=3.5,
            zorder=5,
            label="Pooled mean",
        )
        ax_l.plot(
            [],
            [],
            color=colour,
            alpha=max(faint, 0.25),
            linewidth=0.9,
            label="Individual action (mean)",
        )
        ax_l.legend(loc="upper right", frameon=True)

    # --- TTFT: one percentile, per reconfiguration faint and pooled solid. ---
    pooled_lines = series.ttft_pooled
    for row in series.ttft_per_ep:
        ax_t.plot(centres, row, color=colour, alpha=faint, linewidth=0.9, zorder=2)

    for pct, style, width in IMPACT_POOLED:
        name = "median" if pct == 50 else f"P{pct}"
        ax_t.plot(
            centres,
            pooled_lines[pct],
            color=colour,
            linewidth=width,
            linestyle=style,
            marker="o",
            markersize=3.5,
            zorder=5,
            label=f"Pooled {name}",
        )
    ax_t.plot(
        [],
        [],
        color=colour,
        alpha=max(faint, 0.25),
        linewidth=0.9,
        label=f"Individual action ({PCTL_LABEL.lower()})",
    )
    ax_t.legend(loc="upper right", frameon=True)

    # --- Quality: Eq. 5.1 evaluated per bin, token-weighted throughout. ---
    q_rel, q_pct, q_tok = data.q_rel, data.q_pct, data.q_tokens
    if q_rel.size:
        q_idx = np.digitize(q_rel, edges) - 1
        keep = (q_idx >= 0) & (q_idx < n_bins)
        # A weighted mean is a ratio of two sums, so bincount gives every
        # (episode, bin) cell in two passes.
        key = data.q_ep[keep] * n_bins + q_idx[keep]
        size = data.n_episodes * n_bins
        num = np.bincount(key, weights=(q_pct * q_tok)[keep], minlength=size)
        den = np.bincount(key, weights=q_tok[keep], minlength=size)
        with np.errstate(invalid="ignore", divide="ignore"):
            cells = np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)
        for row in cells.reshape(data.n_episodes, n_bins):
            ax_q.plot(centres, row, color=colour, alpha=faint, linewidth=0.9, zorder=2)

        pooled_num = np.bincount(
            q_idx[keep], weights=(q_pct * q_tok)[keep], minlength=n_bins
        )
        pooled_den = np.bincount(q_idx[keep], weights=q_tok[keep], minlength=n_bins)
        score = [
            float(pooled_num[k] / pooled_den[k]) if pooled_den[k] > 0 else None
            for k in range(n_bins)
        ]
        ax_q.plot(
            centres,
            score,
            color=colour,
            linewidth=2.5,
            marker="o",
            markersize=3.5,
            zorder=5,
            label="Pooled score",
        )
        ax_q.plot(
            [],
            [],
            color=colour,
            alpha=max(faint, 0.25),
            linewidth=0.9,
            label="Individual action",
        )
        ax_q.legend(loc="lower right", frameon=True)

    # The action fires at t=0; everything left of it is pre-action context.
    #
    # Bin edges are drawn as minor gridlines and every plotted value carries a
    # marker at its bin centre. Without them a reader takes the segment between
    # the last pre-action point (-7.5s) and the first post-action point (+7.5s)
    # as a slope through t=0, and reads the drop as starting before the action.
    # Markers show where the data actually is; the edge lines show that the
    # segment spans one bin boundary, which is where t=0 sits.
    for ax in (ax_l, ax_t, ax_q):
        ax.xaxis.set_minor_locator(MultipleLocator(IMPACT_BIN))
        ax.grid(which="minor", axis="x", linewidth=0.4, alpha=0.4, zorder=0)
        ax.axvline(0.0, color="#4a4a46", linewidth=1.6, zorder=3)

    # linscale compresses the 0..linthresh linear band. At the default it claims a
    # decade's worth of height for a region no line ever enters, since the quiet
    # baseline sits around 0.2s.
    ax_l.set_yscale("symlog", linthresh=QUEUE_LINTHRESH, linscale=0.25)
    ax_l.set_ylim(0, queue_top)
    ax_l.set_ylabel("Waiting Requests")
    ax_t.set_yscale("symlog", linthresh=IMPACT_LINTHRESH, linscale=0.25)
    ax_t.set_ylim(0, ttft_top)
    ax_t.set_ylabel("Time to First Token (s)")
    ax_q.set_ylim(0, 100)
    ax_q.set_ylabel("Model-Tier Quality Score (%)")
    ax_q.set_xlim(-IMPACT_PRE, IMPACT_POST)
    ax_q.set_xlabel("Time Relative to Reconfiguration Action (s)")

    subject = f"{policy} Policy"
    if category_label:
        subject += f" -- {category_label}"
    fig.suptitle(
        f"Queue, Latency and Quality Response to Reconfiguration Actions ({subject})"
    )

    os.makedirs(save_path.parent, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved action-impact plot to {save_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare reconfiguration impact across policies"
    )
    parser.add_argument(
        "--ckpt",
        type=Path,
        default=None,
        help="GO-CART checkpoint zip (trained RL model). Omit to compare baselines only.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/recovery_compare"),
        help="Directory for the action-impact figures (three per policy)",
    )
    return parser.parse_args()


def policy_status(log_path: Path) -> str:
    """Condense a worker's newest tqdm frame into a few characters.

    tqdm rewrites its bar in place with carriage returns and no newline, so the
    live frame is the last CR/LF-delimited chunk in the file. Reading the tail
    keeps this O(1) in log size. The two bars BenchRunner draws are matched
    separately: the step loop carries "n/total", while the flush-period bar has
    no total and reports "n steps".
    """
    try:
        with open(log_path, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            fh.seek(max(0, fh.tell() - 4096))
            tail = fh.read().decode("utf-8", "replace")
    except OSError:
        # The worker imports torch and rebuilds the workload before the first
        # write, so an absent log for the first half-minute is expected.
        return "starting"

    frame = next(
        (c.strip() for c in reversed(re.split(r"[\r\n]", tail)) if c.strip()), ""
    )
    step = re.search(r"(\d+)/(\d+) \[", frame)
    if step:
        return f"{step.group(1)}/{step.group(2)}"
    flush = re.search(r"(\d+) steps", frame)
    if flush:
        return f"flushing {flush.group(1)}"
    # Anything else is a plain line -- a setup breadcrumb, or a message the runner
    # printed between bars. Show it, clipped so three of them still fit a line.
    return frame[:24] if frame else "starting"


def run_parallel(
    policies: List[PolicySpec],
) -> Dict[str, Tuple[RecoveryByAgent, "ImpactData"]]:
    """Run every policy concurrently, one process each, reporting progress."""
    names = [p[0] for p in policies]
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logs = {n: LOG_DIR / f"{n.lower()}.log" for n in names}
    for path in logs.values():
        path.unlink(missing_ok=True)

    print(
        f"Running {len(policies)} policies in parallel ({', '.join(names)}); "
        f"each regenerates the same workload from seed {BENCH_CONFIG.seed}.\n"
        f"Full per-policy output: {LOG_DIR}/<policy>.log"
    )

    done: Dict[str, Tuple[RecoveryByAgent, ImpactData]] = {}
    failed: Dict[str, str] = {}
    live = sys.stdout.isatty()
    t0 = time.perf_counter()
    last_status = 0.0

    def render(final: bool = False) -> None:
        nonlocal last_status
        now = time.perf_counter()
        if not final and not live and now - last_status < QUIET_STATUS_INTERVAL:
            return
        last_status = now
        cells = []
        for n in names:
            if n in done:
                state = "done"
            elif n in failed:
                state = "FAILED"
            else:
                state = policy_status(logs[n])
            cells.append(f"{n} {state}")
        el = now - t0
        line = f"  [{int(el // 60):02d}:{el % 60:04.1f}]  " + "   ".join(cells)
        if live:
            # Pad to overwrite a longer previous frame; no newline, so the next
            # render replaces this one in place.
            print("\r" + line.ljust(110)[:110], end="", flush=True)
        else:
            print(line, flush=True)

    def clear() -> None:
        if live:
            print("\r" + " " * 110 + "\r", end="", flush=True)

    # spawn rather than fork: torch is already imported in the parent, and forking
    # a process with CUDA state initialised is unsafe.
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=len(policies), mp_context=ctx) as pool:
        futures = {pool.submit(run_policy, spec): spec[0] for spec in policies}
        pending = set(futures)
        while pending:
            finished, pending = wait(
                pending, timeout=POLL_INTERVAL, return_when=FIRST_COMPLETED
            )
            for fut in finished:
                name = futures[fut]
                clear()
                try:
                    _, recovery, impact_data = fut.result()
                except Exception as exc:  # noqa: BLE001 - one bad policy must not sink the rest
                    # A stale GO-CART checkpoint (observation-space mismatch) is the
                    # common case; the baselines are still worth reporting.
                    failed[name] = f"{type(exc).__name__}: {exc}"
                    print(
                        f"  {name} FAILED after {time.perf_counter() - t0:.0f}s: "
                        f"{failed[name]}  (see {logs[name]})"
                    )
                    continue
                done[name] = (recovery, impact_data)
                print(
                    f"  {name} done after {time.perf_counter() - t0:.0f}s "
                    f"({len(done) + len(failed)}/{len(policies)})"
                )
            render(final=bool(finished))

    clear()
    print(f"  all policies finished in {time.perf_counter() - t0:.0f}s")
    return done


def main() -> None:
    torch.distributions.Distribution.set_default_validate_args(False)
    args = parse_args()

    policies: List[PolicySpec] = []
    if args.ckpt:
        policies.append(("GO-CART", BenchMode.RL, args.ckpt))
    else:
        print("No --ckpt given; comparing baselines only.")
    policies.append(("HPA", BenchMode.BASELINE_HEURISTIC, None))
    policies.append(("QAS", BenchMode.BASELINE_QAS, None))

    names = [p[0] for p in policies]
    done = run_parallel(policies)

    if not done:
        raise SystemExit("Every policy failed; nothing to report.")

    # Restore declared order, so the audit table does not come out in whatever
    # order the runs happened to finish.
    collected: Dict[str, RecoveryByAgent] = {n: done[n][0] for n in names if n in done}
    impact: Dict[str, ImpactData] = {n: done[n][1] for n in names if n in done}

    audit = summarize(collected)

    print(
        f"\n● Reconfiguration Recovery Episode Accounting"
        f"\n  Recovery failure = waiting queue still non-empty after "
        f"{RECOVERY_HORIZON:.0f}s."
    )
    print(
        tabulate.tabulate(
            audit,
            headers=[
                "Policy",
                "Agent",
                "Recovered",
                "Failed",
                "Failure rate",
                "Discarded: pattern",
                "Discarded: overlap",
                "Total",
            ],
            tablefmt="fancy_grid",
        )
    )
    discarded = sum(r[5] + r[6] for r in audit)
    print(
        f"\nDiscarded {discarded} episode(s) as unattributable "
        "(spanning a workload-pattern change, or overlapping another "
        "reconfiguration). These are excluded from the recovered and failed "
        "counts above."
    )

    # Three figures per policy: every attributable episode, then the same
    # episodes cut by the direction of the capacity change (EPISODE_CATEGORIES).
    figures: List[Tuple[str, ImpactData, Path, Optional[str]]] = []
    for name, windows in impact.items():
        stem = args.out_dir / f"action_impact_{name.lower()}"
        figures.append((name, windows, stem.with_name(f"{stem.name}.png"), None))
        for key, suffix, label in EPISODE_CATEGORIES:
            figures.append((
                name,
                category_subset(windows, key),
                stem.with_name(f"{stem.name}_{suffix}.png"),
                label,
            ))

    print("\n● Episodes drawn per figure (attributable episodes only)")
    for name, windows in impact.items():
        counts = " ".join(
            f"{label.split(':')[0]} {int((windows.ep_cat == key).sum())}"
            for key, _, label in EPISODE_CATEGORIES
        )
        uncategorised = int((windows.ep_cat == "").sum())
        line = f"  {name:<8} total {windows.n_episodes:<5} {counts}"
        print(line + (f"  uncategorised {uncategorised}" if uncategorised else ""))

    # All nine are binned before any is drawn: the shared axis limits depend on
    # the binned values, so every figure has to exist as a series first.
    series = {str(path): impact_series(d) for _, d, path, _ in figures if d.n_episodes}
    top = shared_ttft_top(series)
    q_top = shared_queue_top(series)
    for name, windows, path, label in figures:
        if str(path) not in series:
            what = f"{name} ({label})" if label else name
            print(f"{what}: no attributable reconfigurations; skipping impact plot.")
            continue
        plot_action_impact(
            name,
            windows,
            series[str(path)],
            path,
            ttft_top=top,
            queue_top=q_top,
            category_label=label,
        )


if __name__ == "__main__":
    main()
