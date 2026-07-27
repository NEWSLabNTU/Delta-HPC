"""Compare reconfiguration impact across policies on one shared workload.

Runs GO-CART, the HPA rule-based heuristic and QAS over an identical request
stream, then draws one figure per policy showing how latency and model-tier
quality respond in a window around each reconfiguration action. Episode
accounting is printed to the terminal.

The policies run concurrently, one process each. Threads would not help -- the
simulator is a pure-Python event loop and holds the GIL throughout.

Episode accounting differs by exclusion reason, deliberately:

  - Episodes that never cleared inside the observation window are a *result*, not
    a measurement failure -- they are reported as each policy's failure rate in
    the accounting table rather than being dropped silently.
  - Episodes spanning a workload-pattern change are discarded outright. The queue
    there moves for reasons unrelated to the reconfiguration, so they belong in
    neither the recovered nor the failed count. They are counted and printed so
    the discard volume stays visible.
  - Episodes overlapping another reconfiguration are likewise unattributable and
    discarded on the same terms.

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
IMPACT_POST = 300.0  # seconds tracked afterwards
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

PolicySpec = Tuple[str, BenchMode, Optional[Path]]
RecoveryByAgent = Dict[m.AgentId, Dict[str, Any]]


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
    n_episodes: int


def action_windows(runner: BenchRunner) -> ImpactData:
    """Collect per-request TTFT and per-action quality traces around each trigger.

    The two panels are indexed by different clocks, deliberately. Latency is a
    property of a request's whole journey, so it belongs at its arrival. Quality
    is fixed by the engine that served it, so it belongs at the moment service
    began -- binning it by arrival credits a post-reconfiguration tier to a
    pre-action bin whenever the request sat in the queue across the trigger, and
    under HPA (pre-action P99 TTFT ~10^3 s) that reaches back tens of
    seconds and makes quality appear to fall before the action that caused it.
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
    n_episodes = 0

    for ep in runner.env.sim.reconfig_episodes:
        # Overlapping reconfigurations put two disturbances in one window, so the
        # trace could not be attributed to this action.
        if ep.interrupted or ep.t_boot_done is None:
            continue

        aid = ep.agent_id
        lo = bisect.bisect_left(arrivals[aid], ep.t_trigger - IMPACT_PRE)
        hi = bisect.bisect_right(arrivals[aid], ep.t_trigger + IMPACT_POST)
        window = by_agent[aid][lo:hi]
        if not window:
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
        n_episodes=n_episodes,
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


LatencySeries = Tuple[np.ndarray, Dict[int, List[Optional[float]]]]


def impact_series(data: ImpactData) -> LatencySeries:
    """Binned latency lines for one policy: per-episode matrix, then pooled lines.

    Split out of the plotting so the three policy figures can share one y-axis:
    the limit must be known before any figure is drawn, and it depends on these
    binned values, not on raw TTFTs. A single request can reach 10^3 s while no
    plotted line goes near it.
    """
    centres = data.centres
    n_bins = len(centres)
    edges = np.asarray(centres) - IMPACT_BIN / 2
    edges = np.append(edges, centres[-1] + IMPACT_BIN / 2)
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
    return per_ep, pooled


def shared_ttft_top(series: Dict[str, LatencySeries]) -> Optional[float]:
    """Highest latency value any policy plots, with headroom for the legend.

    Multiplicative headroom, not additive: the axis is logarithmic above
    IMPACT_LINTHRESH, so a fixed margin would be invisible at the top and enormous
    at the bottom.
    """
    highest: List[float] = []
    for per_ep, pooled in series.values():
        if np.isfinite(per_ep).any():
            highest.append(float(np.nanmax(per_ep)))
        for line in pooled.values():
            drawn = [v for v in line if v is not None]
            if drawn:
                highest.append(max(drawn))
    return max(highest) * 1.6 if highest else None


def plot_action_impact(
    policy: str,
    data: ImpactData,
    series: LatencySeries,
    save_path: Path,
    ttft_top: Optional[float] = None,
) -> None:
    """Stacked panels per policy: TTFT above, quality score below, shared time axis.

    Each reconfiguration is drawn as its own faint line; the value pooled over all
    of them carries the solid colour. Two panels rather than twin y-axes: the
    measures have no common scale, so overlaying them would imply crossings and
    gaps that are artefacts of the axis limits rather than properties of the data.

    ttft_top is the latency limit shared by every policy's figure. Left to
    autoscale, each policy would get its own and the three could not be read
    against one another -- HPA's ~10^3 s recovery would occupy the same height as
    GO-CART's few seconds.
    """
    if data.n_episodes == 0 or data.rel.size == 0:
        print(f"{policy}: no attributable reconfigurations; skipping impact plot.")
        return

    centres = data.centres
    colour = POLICY_COLORS[policy]

    sns.set_style("whitegrid")
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = SERIF_STACK
    plt.rcParams["mathtext.fontset"] = "stix"

    fig, (ax_t, ax_q) = plt.subplots(
        2, 1, sharex=True, figsize=(10, 8), layout="constrained"
    )

    n_bins = len(centres)
    edges = np.asarray(centres) - IMPACT_BIN / 2
    edges = np.append(edges, centres[-1] + IMPACT_BIN / 2)

    # Fainter as episodes accumulate, so a policy that reconfigures often does not
    # render as a solid block that hides the pooled line.
    faint = float(np.clip(2.5 / max(data.n_episodes, 1), 0.05, 0.45))

    # --- TTFT: one percentile, per reconfiguration faint and pooled solid. ---
    per_ep, pooled_lines = series
    for row in per_ep:
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
    ax_t.plot([], [], color=colour, alpha=max(faint, 0.25), linewidth=0.9,
              label=f"Individual action ({PCTL_LABEL.lower()})")
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

        pooled_num = np.bincount(q_idx[keep], weights=(q_pct * q_tok)[keep], minlength=n_bins)
        pooled_den = np.bincount(q_idx[keep], weights=q_tok[keep], minlength=n_bins)
        score = [
            float(pooled_num[k] / pooled_den[k]) if pooled_den[k] > 0 else None
            for k in range(n_bins)
        ]
        ax_q.plot(
            centres, score, color=colour, linewidth=2.5, marker="o", markersize=3.5,
            zorder=5, label="Pooled score",
        )
        ax_q.plot([], [], color=colour, alpha=max(faint, 0.25), linewidth=0.9,
                  label="Individual action")
        ax_q.legend(loc="lower right", frameon=True)

    # The action fires at t=0; everything left of it is pre-action context.
    #
    # Bin edges are drawn as minor gridlines and every plotted value carries a
    # marker at its bin centre. Without them a reader takes the segment between
    # the last pre-action point (-7.5s) and the first post-action point (+7.5s)
    # as a slope through t=0, and reads the drop as starting before the action.
    # Markers show where the data actually is; the edge lines show that the
    # segment spans one bin boundary, which is where t=0 sits.
    for ax in (ax_t, ax_q):
        ax.xaxis.set_minor_locator(MultipleLocator(IMPACT_BIN))
        ax.grid(which="minor", axis="x", linewidth=0.4, alpha=0.4, zorder=0)
        ax.axvline(0.0, color="#4a4a46", linewidth=1.6, zorder=3)

    # linscale compresses the 0..linthresh linear band. At the default it claims a
    # decade's worth of height for a region no line ever enters, since the quiet
    # baseline sits around 0.2s.
    ax_t.set_yscale("symlog", linthresh=IMPACT_LINTHRESH, linscale=0.25)
    ax_t.set_ylim(0, ttft_top)
    ax_t.set_ylabel("Time to First Token (s)")
    ax_q.set_ylim(0, 100)
    ax_q.set_ylabel("Model-Tier Quality Score (%)")
    ax_q.set_xlim(-IMPACT_PRE, IMPACT_POST)
    ax_q.set_xlabel("Time Relative to Reconfiguration Action (s)")

    fig.suptitle(
        f"Latency and Quality Response to Reconfiguration Actions ({policy} Policy)"
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
        help="Directory for the per-policy action-impact figures",
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

    frame = next((c.strip() for c in reversed(re.split(r"[\r\n]", tail)) if c.strip()), "")
    step = re.search(r"(\d+)/(\d+) \[", frame)
    if step:
        return f"{step.group(1)}/{step.group(2)}"
    flush = re.search(r"(\d+) steps", frame)
    if flush:
        return f"flushing {flush.group(1)}"
    # Anything else is a plain line -- a setup breadcrumb, or a message the runner
    # printed between bars. Show it, clipped so three of them still fit a line.
    return frame[:24] if frame else "starting"


def run_parallel(policies: List[PolicySpec]) -> Dict[str, Tuple[RecoveryByAgent, "ImpactData"]]:
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
                    print(f"  {name} FAILED after {time.perf_counter() - t0:.0f}s: "
                          f"{failed[name]}  (see {logs[name]})")
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

    # Every policy's lines are binned first so the three figures can be drawn on
    # one latency scale and read against each other.
    series = {name: impact_series(d) for name, d in impact.items() if d.n_episodes}
    top = shared_ttft_top(series)
    for name, windows in impact.items():
        if name not in series:
            print(f"{name}: no attributable reconfigurations; skipping impact plot.")
            continue
        plot_action_impact(
            name,
            windows,
            series[name],
            args.out_dir / f"action_impact_{name.lower()}.png",
            ttft_top=top,
        )


if __name__ == "__main__":
    main()
