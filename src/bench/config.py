import yaml
from pathlib import Path
from typing import List, Tuple, Union, Any

from src.bench.models import Workload
import src.share.models as m
import src.simulation.config as sim_config
from src.simulation.config import GPU_MIG_PROFILE
from src.training.config import TRAINING_CONFIG


def denormalize_arrival_rate(val: float) -> float:
    """Shared by RuleBasedHeuristic and QualityAwareScheduler to convert a
    normalized observed arrival rate back to req/s."""
    return val * TRAINING_CONFIG.norm_arrival_rate


class BenchConfig:
    def __init__(self, config_path: Path):
        with open(config_path, "r") as f:
            data = yaml.safe_load(f)
            self._workloads = data["workloads"]
            self._length = data["benchmark-length"]
            self._seed = data.get("seed", 42)
            self._heuristic = data.get(
                "heuristic", {"watermark_high": 20.0, "watermark_low": 5.0}
            )
            self._qas = data.get("qas", {})
            self._workload_sequence = data.get("workload_sequence", None)

    @property
    def workload_sequence(self) -> list[Any] | None:
        return self._workload_sequence

    @property
    def utilization_factor(self) -> float:
        return float(self._heuristic.get("utilization_factor", 0.8))

    @property
    def high_threshold(self) -> float:
        return float(self._heuristic.get("high_threshold", 1.2))

    @property
    def low_threshold(self) -> float:
        return float(self._heuristic.get("low_threshold", 0.8))

    def get_service_rate(
        self,
        agent_id: m.AgentId,
        mig_profile: Union[m.MIGProfile, m.MIGProfileBase],
        gpu_id: int = 0,
    ) -> float:
        rates = self._heuristic.get("service_rates", {})

        if isinstance(mig_profile, m.MIGProfile):
            # Resolve logical profile to concrete hardware profile for the given gpu_id
            hw_prof = next(
                p for p in GPU_MIG_PROFILE[gpu_id] if p.profile_type == mig_profile
            )
        else:
            hw_prof = mig_profile

        gpu_model = hw_prof.gpu_model

        # Structure: service_rates[gpu_model][agent_id][mig_str]
        model_rates = rates.get(gpu_model, {})
        agent_rates = model_rates.get(agent_id.value, {})

        prof_str = hw_prof.string
        original_rate = float(agent_rates.get(prof_str, 0.0))

        match hw_prof.profile_type:
            case m.MIGProfile.MIG_7G:
                factor = 1.0
            case m.MIGProfile.MIG_4G | m.MIGProfile.MIG_3G:
                factor = 0.8
            case (
                m.MIGProfile.MIG_2G
                | m.MIGProfile.MIG_1G_LARGE
                | m.MIGProfile.MIG_1G_SMALL
            ):
                factor = 0.5
            case _:
                raise ValueError(f"Unknown MIG profile type: {hw_prof.profile_type}")

        return original_rate * factor

    @property
    def qas_split_headroom(self) -> float:
        """Headroom margin for QAS's waterfall load split across an agent's
        engines (see QualityAwareScheduler._aggregate): every engine except
        the lowest-priority (smallest) one is capped at this fraction of its
        own service rate, rather than the full rate. Without this, the
        highest-priority engine gets driven to exactly 100% utilization
        whenever demand exceeds it alone, which blows its predicted TTFT up
        to infinity and poisons the whole allocation's aggregate -- even when
        the *other* engines in the same allocation have ample spare capacity.
        Deliberately a separate knob from utilization_factor (HPA's
        scaling-trigger threshold) even though both currently default to 0.8,
        since tuning one should not silently change the other's behavior."""
        return float(self._qas.get("split_headroom_factor", 0.8))

    @property
    def qas_min_quality_gain(self) -> float:
        """Minimum total-Q_f improvement (summed across agents) required
        before QAS will take a reconfiguration action when the cluster is
        already fully TTFT-feasible -- QAS's equivalent of HPA's
        high_threshold/low_threshold deadband (RuleBasedHeuristic.
        decide_action's `needs_action` check). Without this (margin=0), QAS
        reconfigures on *any* strictly-positive quality gain, however
        marginal, every decision step -- but each split/merge/transfer costs
        a real ~60-70s BOOTING downtime (SIM_CONFIG.get_restart_time), so
        chasing negligible gains risks a backlog under load that the arrival
        rate never lets the queue recover from (confirmed: margin=0 gives a
        894s CodingAgent P99 TTFT with a runaway queue). Full-length sweeps
        at margin in {0.0, 0.1, 0.25, 0.35, 0.5, 1.0, 1.5, 2.0} found a flat
        optimum at 0.25-0.5 (identical results across that whole sub-range:
        best P99 and best avg Q_f for both agents) -- below it (0.1) transfer
        actions start firing again but the result is marginally *worse* on
        every metric (transfers aren't actually worth their downtime cost
        here, given TRAINING_CONFIG.qf is agent-invariant so a pure
        ownership transfer nets a much smaller Q_f delta than a reshape);
        above it (1.0, 1.5, 2.0) both quality and P99 degrade monotonically.
        0.25 sits in the middle of that flat optimum. Only gates same-
        feasibility-tier comparisons -- an action that fixes an SLO
        violation is never held back by this margin (see decide_action)."""
        return float(self._qas.get("min_quality_gain", 0.25))

    @property
    def l_target(self) -> float:
        """QAS's base TTFT ceiling (seconds), before any per-agent
        adjustment -- see l_target_for(). Earlier sweeps over this value in
        isolation were confounded by RAGAgent's mandatory RAG-search overhead
        (see l_target_for): any base value below RAGAgent's fixed overhead
        floor makes RAGAgent permanently TTFT-infeasible regardless of MIG
        allocation, which (via decide_action's lexicographic scoring)
        freezes every agent's allocation, not just RAGAgent's. Once
        l_target_for's offset is applied, a full-length sweep over
        {0.25, 0.3, 0.4, 0.5, 0.6, 0.75, 1.0} found this value has *no*
        effect in that whole range once qas_min_quality_gain is set
        correctly -- it produced bit-for-bit identical outcomes at 0.25 and
        0.4, since qas_min_quality_gain is the binding constraint on
        reconfiguration once RAGAgent's floor is no longer the issue. 0.4
        was kept as a comfortable value well inside that flat range."""
        return float(self._qas.get("L_target", 0.4))

    def l_target_for(self, agent_id: m.AgentId, gpu_id: int = 0) -> float:
        """Per-agent TTFT ceiling: l_target plus this agent's mean fixed
        pre-processing overhead, if any (e.g. RAGAgent's random RAG-search
        delay sampled before generation begins -- src/simulation/simulator.py
        ``_sample_rag_searches``/``SIM_CONFIG.get_rag_overhead``, config key
        ``search-overhead``). That delay is added to every RAGAgent
        request's measured TTFT regardless of MIG allocation, so it isn't
        something scheduling quality can ever improve -- charging it against
        the same SLO budget as allocation-driven latency would make RAGAgent
        permanently infeasible (see l_target's docstring) no matter which
        profile it holds."""
        base = self.l_target
        agent_cfg = sim_config.GPU_AGENTS_CONFIG.get(gpu_id, {}).get(agent_id.value, {})
        overhead = agent_cfg.get("search-overhead")
        if overhead:
            base += (float(overhead["min"]) + float(overhead["max"])) / 2
        return base

    def get_ttft_curve(
        self,
        agent_id: m.AgentId,
        mig_profile: Union[m.MIGProfile, m.MIGProfileBase],
        gpu_id: int = 0,
    ) -> List[Tuple[float, float]]:
        """Returns the (lambda, TTFT) points built offline by
        `python -m src.bench.qas_profile`, sorted ascending by lambda."""
        table = self._qas.get("profile_table", {})

        if isinstance(mig_profile, m.MIGProfile):
            hw_prof = next(
                p for p in GPU_MIG_PROFILE[gpu_id] if p.profile_type == mig_profile
            )
        else:
            hw_prof = mig_profile

        model_table = table.get(hw_prof.gpu_model, {})
        agent_table = model_table.get(agent_id.value, {})
        points = agent_table.get(hw_prof.string, [])
        return [(float(lam), float(ttft)) for lam, ttft in points]

    # Floor slope (extra TTFT seconds per extra req/s of overload) used to
    # extrapolate predict_ttft beyond the profiled range -- see its docstring.
    _MIN_OVERLOAD_SLOPE = 1.0

    def predict_ttft(
        self,
        agent_id: m.AgentId,
        mig_profile: Union[m.MIGProfile, m.MIGProfileBase],
        gpu_id: int,
        lam: float,
    ) -> float:
        """Piecewise-linear interpolation of the profiled TTFT-vs-lambda curve.
        Clamps to the lowest profiled TTFT below the min profiled lambda.

        Above the max profiled lambda, extrapolates linearly from the curve's
        own tail slope (floored at _MIN_OVERLOAD_SLOPE) rather than returning
        a flat +inf. A flat +inf was tried first and empirically causes QAS
        to freeze: once demand is high enough that every single-step-reachable
        allocation exceeds its curve's profiled range, every candidate ties at
        (tier=1, violation=inf), and is_better's `candidate < baseline -
        margin` comparison can never resolve an inf-vs-inf tie -- confirmed on
        the A100 bimodal benchmark, where this produced a 104-step (~3.5h
        simulated) then a 39-step consecutive NO_ACTION streak with every
        candidate scored infeasible, driving CodingAgent's P99 TTFT past
        1500s regardless of L_target/min_quality_gain. Extrapolating keeps
        these candidates correctly classified as infeasible (the projected
        TTFT will clear l_target by a wide margin) while still ordering them
        by how overloaded they are, so QAS can take the least-bad step toward
        feasibility instead of freezing. The floored slope guards against
        curves whose last two profiled points are locally flat or noisy (measurement
        sigma, not true capacity) producing a shallow or negative extrapolation
        that would mask real overload."""
        curve = self.get_ttft_curve(agent_id, mig_profile, gpu_id)
        if not curve:
            return float("inf")

        if lam <= curve[0][0]:
            return curve[0][1]
        if lam >= curve[-1][0]:
            if lam == curve[-1][0]:
                return curve[-1][1]
            lam_hi, ttft_hi = curve[-1]
            slope = 0.0
            if len(curve) >= 2:
                lam_lo, ttft_lo = curve[-2]
                slope = (ttft_hi - ttft_lo) / (lam_hi - lam_lo) if lam_hi > lam_lo else 0.0
            slope = max(slope, self._MIN_OVERLOAD_SLOPE)
            return ttft_hi + slope * (lam - lam_hi)

        for (lam_lo, ttft_lo), (lam_hi, ttft_hi) in zip(curve, curve[1:]):
            if lam_lo <= lam <= lam_hi:
                if lam_hi == lam_lo:
                    return ttft_lo
                frac = (lam - lam_lo) / (lam_hi - lam_lo)
                return ttft_lo + frac * (ttft_hi - ttft_lo)

        return float("inf")

    def get_rate_range(
        self, workload: Workload, agent_id: m.AgentId
    ) -> Tuple[float, float]:
        cfg = self._workloads[workload.value]["rate"]
        if isinstance(cfg, dict):
            cfg = cfg[agent_id.value]
        return float(cfg[0]), float(cfg[1])

    def get_duration_range(self, workload: Workload) -> Tuple[float, float]:
        cfg = self._workloads[workload.value]["duration"]
        return float(cfg[0]), float(cfg[1])

    @property
    def benchmark_length(self) -> int:
        return int(self._length)

    @property
    def seed(self) -> int:
        return int(self._seed)


# Automatically load from default path
project_root = Path(".")
BENCH_CONFIG = BenchConfig(project_root / "configs" / "bench_config.yaml")
