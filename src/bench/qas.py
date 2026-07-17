import logging
from typing import Dict, List, Tuple

import src.share.models as m
from src.share.mig_matrix import STATE_DEFINITIONS
from src.bench.config import BENCH_CONFIG, denormalize_arrival_rate
from src.training.config import TRAINING_CONFIG
import src.simulation.config as sim_config

logger = logging.getLogger(__name__)

AgentProfiles = Dict[m.AgentId, List[Tuple[int, m.MIGProfileBase]]]


class QualityAwareScheduler:
    """Quality-Aware Scheduling (QAS) baseline, adapted from the SLA-aware MIG
    selection algorithm in Zhu et al. (IBM, MASCOTS 2024): inverts IBM's
    cost-minimizing search (smallest MIG profile meeting a per-token SLA) into
    a quality-maximizing one, directly over the real admissible action set --
    among actions leaving every agent's predicted TTFT under its own
    BENCH_CONFIG.l_target_for(agent), pick whichever maximizes total Q_f; if none is
    fully feasible, pick whichever minimizes total SLO violation instead.

    Known limitation (strict 1-step greedy, by design -- mirrors
    RuleBasedHeuristic's own greedy structure and the IBM paper's static,
    non-stepwise origin): an agent given a lone 7G engine cannot sustain
    busy/burst-level arrival rates on its own (e.g. CodingAgent's B200
    7g.180gb tops out at a measured mu=1.5 req/s, while busy/burst arrival
    rates run 3-5 req/s) -- predict_ttft correctly returns inf there since
    the offline profiling itself never reached steady state at that load.
    Pairing that lone 7G with a *transferred* engine from the other agent
    could in principle make it sustainable (the transferred engine absorbs
    the waterfall-split overflow), but reaching that compound allocation
    needs two separate actions (a GPU_x_PROFILE_1_7G reshape plus a
    TRANSFER), each evaluated independently one decision interval apart.
    Neither half looks good in isolation to a 1-step lookahead: the reshape
    alone is infeasible (rejected outright), and the transfer alone barely
    changes total Q_f (TRAINING_CONFIG.qf is agent-invariant, so moving an
    existing profile between agents is close to a wash) while the recipient
    is already feasible without it, so it never clears qas_min_quality_gain
    either. Confirmed empirically: raising Q_f(7g.180gb) from 12 to 18 in
    configs/training_config.yaml produced bit-for-bit identical results
    (7G usage unchanged at ~96% idle / ~1% busy-burst) -- this is a genuine
    greedy-myopia gap, not a scoring-weight problem, and fixing it would
    require evaluating joint reshape+transfer bundles as a single candidate,
    which is out of scope for this baseline (left as a documented
    limitation rather than implemented).
    """

    def __init__(self, get_service_rate=None, get_ttft=None):
        self.get_service_rate = get_service_rate or BENCH_CONFIG.get_service_rate
        self.get_ttft = get_ttft or BENCH_CONFIG.predict_ttft

    def _denormalize_arrival_rate(self, val: float) -> float:
        return denormalize_arrival_rate(val)

    def snapshot_agent_profiles(self, sim: m.Simulator) -> AgentProfiles:
        """Current (gpu_id, hw_profile) held by each agent, across all GPUs."""
        return {
            aid: [
                (e.gpu, e.mig_profile)
                for e in sim.agents[aid].engines
                if e.status != m.EngineStatus.BOOTING
            ]
            for aid in m.AgentId
        }

    def simulate_agent_profiles(
        self, sim: m.Simulator, action: m.ResourceManagerAction
    ) -> AgentProfiles:
        """Counterfactual per-agent profile lists after `action`, mirroring
        RuleBasedHeuristic.simulate_service_rates's src/target diff logic but
        tracking profile identity instead of a summed rate."""
        new_profiles = {
            aid: list(profs) for aid, profs in self.snapshot_agent_profiles(sim).items()
        }
        sim_action = sim.map_to_action(action)
        if sim_action is None:
            return new_profiles

        gpu_id = sim_action.gpu_id
        giver_id = sim.gpu_engines[gpu_id][sim_action.mig_src[0]].owner.agent_id

        # 1. Remove source engines' profiles from their current owners
        for idx in sim_action.mig_src:
            eng = sim.gpu_engines[gpu_id][idx]
            owner_id = eng.owner.agent_id
            entry = (gpu_id, eng.mig_profile)
            if entry in new_profiles[owner_id]:
                new_profiles[owner_id].remove(entry)

        # 2. Add resulting target engines' profiles to their new owners
        if sim_action.target_state_id is None:
            # Pure transfer: no state change, profile just changes ownership
            for idx in sim_action.mig_src:
                eng = sim.gpu_engines[gpu_id][idx]
                owner_id = giver_id
                if sim_action.receiver and sim_action.receiver.mig_idx == idx:
                    owner_id = sim_action.receiver.receiver_id
                new_profiles[owner_id].append((gpu_id, eng.mig_profile))
        else:
            target_profiles = STATE_DEFINITIONS[sim_action.target_state_id]
            for idx in sim_action.mig_target:
                logical_profile = target_profiles[idx]
                hardware_profile = next(
                    hp
                    for hp in sim_config.GPU_MIG_PROFILE[gpu_id]
                    if hp.profile_type == logical_profile
                )
                owner_id = giver_id
                if sim_action.receiver and sim_action.receiver.mig_idx == idx:
                    owner_id = sim_action.receiver.receiver_id
                new_profiles[owner_id].append((gpu_id, hardware_profile))

        return new_profiles

    def _aggregate(
        self,
        agent_id: m.AgentId,
        profiles: List[Tuple[int, m.MIGProfileBase]],
        lam_i: float,
    ) -> Tuple[float, float]:
        """Waterfall-splits lam_i across `profiles` (sorted by .size
        descending, mirroring the dispatcher's own tie-break in
        src/simulation/agent.py's selection_key) and returns the
        traffic-weighted (Q_f, TTFT) for this allocation."""
        if not profiles:
            return 0.0, 0.0

        if lam_i <= 0:
            # No load: quality is whatever the largest held profile offers,
            # with no latency concern (matches the dispatcher's own
            # preference for the largest idle engine).
            best = max(profiles, key=lambda gp: gp[1].size)
            return TRAINING_CONFIG.qf(best[1], agent_id), 0.0

        sorted_profiles = sorted(profiles, key=lambda gp: gp[1].size, reverse=True)
        n = len(sorted_profiles)
        remaining = lam_i
        headroom = BENCH_CONFIG.qas_split_headroom

        weighted_qf = 0.0
        weighted_ttft = 0.0
        for idx, (gpu_id, hw_profile) in enumerate(sorted_profiles):
            if idx == n - 1:
                # Lowest-priority (smallest) engine absorbs whatever's left,
                # uncapped -- this is the only place true overload (this
                # allocation's aggregate capacity exhausted) should surface.
                lam_j = remaining
            else:
                mu_j = self.get_service_rate(agent_id, hw_profile, gpu_id=gpu_id)
                # Cap below full capacity so this engine alone exceeding
                # demand doesn't drive it to exactly 100% utilization (TTFT
                # -> infinity), which would poison the aggregate even when
                # other engines in this same allocation have spare capacity.
                lam_j = min(remaining, headroom * mu_j)
                remaining -= lam_j

            weight = lam_j / lam_i
            weighted_qf += weight * TRAINING_CONFIG.qf(hw_profile, agent_id)
            weighted_ttft += weight * self.get_ttft(agent_id, hw_profile, gpu_id, lam_j)

        return weighted_qf, weighted_ttft

    def decide_action(self, sim: m.Simulator) -> m.ResourceManagerAction:
        state = sim.get_state()
        mask = sim.get_action_mask(ignore_cooldowns=False)
        all_actions = list(m.ResourceManagerAction)
        valid_actions = [
            a
            for i, a in enumerate(all_actions)
            if mask[i] and a != m.ResourceManagerAction.NO_ACTION
        ]

        arrival_rates: Dict[m.AgentId, float] = {}
        for aid in m.AgentId:
            arr_rate = self._denormalize_arrival_rate(state["arrival_rate"][aid])

            agent = sim.agents[aid]
            total_queue = sum(
                len(e.waiting_queue)
                for e in agent.engines
                if hasattr(e, "waiting_queue")
            )
            arr_rate += total_queue / TRAINING_CONFIG.action_interval

            arrival_rates[aid] = arr_rate

        current_profiles = self.snapshot_agent_profiles(sim)
        l_target_by_agent = {
            aid: BENCH_CONFIG.l_target_for(aid) for aid in m.AgentId
        }

        current_qf: Dict[m.AgentId, float] = {}
        current_ttft: Dict[m.AgentId, float] = {}
        for aid in m.AgentId:
            qf, ttft = self._aggregate(aid, current_profiles[aid], arrival_rates[aid])
            current_qf[aid] = qf
            current_ttft[aid] = ttft

        def get_score(
            qf_by_agent: Dict[m.AgentId, float], ttft_by_agent: Dict[m.AgentId, float]
        ) -> Tuple[int, float]:
            """Lexicographic score: any action keeping every agent under its
            own l_target_for(...) ceiling beats any action that doesn't,
            regardless of total quality -- only within the same feasibility
            tier does the quality objective (or, failing full feasibility,
            the total SLO violation) act as the tiebreaker. Feasible actions
            are ranked by -sum(Q_f) so that `<` (lower score wins) picks the
            highest total quality among them."""
            violation = sum(
                max(0.0, ttft_by_agent[aid] - l_target_by_agent[aid])
                for aid in m.AgentId
            )
            if violation <= 0.0:
                return (0, -sum(qf_by_agent.values()))
            return (1, violation)

        current_score = get_score(current_qf, current_ttft)
        min_quality_gain = BENCH_CONFIG.qas_min_quality_gain

        def is_better(candidate: Tuple[int, float], baseline: Tuple[int, float]) -> bool:
            """True if `candidate` should replace `baseline`. A tier change
            (fixing or newly causing an SLO violation) always wins outright
            -- that's an urgent correctness difference, not a quality
            trade-off. Within the same tier, a reconfiguration must clear
            min_quality_gain to be worth its ~60-70s BOOTING downtime risk
            (mirrors HPA's high/low threshold deadband)."""
            if candidate[0] != baseline[0]:
                return candidate[0] < baseline[0]
            return candidate[1] < baseline[1] - min_quality_gain

        if not valid_actions:
            current_str = ", ".join(f"{aid.name}: {current_qf[aid]:.2f}" for aid in m.AgentId)
            logger.info(
                "QAS deciding NO_ACTION (no valid actions). Current Q_f: %s.",
                current_str,
            )
            return m.ResourceManagerAction.NO_ACTION

        best_action = m.ResourceManagerAction.NO_ACTION
        best_score = current_score
        best_qf_by_agent = current_qf

        action_evaluations = []
        for action in valid_actions:
            new_profiles = self.simulate_agent_profiles(sim, action)
            new_qf_by_agent: Dict[m.AgentId, float] = {}
            new_ttft_by_agent: Dict[m.AgentId, float] = {}
            for aid in m.AgentId:
                qf, ttft = self._aggregate(aid, new_profiles[aid], arrival_rates[aid])
                new_qf_by_agent[aid] = qf
                new_ttft_by_agent[aid] = ttft
            score = get_score(new_qf_by_agent, new_ttft_by_agent)

            qf_parts = ", ".join(
                f"{aid.name}: {new_qf_by_agent[aid]:.2f}" for aid in m.AgentId
            )
            action_evaluations.append(
                f"    {action.name:<30} -> {qf_parts} (feasible: {score[0] == 0}, "
                f"score: {score[1]:.2f})"
            )

            # Gate against the NO_ACTION baseline (not the running best) so
            # the min_quality_gain margin doesn't compound across candidates
            # -- among actions that clear the deadband, still pick whichever
            # scores best outright.
            if is_better(score, current_score) and score < best_score:
                best_score = score
                best_action = action
                best_qf_by_agent = new_qf_by_agent

        eval_str = "\n".join(action_evaluations)
        current_str = ", ".join(f"{aid.name}={current_qf[aid]:.2f}" for aid in m.AgentId)
        best_str = ", ".join(
            f"{aid.name}={best_qf_by_agent[aid]:.2f}" for aid in m.AgentId
        )
        logger.info(
            "[QAS] Evaluating actions.\n"
            "  Current Q_f      : %s\n"
            "  Evaluated Actions:\n%s\n"
            "  Action Taken     : %s\n"
            "  Resulting Q_f    : %s",
            current_str,
            eval_str,
            best_action.name,
            best_str,
        )

        return best_action
