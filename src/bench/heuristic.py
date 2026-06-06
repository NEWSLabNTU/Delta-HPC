import math
import logging
from typing import Dict

import src.share.models as m
from src.share.mig_matrix import STATE_DEFINITIONS
from src.bench.config import BENCH_CONFIG
from src.training.config import TRAINING_CONFIG
from src.simulation.config import GPU_MIG_PROFILE

logger = logging.getLogger(__name__)


class RuleBasedHeuristic:
    def __init__(self, get_service_rate=None):
        if get_service_rate is None:
            self.get_service_rate = BENCH_CONFIG.get_service_rate
        else:
            self.get_service_rate = get_service_rate

    def _denormalize_arrival_rate(self, val: float) -> float:
        return val * TRAINING_CONFIG.norm_arrival_rate

    def decide_action(self, sim: m.Simulator) -> m.ResourceManagerAction:
        state = sim.get_state()
        mask = sim.get_action_mask(ignore_cooldowns=True)
        all_actions = list(m.ResourceManagerAction)
        valid_actions = [
            a
            for i, a in enumerate(all_actions)
            if mask[i] and a != m.ResourceManagerAction.NO_ACTION
        ]

        util_factor = BENCH_CONFIG.utilization_factor
        high_thresh = BENCH_CONFIG.high_threshold
        low_thresh = BENCH_CONFIG.low_threshold

        arrival_rates: Dict[m.AgentId, float] = {}
        service_rates: Dict[m.AgentId, float] = {}
        scaling_ratios: Dict[m.AgentId, float] = {}

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

            agent = sim.agents[aid]
            srv_rate = sum(
                self.get_service_rate(aid, e.mig_profile, gpu_id=e.gpu)
                for e in agent.engines
                if e.status != m.EngineStatus.BOOTING
            )
            service_rates[aid] = srv_rate

            if srv_rate > 0:
                scaling_ratios[aid] = arr_rate / (util_factor * srv_rate)
            else:
                scaling_ratios[aid] = float("inf")

        # Check if any agent needs scaling
        needs_action = any(
            scaling_ratios[aid] > high_thresh or scaling_ratios[aid] < low_thresh
            for aid in m.AgentId
        )

        if not needs_action or not valid_actions:
            ratio_str = ", ".join([
                f"{k.name}: {v:.2f}" for k, v in scaling_ratios.items()
            ])
            logger.info(
                f"Heuristic deciding NO_ACTION. Current scaling ratios: {ratio_str}"
            )
            return m.ResourceManagerAction.NO_ACTION

        def simulate_service_rates(
            action: m.ResourceManagerAction,
        ) -> Dict[m.AgentId, float]:
            new_rates = service_rates.copy()
            sim_action = sim.map_to_action(action)
            if sim_action is None:
                return new_rates

            gpu_id = sim_action.gpu_id

            # 1. Remove service rates of source engines
            for idx in sim_action.mig_src:
                eng = sim.gpu_engines[gpu_id][idx]
                # Decrease capacity for giver
                new_rates[eng.owner.agent_id] -= self.get_service_rate(
                    eng.owner.agent_id, eng.mig_profile, gpu_id=gpu_id
                )

            # 2. Add service rates of target engines
            # Identify the "giver" (current owner of source engines)
            giver_id = sim.gpu_engines[gpu_id][sim_action.mig_src[0]].owner.agent_id

            if sim_action.target_state_id is None:
                # Pure transfer: no state change
                for idx in sim_action.mig_src:
                    eng = sim.gpu_engines[gpu_id][idx]
                    owner_id = giver_id
                    if sim_action.receiver and sim_action.receiver.mig_idx == idx:
                        owner_id = sim_action.receiver.receiver_id

                    new_rates[owner_id] += self.get_service_rate(
                        owner_id, eng.mig_profile, gpu_id=gpu_id
                    )
            else:
                target_profiles = STATE_DEFINITIONS[sim_action.target_state_id]
                for idx in sim_action.mig_target:
                    logical_profile = target_profiles[idx]
                    # Find corresponding hardware profile
                    hardware_profile = None
                    for hp in GPU_MIG_PROFILE[gpu_id]:
                        if hp.profile_type == logical_profile:
                            hardware_profile = hp
                            break
                    assert hardware_profile is not None

                    owner_id = giver_id
                    if sim_action.receiver and sim_action.receiver.mig_idx == idx:
                        owner_id = sim_action.receiver.receiver_id

                    # Add new capacity for whoever owns this new profile
                    new_rates[owner_id] += self.get_service_rate(
                        owner_id, hardware_profile, gpu_id=gpu_id
                    )

            for aid in m.AgentId:
                new_rates[aid] = max(0.0, new_rates[aid])

            return new_rates

        best_action = m.ResourceManagerAction.NO_ACTION

        def get_deviation(ratios: Dict[m.AgentId, float]) -> float:
            dev = 0.0
            for r in ratios.values():
                if r == float("inf"):
                    dev += 1000.0
                else:
                    dev += math.exp(abs(r - 1.0)) - 1.0
            return dev

        current_deviation = get_deviation(scaling_ratios)
        best_deviation = current_deviation
        best_ratios = scaling_ratios

        action_evaluations = []

        for action in valid_actions:
            new_service_rates = simulate_service_rates(action)

            new_ratios = {}
            for aid in m.AgentId:
                if new_service_rates[aid] > 0:
                    new_ratios[aid] = arrival_rates[aid] / (
                        util_factor * new_service_rates[aid]
                    )
                else:
                    new_ratios[aid] = float("inf")

            action_deviation = get_deviation(new_ratios)

            ratio_parts = ", ".join(
                f"{aid.name}: {new_ratios[aid]:.2f}" for aid in m.AgentId
            )
            action_evaluations.append(
                f"    {action.name:<30} -> {ratio_parts} (dev: {action_deviation:.2f})"
            )

            if action_deviation < best_deviation:
                best_deviation = action_deviation
                best_action = action
                best_ratios = new_ratios

        eval_str = "\n".join(action_evaluations)

        current_ratio_str = ", ".join(
            f"{aid.name}={scaling_ratios[aid]:.2f}" for aid in m.AgentId
        )
        best_ratio_str = ", ".join(
            f"{aid.name}={best_ratios[aid]:.2f}" for aid in m.AgentId
        )
        logger.info(
            "[Heuristic] Scaling Triggered!\n"
            "  Current Ratios : %s\n"
            "  Evaluated Actions:\n%s\n"
            "  Action Taken   : %s\n"
            "  Expected Ratios: %s",
            current_ratio_str,
            eval_str,
            best_action.name,
            best_ratio_str,
        )

        return best_action
