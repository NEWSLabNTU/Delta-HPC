from typing import Dict

import src.simulation.models as m
import src.simulation.utils as utils
from src.bench.config import BENCH_CONFIG
from src.training.config import TRAINING_CONFIG


class RuleBasedHeuristic:
    def __init__(self):
        pass

    def _denormalize_arrival_rate(self, val: float) -> float:
        return val * TRAINING_CONFIG.norm_arrival_rate

    def decide_action(self, sim: m.Simulator) -> m.ResourceManagerAction:
        state = sim.get_state()
        mask = sim.get_action_mask()
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
            arrival_rates[aid] = arr_rate

            agent = sim.agents[aid]
            srv_rate = sum(
                BENCH_CONFIG.get_service_rate(aid, e.mig_profile)
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
            return m.ResourceManagerAction.NO_ACTION

        def simulate_service_rates(
            action: m.ResourceManagerAction,
        ) -> Dict[m.AgentId, float]:
            new_rates = service_rates.copy()
            val = action.value

            if isinstance(val, m.VramTransferAction):
                giver_rate = BENCH_CONFIG.get_service_rate(val.giver, val.mig)
                receiver_rate = BENCH_CONFIG.get_service_rate(val.receiver, val.mig)
                new_rates[val.giver] -= giver_rate
                new_rates[val.receiver] += receiver_rate

            elif isinstance(val, m.MigAction):
                victim = val.victim
                receiver = val.receiver

                if val.action == "split":
                    best_split = utils.MIG_RULES.get_best_specific_split(
                        sim.agents[victim], val.profiles
                    )
                    if best_split:
                        eng, new_profiles = best_split
                        new_rates[victim] -= BENCH_CONFIG.get_service_rate(
                            victim, eng.mig_profile
                        )
                        for p in new_profiles:
                            if receiver is not None and p == val.transfer_profile:
                                new_rates[receiver] += BENCH_CONFIG.get_service_rate(
                                    receiver, p
                                )
                            else:
                                new_rates[victim] += BENCH_CONFIG.get_service_rate(
                                    victim, p
                                )

                elif val.action == "merge":
                    best_merge = utils.MIG_RULES.get_best_specific_merge(
                        sim.agents[victim], val.profiles
                    )
                    if best_merge:
                        engs, new_profile = best_merge
                        for eng in engs:
                            new_rates[victim] -= BENCH_CONFIG.get_service_rate(
                                victim, eng.mig_profile
                            )

                        if receiver is not None and new_profile == val.transfer_profile:
                            new_rates[receiver] += BENCH_CONFIG.get_service_rate(
                                receiver, new_profile
                            )
                        else:
                            new_rates[victim] += BENCH_CONFIG.get_service_rate(
                                victim, new_profile
                            )

            for aid in m.AgentId:
                new_rates[aid] = max(0.0, new_rates[aid])

            return new_rates

        best_action = m.ResourceManagerAction.NO_ACTION

        def get_deviation(ratios: Dict[m.AgentId, float]) -> float:
            return sum(
                abs(r - 1.0) if r != float("inf") else 1000.0 for r in ratios.values()
            )

        current_deviation = get_deviation(scaling_ratios)
        best_deviation = current_deviation
        # best_ratios = scaling_ratios

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

            if action_deviation < best_deviation:
                best_deviation = action_deviation
                best_action = action
                # best_ratios = new_ratios

        # if best_action != m.ResourceManagerAction.NO_ACTION:
        # mig_config_strs = []
        # for aid in m.AgentId:
        #     profiles = [
        #         e.mig_profile.string
        #         for e in sim.agents[aid].engines
        #         if e.status != m.EngineStatus.BOOTING
        #     ]
        #     mig_config_strs.append(f"{aid.name}: [{', '.join(profiles)}]")
        # current_config_str = " | ".join(mig_config_strs)

        # log_str = (
        #     f"[Heuristic] Scaling Triggered!\n"
        #     f"  Current Config : {current_config_str}\n"
        #     f"  Current Ratios : Coding={scaling_ratios[m.AgentId.CODING]:.2f}, RAG={scaling_ratios[m.AgentId.RAG]:.2f}\n"
        #     f"  Action Taken   : {best_action.name}\n"
        #     f"  Expected Ratios: Coding={best_ratios[m.AgentId.CODING]:.2f}, RAG={best_ratios[m.AgentId.RAG]:.2f}"
        # )
        # print(log_str)

        return best_action
