from __future__ import annotations

from typing import Tuple, Dict, Any

from src.simulation.models import *
import src.simulation.utils as utils


class WorkerImpl(Worker):
    def __init__(self):
        pass

    def transfer(
        self,
        details: TransferDetails,
        agents: Dict[AgentId, Agent],
    ) -> Tuple[str, Any] | None:
        giver = agents[details.giver_id]
        receiver = agents[details.receiver_id]
        amount = details.amount

        # 1. Exact-match: hand off an engine whose VRAM == requested amount
        exact_matches = [
            e
            for e in giver.engines
            if e.status == EngineStatus.ACTIVE
            and e.mig_profile.vram == amount
            and not e.is_permanent
        ]
        if exact_matches:
            engine_to_shift = min(
                exact_matches,
                key=lambda e: len(e.running_queue) + len(e.waiting_queue),
            )
            return (
                "exact",
                {
                    "engine": engine_to_shift,
                    "giver": giver,
                    "receiver": receiver,
                    "amount": amount,
                },
            )

        # 2. Merge
        merge_candidates = utils.MIG_RULES.get_best_merge(giver, amount)
        if merge_candidates:
            engs, new_profile = merge_candidates
            return (
                "merge",
                {
                    "engines": engs,
                    "new_profile": new_profile,
                    "giver": giver,
                    "receiver": receiver,
                    "amount": amount,
                },
            )

        # 3. Split
        split_candidates = utils.MIG_RULES.get_best_split(giver, amount)
        if split_candidates:
            eng, new_migs = split_candidates
            mig_to_transfer = min(new_migs, key=lambda m: m.size)
            return (
                "split",
                {
                    "engine": eng,
                    "new_profiles": new_migs,
                    "mig_to_transfer": mig_to_transfer,
                    "giver": giver,
                    "receiver": receiver,
                    "amount": amount,
                },
            )

        return None
