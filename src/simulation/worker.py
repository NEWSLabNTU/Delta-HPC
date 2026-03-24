from __future__ import annotations

from typing import Tuple, Dict, Any

from src.simulation.models import *
import src.simulation.global_vars as g


class WorkerImpl(Worker):
    def __init__(self):
        pass

    def transfer(
        self,
        current_time: float,
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
            if e.status == EngineStatus.ACTIVE and e.mig_profile.vram == amount
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
        merge_candidates = [
            (engs, new_profile)
            for engs, new_profile in g.MIG_RULES.get_possible_merges(giver)
            if new_profile.vram == amount
        ]
        if merge_candidates:
            # Among qualifying pairs, prefer lowest combined queue load
            engs, new_profile = min(
                merge_candidates,
                key=lambda c: sum(
                    len(e.running_queue) + len(e.waiting_queue) for e in c[0]
                ),
            )
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
        split_candidates = [
            (eng, migs)
            for eng, migs in g.MIG_RULES.get_possible_splits(giver)
            if amount in [mig.vram for mig in migs]
        ]
        if split_candidates:
            eng, new_migs = min(
                split_candidates,
                key=lambda c: (
                    len(c[0].running_queue) + len(c[0].waiting_queue),
                    len(c[1]),
                ),
            )

            mig_to_transfer = min(
                filter(lambda m: m.vram == amount, new_migs), key=lambda m: m.size
            )
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
