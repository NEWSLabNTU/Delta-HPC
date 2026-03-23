from __future__ import annotations

from src.simulation.models import *
import src.simulation.global_vars as g


class WorkerImpl(Worker):
    def __init__(self, simulator: Simulator):
        self.simulator = simulator

    def start_transfer(self, current_time: float, details: TransferDetails):
        giver = self.simulator.agents[details.giver_id]
        receiver = self.simulator.agents[details.receiver_id]
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
            shutdown_payload: ShutdownReallocatePayload = {
                "engine_id": engine_to_shift.engine_id,
                "purpose": OperationPurpose.REALLOCATE,
                "receiver_id": receiver.agent_id,
            }
            self.simulator.logger.log_vram_transfer(
                current_time,
                giver.agent_id,
                receiver.agent_id,
                amount,
                [engine_to_shift.engine_id],
            )
            evt = engine_to_shift.trigger_shutdown(shutdown_payload, current_time)
            if evt:
                self.simulator.events.add(evt)
            return

        # Merge
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
            merge_payload: ShutdownMergePayload = {
                "engine_id": engs[0].engine_id,  # will be overwritten
                "purpose": OperationPurpose.MERGE,
                "merge_engine_ids": tuple(e.engine_id for e in engs),
                "drained_ids": [],
                "new_profile": new_profile,
                "agent_id": giver.agent_id,
                "gpu": engs[0].gpu,
                "receiver_id": receiver.agent_id,
            }
            for e in engs:
                per_engine_payload: ShutdownMergePayload = {
                    **merge_payload,
                    "engine_id": e.engine_id,
                }
                evt = e.trigger_shutdown(per_engine_payload, current_time)
                if evt:
                    self.simulator.events.add(evt)
            self.simulator.logger.log_mig_merge_trigger(
                current_time, [e.engine_id for e in engs], engs[0].gpu
            )
            self.simulator.logger.log_vram_transfer(
                current_time,
                giver.agent_id,
                receiver.agent_id,
                amount,
                [e.engine_id for e in engs],
            )
            return

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
                # if same workload, choose one with fewer result instance
            )

            mig_to_transfer = min(
                filter(lambda m: m.vram == amount, new_migs), key=lambda m: m.size
            )
            split_payload: ShutdownSplitPayload = {
                "engine_id": eng.engine_id,
                "purpose": OperationPurpose.SPLIT,
                "new_profiles": new_migs,
                "agent_id": giver.agent_id,
                "gpu": eng.gpu,
                "receiver_id": receiver.agent_id,
                "received_profile": mig_to_transfer,
            }
            evt = eng.trigger_shutdown(split_payload, current_time)
            if evt:
                self.simulator.events.add(evt)
            self.simulator.logger.log_mig_split_trigger(
                current_time, eng.engine_id, eng.gpu
            )
            self.simulator.logger.log_vram_transfer(
                current_time, giver.agent_id, receiver.agent_id, amount, [eng.engine_id]
            )
            return

        # 4. Give up — log and discard
        self.simulator.logger.log_discard_vram_transfer(current_time, details)
