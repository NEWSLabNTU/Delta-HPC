from __future__ import annotations

import random
from typing import Any, List, Dict, Tuple

import src.simulation.utils as utils
import src.simulation.models as m


class ResourceManager:
    def __init__(self):
        self.trigger_interval = 3600.0
        self.next_vram_transfer_time = 1800.0
        self.next_mig_trigger_time = 3600.0

    def act(
        self,
        current_time: float,
        state: m.EnvironmentStateData,
        agents: Dict[m.AgentId, m.Agent],
    ) -> Tuple[m.TransferDetails | None, Tuple[str, Any] | None]:
        """
        Periodically checks if it is time to trigger VRAM transfer or MIG split/merge.
        """
        mig_decision = None
        vram_transfer_details = None

        if current_time >= self.next_mig_trigger_time:
            mig_decision = self.trigger_mig(current_time, agents)
            self.next_mig_trigger_time += self.trigger_interval

        if current_time >= self.next_vram_transfer_time:
            vram_transfer_details = self.trigger_vram_transfer(current_time, agents)
            self.next_vram_transfer_time += self.trigger_interval

        return vram_transfer_details, mig_decision

    def trigger_vram_transfer(
        self, current_time: float, agents: Dict[m.AgentId, m.Agent]
    ) -> m.TransferDetails | None:
        """
        Periodically transfers VRAM between agents.
        Randomly selects an agent. If it has >20GB active VRAM, transfers 10 or 20GB.
        If it has exactly 20GB, transfers 10GB.
        Otherwise, it attempts the same logic on the other agent.
        """
        agents_list = list(agents.values())
        if len(agents_list) < 2:
            return None

        random.shuffle(agents_list)

        for i in range(2):
            giver = agents_list[i]
            receiver = agents_list[1 - i]

            active_vram = sum(
                e.mig_profile.vram
                for e in giver.engines
                if e.status == m.EngineStatus.ACTIVE
            )

            amount = 0
            if active_vram > 20:
                amount = random.choice([10, 20])
            elif active_vram == 20:
                amount = 10

            if amount > 0:
                return m.TransferDetails(
                    amount, giver_id=giver.agent_id, receiver_id=receiver.agent_id
                )
        return None

    def trigger_mig(
        self, current_time: float, agents: Dict[m.AgentId, m.Agent]
    ) -> Tuple[str, Any] | None:
        candidates: List[Tuple[str, Any]] = []  # List of (action type, data)
        for agent in agents.values():
            # 1. Look for Merges
            possible_merges = utils.MIG_RULES.get_possible_merges(agent)
            for engs, new_profile in possible_merges:
                candidates.append(
                    (
                        "merge",
                        {
                            "engines": engs,
                            "new_profile": new_profile,
                            "agent_id": agent.agent_id,
                            "gpu": engs[0].gpu,
                        },
                    )
                )

            # 2. Look for Splits
            possible_splits = utils.MIG_RULES.get_possible_splits(agent)
            for eng, new_profiles in possible_splits:
                candidates.append(
                    (
                        "split",
                        {
                            "engine": eng,
                            "new_profiles": new_profiles,
                            "agent_id": agent.agent_id,
                            "gpu": eng.gpu,
                        },
                    )
                )

        if not candidates:
            return None

        return random.choice(candidates)
