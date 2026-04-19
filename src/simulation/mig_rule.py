from collections import defaultdict, Counter
from typing import Dict, List, Tuple

import src.simulation.models as m


class MIGProfileRuleImpl(m.MIGProfileRule):
    """
    Possible Splits/Merges
    7 <-> (4,3)
    7 <-> (3,2,2)
    7 <-> (2,2,2,1)
    4 <-> (2,2)
    3 <-> (2,1)
    """

    _split_rules: Dict[m.MIGProfile, List[m.MIGConfigType]] = {}
    _merge_rules: Dict[m.MIGConfigType, m.MIGProfile] = {}

    def _normalize_value(self, v: m.MIGConfigType) -> m.MIGConfigType:
        # sort to non-increasing
        return tuple(sorted(v, key=lambda x: x.size, reverse=True))

    def __init__(self) -> None:
        # Populate possible splits/merges
        mig_4_3 = self._normalize_value(
            (m.MIGProfile.MIG_4G_20GB, m.MIGProfile.MIG_3G_20GB)
        )
        mig_3_2_2 = self._normalize_value(
            (
                m.MIGProfile.MIG_3G_20GB,
                m.MIGProfile.MIG_2G_10GB,
                m.MIGProfile.MIG_2G_10GB,
            )
        )
        mig_2_2_2_1 = self._normalize_value(
            (
                m.MIGProfile.MIG_2G_10GB,
                m.MIGProfile.MIG_2G_10GB,
                m.MIGProfile.MIG_2G_10GB,
                m.MIGProfile.MIG_1G_10GB,
            )
        )
        mig_2_2 = self._normalize_value(
            (m.MIGProfile.MIG_2G_10GB, m.MIGProfile.MIG_2G_10GB)
        )
        mig_2_1 = self._normalize_value(
            (m.MIGProfile.MIG_2G_10GB, m.MIGProfile.MIG_1G_10GB)
        )
        mig_4_2_1 = self._normalize_value(
            (
                m.MIGProfile.MIG_4G_20GB,
                m.MIGProfile.MIG_2G_10GB,
                m.MIGProfile.MIG_1G_10GB,
            )
        )

        self._split_rules[m.MIGProfile.MIG_7G_40GB] = [
            mig_4_3,
            mig_3_2_2,
            mig_2_2_2_1,
            mig_4_2_1,
        ]
        self._split_rules[m.MIGProfile.MIG_4G_20GB] = [mig_2_2]
        self._split_rules[m.MIGProfile.MIG_3G_20GB] = [mig_2_1]
        self._merge_rules[mig_4_3] = m.MIGProfile.MIG_7G_40GB
        self._merge_rules[mig_3_2_2] = m.MIGProfile.MIG_7G_40GB
        self._merge_rules[mig_2_2_2_1] = m.MIGProfile.MIG_7G_40GB
        self._merge_rules[mig_4_2_1] = m.MIGProfile.MIG_7G_40GB
        self._merge_rules[mig_2_2] = m.MIGProfile.MIG_4G_20GB
        self._merge_rules[mig_2_1] = m.MIGProfile.MIG_3G_20GB

    def get_possible_merges(
        self, agent: m.Agent
    ) -> List[Tuple[List[m.LLMEngine], m.MIGProfile]]:
        by_gpu: Dict[int, List[m.LLMEngine]] = defaultdict(list)
        for e in agent.engines:
            if e.status == m.EngineStatus.ACTIVE and not e.is_permanent:
                by_gpu[e.gpu].append(e)

        possible_merges: List[Tuple[List[m.LLMEngine], m.MIGProfile]] = []
        import itertools

        for engines in by_gpu.values():
            by_mig: Dict[m.MIGProfile, List[m.LLMEngine]] = defaultdict(list)
            for e in engines:
                by_mig[e.mig_profile].append(e)

            for befores, after in self._merge_rules.items():
                needed = Counter(befores)

                if all(
                    len(by_mig[profile]) >= count for profile, count in needed.items()
                ):
                    profile_combinations = []
                    for profile, count in needed.items():
                        profile_combinations.append(
                            list(itertools.combinations(by_mig[profile], count))
                        )

                    for combo_tuple in itertools.product(*profile_combinations):
                        selected_engs = [eng for combo in combo_tuple for eng in combo]
                        possible_merges.append((selected_engs, after))

        return possible_merges

    def get_possible_splits(
        self, agent: m.Agent
    ) -> List[Tuple[m.LLMEngine, List[m.MIGProfile]]]:
        by_gpu: Dict[int, List[m.LLMEngine]] = defaultdict(list)
        for e in agent.engines:
            if e.status == m.EngineStatus.ACTIVE and not e.is_permanent:
                by_gpu[e.gpu].append(e)

        possible_splits: List[Tuple[m.LLMEngine, List[m.MIGProfile]]] = []
        for engines in by_gpu.values():
            by_mig: Dict[m.MIGProfile, List[m.LLMEngine]] = defaultdict(list)
            for e in engines:
                by_mig[e.mig_profile].append(e)

            for before, afters in self._split_rules.items():
                if before in by_mig:
                    for mig_e in by_mig[before]:
                        for result_profiles in afters:
                            possible_splits.append((mig_e, list(result_profiles)))
        return possible_splits

    def get_best_specific_split(
        self, agent: m.Agent, target_profiles: Tuple[m.MIGProfile, ...]
    ) -> Tuple[m.LLMEngine, List[m.MIGProfile]] | None:
        possibles = self.get_possible_splits(agent)
        possibles = [c for c in possibles if tuple(c[1]) == target_profiles]
        possibles.sort(key=lambda c: len(c[1]))
        if possibles:
            return min(
                possibles,
                key=lambda c: len(c[0].waiting_queue) + len(c[0].running_queue),
            )
        return None

    def get_best_specific_merge(
        self, agent: m.Agent, target_profiles: Tuple[m.MIGProfile, ...]
    ) -> Tuple[List[m.LLMEngine], m.MIGProfile] | None:
        possibles = self.get_possible_merges(agent)
        possibles = [
            c
            for c in possibles
            if Counter(e.mig_profile for e in c[0]) == Counter(target_profiles)
        ]
        possibles.sort(key=lambda c: len(c[0]))
        if possibles:
            return min(
                possibles,
                key=lambda c: sum(
                    len(e.waiting_queue) + len(e.running_queue) for e in c[0]
                ),
            )
        return None

    def has_exact_match(self, agent: m.Agent, mig: m.MIGProfile) -> bool:
        return any(
            e.status == m.EngineStatus.ACTIVE
            and e.mig_profile == mig
            and not e.is_permanent
            for e in agent.engines
        )

    def get_best_exact_match(
        self,
        giver_aid: m.AgentId,
        mig: m.MIGProfile,
        receiver_aid: m.AgentId,  # Pass the ID of the agent RECEIVING the MIG
        all_engines: m.List[m.LLMEngine],
    ) -> m.LLMEngine | None:
        exact_matches = [
            e
            for e in all_engines
            if e.owner.agent_id == giver_aid
            and e.status == m.EngineStatus.ACTIVE
            and e.mig_profile == mig
            and not e.is_permanent
        ]
        if not exact_matches:
            return None

        def calculate_gathering_score(gpu: int) -> Tuple[int, int]:
            gpu_engines = [e for e in all_engines if e.gpu == gpu]

            # Current SM counts on this GPU
            sm_counts: Dict[m.AgentId, int] = defaultdict(int)
            for e in gpu_engines:
                sm_counts[e.owner.agent_id] += e.mig_profile.size

            # 1. Calculate Target Agent's SMs AFTER transfer
            target_sm_after = sm_counts[receiver_aid] + mig.size

            # 2. Calculate unique agents remaining on GPU AFTER transfer
            # We simulate the counts to see how many agents actually 'exist' post-transfer
            hypothetical_counts = sm_counts.copy()
            hypothetical_counts[receiver_aid] = (
                hypothetical_counts.get(receiver_aid, 0) + mig.size
            )
            hypothetical_counts[giver_aid] -= mig.size

            # Only count agents that still have more than 0 SMs
            final_agent_count = sum(
                1 for count in hypothetical_counts.values() if count > 0
            )

            # We want to MAXIMIZE target_sm_after and MINIMIZE final_agent_count
            # min() sorts ascending, so we negate the target_sm
            return (-target_sm_after, final_agent_count)

        return min(
            exact_matches,
            key=lambda e: (
                calculate_gathering_score(e.gpu),
                len(e.running_queue) + len(e.waiting_queue),
            ),
        )

    def select_best_split_action(
        self, agent: m.Agent, mask: List[bool], all_actions: List[m.ResourceManagerAction]
    ) -> m.ResourceManagerAction | None:
        best_action = None
        min_running = float("inf")

        for i, action in enumerate(all_actions):
            if (
                mask[i]
                and isinstance(action.value, m.MigAction)
                and action.value.action == "split"
                and action.value.victim == agent.agent_id
            ):
                res = self.get_best_specific_split(agent, action.value.profiles)
                if res:
                    eng, _ = res
                    running = len(eng.running_queue)
                    if running < min_running:
                        min_running = running
                        best_action = action
        return best_action

    def select_best_merge_action(
        self, agent: m.Agent, mask: List[bool], all_actions: List[m.ResourceManagerAction]
    ) -> m.ResourceManagerAction | None:
        best_action = None
        min_avg_running = float("inf")

        for i, action in enumerate(all_actions):
            if (
                mask[i]
                and isinstance(action.value, m.MigAction)
                and action.value.action == "merge"
                and action.value.victim == agent.agent_id
            ):
                res = self.get_best_specific_merge(agent, action.value.profiles)
                if res:
                    engs, _ = res
                    avg_running = sum(len(e.running_queue) for e in engs) / len(engs)
                    if avg_running < min_avg_running:
                        min_avg_running = avg_running
                        best_action = action
        return best_action

    def select_best_transfer_action(
        self,
        giver: m.Agent,
        receiver_aid: m.AgentId,
        mask: List[bool],
        all_actions: List[m.ResourceManagerAction],
        all_engines: List[m.LLMEngine],
    ) -> m.ResourceManagerAction | None:
        if giver.agent_id == receiver_aid:
            return None

        best_action = None
        min_total_q = float("inf")

        for i, action in enumerate(all_actions):
            if mask[i] and isinstance(action.value, m.VramTransferAction):
                val = action.value
                if val.giver == giver.agent_id and val.receiver == receiver_aid:
                    eng = self.get_best_exact_match(
                        giver.agent_id, val.mig, receiver_aid, all_engines
                    )
                    if eng:
                        total_q = len(eng.running_queue) + len(eng.waiting_queue)
                        if total_q < min_total_q:
                            min_total_q = total_q
                            best_action = action
        return best_action
