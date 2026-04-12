import math
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
        agent: m.Agent,
        mig: m.MIGProfile,
        all_engines: m.Optional[m.List[m.LLMEngine]] = None,
    ) -> m.LLMEngine | None:
        exact_matches = [
            e
            for e in agent.engines
            if e.status == m.EngineStatus.ACTIVE
            and e.mig_profile == mig
            and not e.is_permanent
        ]
        if not exact_matches:
            return None

        if all_engines is None:
            return min(
                exact_matches,
                key=lambda e: len(e.running_queue) + len(e.waiting_queue),
            )

        # Entropy-aware selection
        def calculate_gpu_entropy_after_transfer(
            gpu: int, engine_to_transfer: m.LLMEngine
        ) -> float:
            gpu_engines = [e for e in all_engines if e.gpu == gpu]
            # SM slice counts per agent
            sm_counts: Dict[m.AgentId, int] = defaultdict(int)
            total_sm = 0
            for e in gpu_engines:
                owner_id = e.owner.agent_id
                size = e.mig_profile.size
                if e.engine_id == engine_to_transfer.engine_id:
                    # Hypothetically transfer to the other agent
                    other_agent_id = (
                        m.AgentId.RAG
                        if owner_id == m.AgentId.CODING
                        else m.AgentId.CODING
                    )
                    sm_counts[other_agent_id] += size
                else:
                    sm_counts[owner_id] += size
                total_sm += size

            if total_sm == 0:
                return 0.0

            entropy = 0.0
            for count in sm_counts.values():
                p = count / total_sm
                if p > 0:
                    entropy -= p * math.log2(p)
            return entropy

        return min(
            exact_matches,
            key=lambda e: (
                calculate_gpu_entropy_after_transfer(e.gpu, e),
                len(e.running_queue) + len(e.waiting_queue),
            ),
        )
