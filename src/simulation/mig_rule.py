from typing import Dict, List, Optional, Tuple
from collections import defaultdict, Counter

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

        self._split_rules[m.MIGProfile.MIG_7G_40GB] = [mig_4_3, mig_3_2_2, mig_2_2_2_1]
        self._split_rules[m.MIGProfile.MIG_4G_20GB] = [mig_2_2]
        self._split_rules[m.MIGProfile.MIG_3G_20GB] = [mig_2_1]
        self._merge_rules[mig_4_3] = m.MIGProfile.MIG_7G_40GB
        self._merge_rules[mig_3_2_2] = m.MIGProfile.MIG_7G_40GB
        self._merge_rules[mig_2_2_2_1] = m.MIGProfile.MIG_7G_40GB
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
        for engines in by_gpu.values():
            by_mig: Dict[m.MIGProfile, List[m.LLMEngine]] = defaultdict(list)
            for e in engines:
                by_mig[e.mig_profile].append(e)

            for befores, after in self._merge_rules.items():
                needed = Counter(befores)
                selected_engs: List[m.LLMEngine] = []

                if all(
                    len(by_mig[profile]) >= count for profile, count in needed.items()
                ):
                    for mig, cnt in needed.items():
                        eng_by_load_inc = sorted(
                            by_mig[mig],
                            key=lambda e: len(e.running_queue) + len(e.waiting_queue),
                        )
                        selected_engs.extend(eng_by_load_inc[:cnt])
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

    def get_best_split(
        self, agent: m.Agent, desired_vram: Optional[float] = None
    ) -> Tuple[m.LLMEngine, List[m.MIGProfile]] | None:
        possibles = self.get_possible_splits(agent)
        if desired_vram:
            possibles = list(
                filter(lambda c: any(m.vram == desired_vram for m in c[1]), possibles)
            )
        possibles.sort(key=lambda c: len(c[1]))
        if possibles:
            return min(
                possibles,
                key=lambda c: len(c[0].waiting_queue) + len(c[0].running_queue),
            )
        return None

    def get_best_merge(
        self, agent: m.Agent, desired_vram: Optional[float] = None
    ) -> Tuple[List[m.LLMEngine], m.MIGProfile] | None:
        possibles = self.get_possible_merges(agent)
        if desired_vram:
            possibles = list(filter(lambda c: c[1].vram == desired_vram, possibles))
        possibles.sort(key=lambda c: len(c[0]))
        if possibles:
            return min(
                possibles,
                key=lambda c: sum(
                    len(e.waiting_queue) + len(e.running_queue) for e in c[0]
                ),
            )
