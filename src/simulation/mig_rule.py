from typing import Dict, List, Tuple
from collections import defaultdict, Counter

from src.simulation.models import *


class MIGProfileRuleImpl(MIGProfileRule):
    """
    Possible Splits/Merges
    7 <-> (4,3)
    7 <-> (3,2,2)
    7 <-> (2,2,2,1)
    4 <-> (2,2)
    3 <-> (2,1)
    """

    _split_rules: Dict[MIGProfile, List[MIGConfigType]] = {}
    _merge_rules: Dict[MIGConfigType, MIGProfile] = {}

    def _normalize_value(self, v: MIGConfigType) -> MIGConfigType:
        # sort to non-increasing
        return tuple(sorted(v, key=lambda x: x.size, reverse=True))

    def __init__(self) -> None:
        # Populate possible splits/merges
        mig_4_3 = self._normalize_value(
            (MIGProfile.MIG_4G_20GB, MIGProfile.MIG_3G_20GB)
        )
        mig_3_2_2 = self._normalize_value(
            (MIGProfile.MIG_3G_20GB, MIGProfile.MIG_2G_10GB, MIGProfile.MIG_2G_10GB)
        )
        mig_2_2_2_1 = self._normalize_value(
            (
                MIGProfile.MIG_2G_10GB,
                MIGProfile.MIG_2G_10GB,
                MIGProfile.MIG_2G_10GB,
                MIGProfile.MIG_1G_10GB,
            )
        )
        mig_2_2 = self._normalize_value(
            (MIGProfile.MIG_2G_10GB, MIGProfile.MIG_2G_10GB)
        )
        mig_2_1 = self._normalize_value(
            (MIGProfile.MIG_2G_10GB, MIGProfile.MIG_1G_10GB)
        )

        self._split_rules[MIGProfile.MIG_7G_40GB] = [mig_4_3, mig_3_2_2, mig_2_2_2_1]
        self._split_rules[MIGProfile.MIG_4G_20GB] = [mig_2_2]
        self._split_rules[MIGProfile.MIG_3G_20GB] = [mig_2_1]
        self._merge_rules[mig_4_3] = MIGProfile.MIG_7G_40GB
        self._merge_rules[mig_3_2_2] = MIGProfile.MIG_7G_40GB
        self._merge_rules[mig_2_2_2_1] = MIGProfile.MIG_7G_40GB
        self._merge_rules[mig_2_2] = MIGProfile.MIG_4G_20GB
        self._merge_rules[mig_2_1] = MIGProfile.MIG_3G_20GB

    def get_possible_merges(
        self, agent: Agent
    ) -> List[Tuple[List[LLMEngine], MIGProfile]]:
        by_gpu: Dict[int, List[LLMEngine]] = defaultdict(list)
        for e in agent.engines:
            if e.status == EngineStatus.ACTIVE:
                by_gpu[e.gpu].append(e)

        possible_merges: List[Tuple[List[LLMEngine], MIGProfile]] = []
        for engines in by_gpu.values():

            by_mig: Dict[MIGProfile, List[LLMEngine]] = defaultdict(list)
            for e in engines:
                by_mig[e.mig_profile].append(e)

            for befores, after in self._merge_rules.items():
                needed = Counter(befores)
                selected_engs: List[LLMEngine] = []

                if all(
                    len(by_mig[profile]) >= count for profile, count in needed.items()
                ):
                    for m, cnt in needed.items():
                        eng_by_load_inc = sorted(
                            by_mig[m],
                            key=lambda e: len(e.running_queue) + len(e.waiting_queue),
                        )
                        selected_engs.extend(eng_by_load_inc[:cnt])
                    possible_merges.append((selected_engs, after))

        return possible_merges

    def get_possible_splits(
        self, agent: Agent
    ) -> List[Tuple[LLMEngine, List[MIGProfile]]]:
        by_gpu: Dict[int, List[LLMEngine]] = defaultdict(list)
        for e in agent.engines:
            if e.status == EngineStatus.ACTIVE:
                by_gpu[e.gpu].append(e)

        possible_splits: List[Tuple[LLMEngine, List[MIGProfile]]] = []
        for engines in by_gpu.values():

            by_mig: Dict[MIGProfile, List[LLMEngine]] = defaultdict(list)
            for e in engines:
                by_mig[e.mig_profile].append(e)

            for before, afters in self._split_rules.items():
                if before in by_mig:
                    for mig_e in by_mig[before]:
                        for result_profiles in afters:
                            possible_splits.append((mig_e, list(result_profiles)))
        return possible_splits
