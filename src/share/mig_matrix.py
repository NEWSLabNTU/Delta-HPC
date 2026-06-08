import dataclasses
from typing import Tuple, Dict, Callable, Optional

import src.share.models as m

STATE_DEFINITIONS: Dict[int, Tuple[m.MIGProfile, ...]] = {
    1: (m.MIGProfile.MIG_7G,),
    2: (
        m.MIGProfile.MIG_4G,
        m.MIGProfile.MIG_3G,
    ),
    3: (
        m.MIGProfile.MIG_4G,
        m.MIGProfile.MIG_2G,
        m.MIGProfile.MIG_1G_LARGE,
    ),
    4: (
        m.MIGProfile.MIG_4G,
        m.MIGProfile.MIG_1G_SMALL,
        m.MIGProfile.MIG_1G_SMALL,
        m.MIGProfile.MIG_1G_LARGE,
    ),
    8: (
        m.MIGProfile.MIG_2G,
        m.MIGProfile.MIG_2G,
        m.MIGProfile.MIG_3G,
    ),
    9: (
        m.MIGProfile.MIG_2G,
        m.MIGProfile.MIG_1G_SMALL,
        m.MIGProfile.MIG_1G_SMALL,
        m.MIGProfile.MIG_3G,
    ),
    10: (
        m.MIGProfile.MIG_1G_SMALL,
        m.MIGProfile.MIG_1G_SMALL,
        m.MIGProfile.MIG_2G,
        m.MIGProfile.MIG_3G,
    ),
    11: (
        m.MIGProfile.MIG_1G_SMALL,
        m.MIGProfile.MIG_1G_SMALL,
        m.MIGProfile.MIG_1G_SMALL,
        m.MIGProfile.MIG_1G_SMALL,
        m.MIGProfile.MIG_3G,
    ),
    12: (
        m.MIGProfile.MIG_2G,
        m.MIGProfile.MIG_2G,
        m.MIGProfile.MIG_2G,
        m.MIGProfile.MIG_1G_LARGE,
    ),
    13: (
        m.MIGProfile.MIG_2G,
        m.MIGProfile.MIG_1G_SMALL,
        m.MIGProfile.MIG_1G_SMALL,
        m.MIGProfile.MIG_2G,
        m.MIGProfile.MIG_1G_LARGE,
    ),
    14: (
        m.MIGProfile.MIG_1G_SMALL,
        m.MIGProfile.MIG_1G_SMALL,
        m.MIGProfile.MIG_2G,
        m.MIGProfile.MIG_2G,
        m.MIGProfile.MIG_1G_LARGE,
    ),
    15: (
        m.MIGProfile.MIG_2G,
        m.MIGProfile.MIG_1G_SMALL,
        m.MIGProfile.MIG_1G_SMALL,
        m.MIGProfile.MIG_1G_SMALL,
        m.MIGProfile.MIG_1G_SMALL,
        m.MIGProfile.MIG_1G_LARGE,
    ),
    16: (
        m.MIGProfile.MIG_1G_SMALL,
        m.MIGProfile.MIG_1G_SMALL,
        m.MIGProfile.MIG_2G,
        m.MIGProfile.MIG_1G_SMALL,
        m.MIGProfile.MIG_1G_SMALL,
        m.MIGProfile.MIG_1G_LARGE,
    ),
    17: (
        m.MIGProfile.MIG_1G_SMALL,
        m.MIGProfile.MIG_1G_SMALL,
        m.MIGProfile.MIG_1G_SMALL,
        m.MIGProfile.MIG_1G_SMALL,
        m.MIGProfile.MIG_2G,
        m.MIGProfile.MIG_1G_LARGE,
    ),
    19: (
        m.MIGProfile.MIG_1G_SMALL,
        m.MIGProfile.MIG_1G_SMALL,
        m.MIGProfile.MIG_1G_SMALL,
        m.MIGProfile.MIG_1G_SMALL,
        m.MIGProfile.MIG_1G_SMALL,
        m.MIGProfile.MIG_1G_SMALL,
        m.MIGProfile.MIG_1G_LARGE,
    ),
}

STATE_ID_MAP = {sid: i for i, sid in enumerate(sorted(STATE_DEFINITIONS.keys()))}

SLICE_MAPPING = {
    1: [[0, 1, 2, 3, 4, 5, 6]],
    2: [[0, 1, 2, 3], [4, 5, 6]],
    3: [[0, 1, 2, 3], [4, 5], [6]],
    4: [[0, 1, 2, 3], [4], [5], [6]],
    8: [[0, 1], [2, 3], [4, 5, 6]],
    9: [[0, 1], [2], [3], [4, 5, 6]],
    10: [[0], [1], [2, 3], [4, 5, 6]],
    11: [[0], [1], [2], [3], [4, 5, 6]],
    12: [[0, 1], [2, 3], [4, 5], [6]],
    13: [[0, 1], [2], [3], [4, 5], [6]],
    14: [[0], [1], [2, 3], [4, 5], [6]],
    15: [[0, 1], [2], [3], [4], [5], [6]],
    16: [[0], [1], [2, 3], [4], [5], [6]],
    17: [[0], [1], [2], [3], [4, 5], [6]],
    19: [[0], [1], [2], [3], [4], [5], [6]],
}

TRANSITION_MATRIX = {
    (1, 2): m.Action(
        action=m.ActionType.SPLIT,
        gpu_id=-1,
        mig_src=[0],
        mig_target=[0, 1],
        target_state_id=2,
    ),
    (1, 3): m.Action(
        action=m.ActionType.SPLIT,
        gpu_id=-1,
        mig_src=[0],
        mig_target=[0, 1, 2],
        target_state_id=3,
    ),
    (1, 4): m.Action(
        action=m.ActionType.SPLIT,
        gpu_id=-1,
        mig_src=[0],
        mig_target=[0, 1, 2, 3],
        target_state_id=4,
    ),
    (1, 8): m.Action(
        action=m.ActionType.SPLIT,
        gpu_id=-1,
        mig_src=[0],
        mig_target=[0, 1, 2],
        target_state_id=8,
    ),
    (1, 9): m.Action(
        action=m.ActionType.SPLIT,
        gpu_id=-1,
        mig_src=[0],
        mig_target=[0, 1, 2, 3],
        target_state_id=9,
    ),
    (1, 10): m.Action(
        action=m.ActionType.SPLIT,
        gpu_id=-1,
        mig_src=[0],
        mig_target=[0, 1, 2, 3],
        target_state_id=10,
    ),
    (1, 11): m.Action(
        action=m.ActionType.SPLIT,
        gpu_id=-1,
        mig_src=[0],
        mig_target=[0, 1, 2, 3, 4],
        target_state_id=11,
    ),
    (1, 12): m.Action(
        action=m.ActionType.SPLIT,
        gpu_id=-1,
        mig_src=[0],
        mig_target=[0, 1, 2, 3],
        target_state_id=12,
    ),
    (1, 13): m.Action(
        action=m.ActionType.SPLIT,
        gpu_id=-1,
        mig_src=[0],
        mig_target=[0, 1, 2, 3, 4],
        target_state_id=13,
    ),
    (1, 14): m.Action(
        action=m.ActionType.SPLIT,
        gpu_id=-1,
        mig_src=[0],
        mig_target=[0, 1, 2, 3, 4],
        target_state_id=14,
    ),
    (1, 15): m.Action(
        action=m.ActionType.SPLIT,
        gpu_id=-1,
        mig_src=[0],
        mig_target=[0, 1, 2, 3, 4, 5],
        target_state_id=15,
    ),
    (1, 16): m.Action(
        action=m.ActionType.SPLIT,
        gpu_id=-1,
        mig_src=[0],
        mig_target=[0, 1, 2, 3, 4, 5],
        target_state_id=16,
    ),
    (1, 17): m.Action(
        action=m.ActionType.SPLIT,
        gpu_id=-1,
        mig_src=[0],
        mig_target=[0, 1, 2, 3, 4, 5],
        target_state_id=17,
    ),
    (1, 19): m.Action(
        action=m.ActionType.SPLIT,
        gpu_id=-1,
        mig_src=[0],
        mig_target=[0, 1, 2, 3, 4, 5, 6],
        target_state_id=19,
    ),
    (2, 1): m.Action(
        action=m.ActionType.MERGE,
        gpu_id=-1,
        mig_src=[0, 1],
        mig_target=[0],
        target_state_id=1,
    ),
    (2, 3): m.Action(
        action=m.ActionType.SPLIT,
        gpu_id=-1,
        mig_src=[1],
        mig_target=[1, 2],
        target_state_id=3,
    ),
    (2, 4): m.Action(
        action=m.ActionType.SPLIT,
        gpu_id=-1,
        mig_src=[1],
        mig_target=[1, 2, 3],
        target_state_id=4,
    ),
    (2, 8): m.Action(
        action=m.ActionType.SPLIT,
        gpu_id=-1,
        mig_src=[0],
        mig_target=[0, 1],
        target_state_id=8,
    ),
    (2, 9): m.Action(
        action=m.ActionType.SPLIT,
        gpu_id=-1,
        mig_src=[0],
        mig_target=[0, 1, 2],
        target_state_id=9,
    ),
    (2, 10): m.Action(
        action=m.ActionType.SPLIT,
        gpu_id=-1,
        mig_src=[0],
        mig_target=[0, 1, 2],
        target_state_id=10,
    ),
    (2, 11): m.Action(
        action=m.ActionType.SPLIT,
        gpu_id=-1,
        mig_src=[0],
        mig_target=[0, 1, 2, 3],
        target_state_id=11,
    ),
    (3, 1): m.Action(
        action=m.ActionType.MERGE,
        gpu_id=-1,
        mig_src=[0, 1, 2],
        mig_target=[0],
        target_state_id=1,
    ),
    (3, 2): m.Action(
        action=m.ActionType.MERGE,
        gpu_id=-1,
        mig_src=[1, 2],
        mig_target=[1],
        target_state_id=2,
    ),
    (3, 4): m.Action(
        action=m.ActionType.SPLIT,
        gpu_id=-1,
        mig_src=[1],
        mig_target=[1, 2],
        target_state_id=4,
    ),
    (3, 12): m.Action(
        action=m.ActionType.SPLIT,
        gpu_id=-1,
        mig_src=[0],
        mig_target=[0, 1],
        target_state_id=12,
    ),
    (3, 13): m.Action(
        action=m.ActionType.SPLIT,
        gpu_id=-1,
        mig_src=[0],
        mig_target=[0, 1, 2],
        target_state_id=13,
    ),
    (3, 14): m.Action(
        action=m.ActionType.SPLIT,
        gpu_id=-1,
        mig_src=[0],
        mig_target=[0, 1, 2],
        target_state_id=14,
    ),
    (3, 17): m.Action(
        action=m.ActionType.SPLIT,
        gpu_id=-1,
        mig_src=[0],
        mig_target=[0, 1, 2, 3],
        target_state_id=17,
    ),
    (4, 1): m.Action(
        action=m.ActionType.MERGE,
        gpu_id=-1,
        mig_src=[0, 1, 2, 3],
        mig_target=[0],
        target_state_id=1,
    ),
    (4, 2): m.Action(
        action=m.ActionType.MERGE,
        gpu_id=-1,
        mig_src=[1, 2, 3],
        mig_target=[1],
        target_state_id=2,
    ),
    (4, 3): m.Action(
        action=m.ActionType.MERGE,
        gpu_id=-1,
        mig_src=[1, 2],
        mig_target=[1],
        target_state_id=3,
    ),
    (4, 15): m.Action(
        action=m.ActionType.SPLIT,
        gpu_id=-1,
        mig_src=[0],
        mig_target=[0, 1, 2],
        target_state_id=15,
    ),
    (4, 16): m.Action(
        action=m.ActionType.SPLIT,
        gpu_id=-1,
        mig_src=[0],
        mig_target=[0, 1, 2],
        target_state_id=16,
    ),
    (4, 19): m.Action(
        action=m.ActionType.SPLIT,
        gpu_id=-1,
        mig_src=[0],
        mig_target=[0, 1, 2, 3],
        target_state_id=19,
    ),
    (8, 1): m.Action(
        action=m.ActionType.MERGE,
        gpu_id=-1,
        mig_src=[0, 1, 2],
        mig_target=[0],
        target_state_id=1,
    ),
    (8, 2): m.Action(
        action=m.ActionType.MERGE,
        gpu_id=-1,
        mig_src=[0, 1],
        mig_target=[0],
        target_state_id=2,
    ),
    (8, 9): m.Action(
        action=m.ActionType.SPLIT,
        gpu_id=-1,
        mig_src=[1],
        mig_target=[1, 2],
        target_state_id=9,
    ),
    (8, 10): m.Action(
        action=m.ActionType.SPLIT,
        gpu_id=-1,
        mig_src=[0],
        mig_target=[0, 1],
        target_state_id=10,
    ),
    (8, 12): m.Action(
        action=m.ActionType.SPLIT,
        gpu_id=-1,
        mig_src=[2],
        mig_target=[2, 3],
        target_state_id=12,
    ),
    (9, 1): m.Action(
        action=m.ActionType.MERGE,
        gpu_id=-1,
        mig_src=[0, 1, 2, 3],
        mig_target=[0],
        target_state_id=1,
    ),
    (9, 2): m.Action(
        action=m.ActionType.MERGE,
        gpu_id=-1,
        mig_src=[0, 1, 2],
        mig_target=[0],
        target_state_id=2,
    ),
    (9, 8): m.Action(
        action=m.ActionType.MERGE,
        gpu_id=-1,
        mig_src=[1, 2],
        mig_target=[1],
        target_state_id=8,
    ),
    (9, 11): m.Action(
        action=m.ActionType.SPLIT,
        gpu_id=-1,
        mig_src=[0],
        mig_target=[0, 1],
        target_state_id=11,
    ),
    (9, 13): m.Action(
        action=m.ActionType.SPLIT,
        gpu_id=-1,
        mig_src=[3],
        mig_target=[3, 4],
        target_state_id=13,
    ),
    (9, 15): m.Action(
        action=m.ActionType.SPLIT,
        gpu_id=-1,
        mig_src=[3],
        mig_target=[3, 4, 5],
        target_state_id=15,
    ),
    (10, 1): m.Action(
        action=m.ActionType.MERGE,
        gpu_id=-1,
        mig_src=[0, 1, 2, 3],
        mig_target=[0],
        target_state_id=1,
    ),
    (10, 2): m.Action(
        action=m.ActionType.MERGE,
        gpu_id=-1,
        mig_src=[0, 1, 2],
        mig_target=[0],
        target_state_id=2,
    ),
    (10, 8): m.Action(
        action=m.ActionType.MERGE,
        gpu_id=-1,
        mig_src=[0, 1],
        mig_target=[0],
        target_state_id=8,
    ),
    (10, 11): m.Action(
        action=m.ActionType.SPLIT,
        gpu_id=-1,
        mig_src=[2],
        mig_target=[2, 3],
        target_state_id=11,
    ),
    (10, 14): m.Action(
        action=m.ActionType.SPLIT,
        gpu_id=-1,
        mig_src=[3],
        mig_target=[3, 4],
        target_state_id=14,
    ),
    (10, 16): m.Action(
        action=m.ActionType.SPLIT,
        gpu_id=-1,
        mig_src=[3],
        mig_target=[3, 4, 5],
        target_state_id=16,
    ),
    (11, 1): m.Action(
        action=m.ActionType.MERGE,
        gpu_id=-1,
        mig_src=[0, 1, 2, 3, 4],
        mig_target=[0],
        target_state_id=1,
    ),
    (11, 2): m.Action(
        action=m.ActionType.MERGE,
        gpu_id=-1,
        mig_src=[0, 1, 2, 3],
        mig_target=[0],
        target_state_id=2,
    ),
    (11, 9): m.Action(
        action=m.ActionType.MERGE,
        gpu_id=-1,
        mig_src=[0, 1],
        mig_target=[0],
        target_state_id=9,
    ),
    (11, 10): m.Action(
        action=m.ActionType.MERGE,
        gpu_id=-1,
        mig_src=[2, 3],
        mig_target=[2],
        target_state_id=10,
    ),
    (11, 17): m.Action(
        action=m.ActionType.SPLIT,
        gpu_id=-1,
        mig_src=[4],
        mig_target=[4, 5],
        target_state_id=17,
    ),
    (11, 19): m.Action(
        action=m.ActionType.SPLIT,
        gpu_id=-1,
        mig_src=[4],
        mig_target=[4, 5, 6],
        target_state_id=19,
    ),
    (12, 1): m.Action(
        action=m.ActionType.MERGE,
        gpu_id=-1,
        mig_src=[0, 1, 2, 3],
        mig_target=[0],
        target_state_id=1,
    ),
    (12, 3): m.Action(
        action=m.ActionType.MERGE,
        gpu_id=-1,
        mig_src=[0, 1],
        mig_target=[0],
        target_state_id=3,
    ),
    (12, 8): m.Action(
        action=m.ActionType.MERGE,
        gpu_id=-1,
        mig_src=[2, 3],
        mig_target=[2],
        target_state_id=8,
    ),
    (12, 13): m.Action(
        action=m.ActionType.SPLIT,
        gpu_id=-1,
        mig_src=[1],
        mig_target=[1, 2],
        target_state_id=13,
    ),
    (12, 14): m.Action(
        action=m.ActionType.SPLIT,
        gpu_id=-1,
        mig_src=[0],
        mig_target=[0, 1],
        target_state_id=14,
    ),
    (13, 1): m.Action(
        action=m.ActionType.MERGE,
        gpu_id=-1,
        mig_src=[0, 1, 2, 3, 4],
        mig_target=[0],
        target_state_id=1,
    ),
    (13, 3): m.Action(
        action=m.ActionType.MERGE,
        gpu_id=-1,
        mig_src=[0, 1, 2],
        mig_target=[0],
        target_state_id=3,
    ),
    (13, 9): m.Action(
        action=m.ActionType.MERGE,
        gpu_id=-1,
        mig_src=[3, 4],
        mig_target=[3],
        target_state_id=9,
    ),
    (13, 12): m.Action(
        action=m.ActionType.MERGE,
        gpu_id=-1,
        mig_src=[1, 2],
        mig_target=[1],
        target_state_id=12,
    ),
    (13, 15): m.Action(
        action=m.ActionType.SPLIT,
        gpu_id=-1,
        mig_src=[3],
        mig_target=[3, 4],
        target_state_id=15,
    ),
    (13, 17): m.Action(
        action=m.ActionType.SPLIT,
        gpu_id=-1,
        mig_src=[0],
        mig_target=[0, 1],
        target_state_id=17,
    ),
    (14, 1): m.Action(
        action=m.ActionType.MERGE,
        gpu_id=-1,
        mig_src=[0, 1, 2, 3, 4],
        mig_target=[0],
        target_state_id=1,
    ),
    (14, 3): m.Action(
        action=m.ActionType.MERGE,
        gpu_id=-1,
        mig_src=[0, 1, 2],
        mig_target=[0],
        target_state_id=3,
    ),
    (14, 10): m.Action(
        action=m.ActionType.MERGE,
        gpu_id=-1,
        mig_src=[3, 4],
        mig_target=[3],
        target_state_id=10,
    ),
    (14, 12): m.Action(
        action=m.ActionType.MERGE,
        gpu_id=-1,
        mig_src=[0, 1],
        mig_target=[0],
        target_state_id=12,
    ),
    (14, 16): m.Action(
        action=m.ActionType.SPLIT,
        gpu_id=-1,
        mig_src=[3],
        mig_target=[3, 4],
        target_state_id=16,
    ),
    (14, 17): m.Action(
        action=m.ActionType.SPLIT,
        gpu_id=-1,
        mig_src=[2],
        mig_target=[2, 3],
        target_state_id=17,
    ),
    (15, 1): m.Action(
        action=m.ActionType.MERGE,
        gpu_id=-1,
        mig_src=[0, 1, 2, 3, 4, 5],
        mig_target=[0],
        target_state_id=1,
    ),
    (15, 4): m.Action(
        action=m.ActionType.MERGE,
        gpu_id=-1,
        mig_src=[0, 1, 2],
        mig_target=[0],
        target_state_id=4,
    ),
    (15, 9): m.Action(
        action=m.ActionType.MERGE,
        gpu_id=-1,
        mig_src=[3, 4, 5],
        mig_target=[3],
        target_state_id=9,
    ),
    (15, 13): m.Action(
        action=m.ActionType.MERGE,
        gpu_id=-1,
        mig_src=[3, 4],
        mig_target=[3],
        target_state_id=13,
    ),
    (15, 19): m.Action(
        action=m.ActionType.SPLIT,
        gpu_id=-1,
        mig_src=[0],
        mig_target=[0, 1],
        target_state_id=19,
    ),
    (16, 1): m.Action(
        action=m.ActionType.MERGE,
        gpu_id=-1,
        mig_src=[0, 1, 2, 3, 4, 5],
        mig_target=[0],
        target_state_id=1,
    ),
    (16, 4): m.Action(
        action=m.ActionType.MERGE,
        gpu_id=-1,
        mig_src=[0, 1, 2],
        mig_target=[0],
        target_state_id=4,
    ),
    (16, 10): m.Action(
        action=m.ActionType.MERGE,
        gpu_id=-1,
        mig_src=[3, 4, 5],
        mig_target=[3],
        target_state_id=10,
    ),
    (16, 14): m.Action(
        action=m.ActionType.MERGE,
        gpu_id=-1,
        mig_src=[3, 4],
        mig_target=[3],
        target_state_id=14,
    ),
    (16, 19): m.Action(
        action=m.ActionType.SPLIT,
        gpu_id=-1,
        mig_src=[2],
        mig_target=[2, 3],
        target_state_id=19,
    ),
    (17, 1): m.Action(
        action=m.ActionType.MERGE,
        gpu_id=-1,
        mig_src=[0, 1, 2, 3, 4, 5],
        mig_target=[0],
        target_state_id=1,
    ),
    (17, 3): m.Action(
        action=m.ActionType.MERGE,
        gpu_id=-1,
        mig_src=[0, 1, 2, 3],
        mig_target=[0],
        target_state_id=3,
    ),
    (17, 11): m.Action(
        action=m.ActionType.MERGE,
        gpu_id=-1,
        mig_src=[4, 5],
        mig_target=[4],
        target_state_id=11,
    ),
    (17, 13): m.Action(
        action=m.ActionType.MERGE,
        gpu_id=-1,
        mig_src=[0, 1],
        mig_target=[0],
        target_state_id=13,
    ),
    (17, 14): m.Action(
        action=m.ActionType.MERGE,
        gpu_id=-1,
        mig_src=[2, 3],
        mig_target=[2],
        target_state_id=14,
    ),
    (17, 19): m.Action(
        action=m.ActionType.SPLIT,
        gpu_id=-1,
        mig_src=[4],
        mig_target=[4, 5],
        target_state_id=19,
    ),
    (19, 1): m.Action(
        action=m.ActionType.MERGE,
        gpu_id=-1,
        mig_src=[0, 1, 2, 3, 4, 5, 6],
        mig_target=[0],
        target_state_id=1,
    ),
    (19, 4): m.Action(
        action=m.ActionType.MERGE,
        gpu_id=-1,
        mig_src=[0, 1, 2, 3],
        mig_target=[0],
        target_state_id=4,
    ),
    (19, 11): m.Action(
        action=m.ActionType.MERGE,
        gpu_id=-1,
        mig_src=[4, 5, 6],
        mig_target=[4],
        target_state_id=11,
    ),
    (19, 15): m.Action(
        action=m.ActionType.MERGE,
        gpu_id=-1,
        mig_src=[0, 1],
        mig_target=[0],
        target_state_id=15,
    ),
    (19, 16): m.Action(
        action=m.ActionType.MERGE,
        gpu_id=-1,
        mig_src=[2, 3],
        mig_target=[2],
        target_state_id=16,
    ),
    (19, 17): m.Action(
        action=m.ActionType.MERGE,
        gpu_id=-1,
        mig_src=[4, 5],
        mig_target=[4],
        target_state_id=17,
    ),
}


def map_res_action_to_action(
    res_action: m.ResourceManagerAction,
    current_sid: int,
    find_best_index_fn: Callable[[int, m.MIGProfile], int],
    get_owner_fn: Callable[[int, int], m.AgentId],
) -> Optional[m.Action]:
    """Centralized logic to translate ResourceManagerAction to a concrete Action.

    This function handles both pure transfers and state transitions (with optional
    transfers) by abstracting the lookup of engine/slot indices and owners via
    provided callbacks.
    """
    if res_action == m.ResourceManagerAction.NO_ACTION:
        return None

    val = res_action.value
    gpu_id, target_sid, trans_mig = (
        val.gpu_id,
        val.target_state_id,
        val.transfer_mig,
    )

    if target_sid is None:
        # Pure Transfer
        assert trans_mig is not None, "Transfer MIG must be specified for pure transfer"
        mig_idx = find_best_index_fn(gpu_id, trans_mig)
        if mig_idx == -1:
            return None

        current_owner = get_owner_fn(gpu_id, mig_idx)
        if val.receiver_id is not None:
            receiver_id = val.receiver_id
            if receiver_id == current_owner:
                return None
        else:
            receiver_id = (
                m.AgentId.RAG if current_owner == m.AgentId.CODING else m.AgentId.CODING
            )

        return m.Action(
            action=m.ActionType.TRANSFER,
            gpu_id=gpu_id,
            mig_src=[mig_idx],
            mig_target=[mig_idx],
            target_state_id=current_sid,
            receiver=m.Receiver(receiver_id=receiver_id, mig_idx=mig_idx),
        )

    # State Transition (potentially with transfer)
    transition_action = TRANSITION_MATRIX.get((current_sid, target_sid))
    if not transition_action:
        return None

    receiver = None
    if trans_mig:
        # Find which resulting index matches the transfer profile
        found_idx = -1
        for idx in transition_action.mig_target:
            if STATE_DEFINITIONS[target_sid][idx] == trans_mig:
                found_idx = idx
                break

        if found_idx != -1:
            src_idx = transition_action.mig_src[0]
            current_owner = get_owner_fn(gpu_id, src_idx)
            if val.receiver_id is not None:
                receiver_id = val.receiver_id
                if receiver_id == current_owner:
                    return None
            else:
                receiver_id = (
                    m.AgentId.RAG
                    if current_owner == m.AgentId.CODING
                    else m.AgentId.CODING
                )
            receiver = m.Receiver(receiver_id=receiver_id, mig_idx=found_idx)

    return dataclasses.replace(transition_action, gpu_id=gpu_id, receiver=receiver)
