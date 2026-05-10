
from enum import Enum
from scratch.test_load import MIGProfileBase, MIGProfile

class MIGProfileA100(MIGProfileBase):
    A = MIGProfile.MIG_7G

MIG_PROFILE = MIGProfileA100
