from src.simulation.models import MIGProfileBase, MIGProfile, ProfileInfo


class MIGProfileB200(MIGProfileBase):
    MIG_7G_160GB = ProfileInfo(7, 160, MIGProfile.MIG_7G)
    MIG_4G_80GB = ProfileInfo(4, 80, MIGProfile.MIG_4G)
    MIG_3G_80GB = ProfileInfo(3, 80, MIGProfile.MIG_3G)
    MIG_2G_45GB = ProfileInfo(2, 45, MIGProfile.MIG_2G)
    MIG_1G_45GB = ProfileInfo(1, 45, MIGProfile.MIG_1G_LARGE)
    MIG_1G_23GB = ProfileInfo(1, 23, MIGProfile.MIG_1G_SMALL)

    @property
    def gpu_model(self) -> str:
        return "B200"


MIG_PROFILE = MIGProfileB200
