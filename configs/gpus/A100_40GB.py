from src.simulation.models import MIGProfileBase, MIGProfile, ProfileInfo


class MIGProfileA100(MIGProfileBase):
    MIG_7G_40GB = ProfileInfo(7, 40, MIGProfile.MIG_7G)
    MIG_4G_20GB = ProfileInfo(4, 20, MIGProfile.MIG_4G)
    MIG_3G_20GB = ProfileInfo(3, 20, MIGProfile.MIG_3G)
    MIG_2G_10GB = ProfileInfo(2, 10, MIGProfile.MIG_2G)
    MIG_1G_10GB = ProfileInfo(1, 10, MIGProfile.MIG_1G_LARGE)

    @property
    def gpu_model(self) -> str:
        return "A100_40GB"


MIG_PROFILE = MIGProfileA100
