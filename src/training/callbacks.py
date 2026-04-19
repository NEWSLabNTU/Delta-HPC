from pathlib import Path
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from src.training.logger import TrainingLogger


class SaveVecNormalizeCallback(BaseCallback):
    def __init__(
        self,
        save_freq: int,
        save_path: str,
        name_prefix: str = "rl_model",
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix

    def _init_callback(self) -> None:
        Path(self.save_path).mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            vec_env = self.model.get_vec_normalize_env()
            if vec_env is not None:
                path = (
                    Path(self.save_path)
                    / f"{self.name_prefix}_{self.num_timesteps}_steps_vecnormalize.pkl"
                )
                vec_env.save(str(path))
        return True


class EntCoefSchedulerCallback(BaseCallback):
    def __init__(
        self, initial_ent_coef: float, final_ent_coef: float, verbose: int = 0
    ):
        super().__init__(verbose)
        self.initial_ent_coef = initial_ent_coef
        self.final_ent_coef = final_ent_coef

    def _on_step(self) -> bool:
        # Linear decay based on training progress
        progress = self.num_timesteps / self.model._total_timesteps  # type: ignore
        # new_ent_coef = self.initial_ent_coef + progress * (
        #     self.final_ent_coef - self.initial_ent_coef
        # )
        if progress < 0.20:
            new_ent_coef = self.initial_ent_coef
        else:
            r = (progress - 0.20) / 0.80
            new_ent_coef = (
                (self.initial_ent_coef + self.final_ent_coef) / 2
                if r < 0.5
                else self.final_ent_coef
            )

        assert isinstance(self.model, OnPolicyAlgorithm)
        self.model.ent_coef = new_ent_coef

        self.logger.record("train/ent_coef", new_ent_coef)
        return True


class LogCleanupCallback(BaseCallback):
    def __init__(self, env_logger: TrainingLogger, verbose: int = 0):
        super().__init__(verbose)
        self._env_logger = env_logger

    def _on_step(self) -> bool:
        return True

    def _on_training_end(self) -> None:
        self._env_logger.close()
