from typing import Dict, Any, List, Tuple, Optional
import os
import yaml
import shutil
import argparse
from datetime import datetime
from pathlib import Path

import torch
import numpy as np
import numpy.typing as npt
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    CallbackList,
    BaseCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import src.share.models as m
from src.simulation.agent import AgentImpl
from src.share.env import BaseMIGResourceEnv
from src.simulation.simulator import SimulatorImpl
from src.share.request_loader import RequestLoader
from src.training.logger import TrainingLogger
from src.training.config import TRAINING_CONFIG
from src.training.models import AgentPattern
import src.simulation.utils as sim_utils
from src.training.callbacks import (
    SaveVecNormalizeCallback,
    EntCoefSchedulerCallback,
    LogCleanupCallback,
)


# PyTorch distributions validate_args can cause Simplex constraint failures
# due to float32 precision errors during MaskableCategorical softmax normalization
torch.distributions.Distribution.set_default_validate_args(False)


def setup_training_environment(ckpt: Optional[Path] = None) -> str:
    run_id = os.environ.get(
        "TRAINING_RUN_ID", datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3]
    )
    os.environ["TRAINING_RUN_ID"] = run_id

    # Setup results directory
    results_dir = Path(f"results/{run_id}")
    snapshots_dir = results_dir / "snapshots"
    snapshots_dir.mkdir(parents=True, exist_ok=True)

    # Handle snapshotting config
    config_path = Path("configs/training_config.yaml")
    if ckpt is not None:
        old_snapshot = ckpt.parent.parent.parent / "snapshots" / "training_config.yaml"
        if old_snapshot.exists():
            print(f"Using old snapshot config from {old_snapshot}")
            config_path = old_snapshot
        else:
            print(
                f"Warning: Old snapshot not found at {old_snapshot}, using current config"
            )

    snapshot_path = snapshots_dir / "training_config.yaml"
    shutil.copy2(config_path, snapshot_path)

    # Update simulation_config.yaml cluster
    with open(snapshot_path, "r") as f:
        training_data = yaml.safe_load(f)

    with open("configs/simulation_config.yaml", "r") as f:
        sim_data = yaml.safe_load(f)

    sim_data["simulation"]["cluster"] = training_data["training"]["cluster"]

    with open("configs/simulation_config.yaml", "w") as f:
        yaml.dump(sim_data, f, default_flow_style=False)

    return run_id


class TrainingMIGResourceEnv(BaseMIGResourceEnv):
    """
    Concrete Training Environment for MIG Resource Management.
    Handles termination based on max_steps and request replenishment.
    """

    def __init__(self, simulator: m.Simulator, enable_log: bool = True) -> None:
        super().__init__(simulator)
        self.episode_count: int = 0
        self.max_steps: int = TRAINING_CONFIG.episode_length
        run_id = os.environ["TRAINING_RUN_ID"]
        self._logger = TrainingLogger(
            log_dir=f"results/{run_id}/logs/train", enabled=enable_log
        )
        self.request_loader = RequestLoader(
            num_steps=self.max_steps,
            get_rate_range=lambda p, a: TRAINING_CONFIG.pattern_rate(
                AgentPattern(p), a
            ),
            get_duration_range=lambda p: TRAINING_CONFIG.pattern_duration(
                AgentPattern(p)
            ),
            dataset_paths=sim_utils.SIM_CONFIG.datasets,
        )

    @property
    def logger(self) -> TrainingLogger:
        return self._logger

    def step(
        self, action: int
    ) -> Tuple[npt.NDArray[np.float32], float, bool, bool, Dict[str, Any]]:
        # Use base step for core transition
        obs, reward, _, _, _ = super().step(action)

        # Logging
        state_data = self.sim.get_state()
        enum_action = list(m.ResourceManagerAction)[action]
        self._logger.log_step(
            self.current_step - 1,  # Base incremented it
            enum_action.name,
            state_data["current_budget"],
            state_data["arrival_rate"],
            self.sim.agents,
        )

        # Training-specific termination
        terminated = self.current_step >= self.max_steps
        truncated = False

        # Request replenishment
        for agent_id in self.sim.need_requests_replenish():
            max_arr_time = self.sim.latest_arrival_time(agent_id)
            self.load_turn += 1
            new_requests = self.request_loader.generate_requests(
                agent_id=agent_id, start_time=max_arr_time, turn=self.load_turn
            )
            self.sim.add_arrival_events(new_requests)

        return obs, reward, terminated, truncated, {}

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[npt.NDArray[np.float32], Dict[str, Any]]:
        # Base reset for counters
        super().reset(seed=seed)
        self.load_turn = 0

        # Training reset logic
        self.sim.reset()
        self._logger.start_episode(self.episode_count)
        self.request_loader = RequestLoader(
            num_steps=self.max_steps,
            get_rate_range=lambda p, a: TRAINING_CONFIG.pattern_rate(
                AgentPattern(p), a
            ),
            get_duration_range=lambda p: TRAINING_CONFIG.pattern_duration(
                AgentPattern(p)
            ),
            dataset_paths=sim_utils.SIM_CONFIG.datasets,
        )

        requests: List[m.Request] = []
        for aid in m.AgentId:
            requests.extend(
                self.request_loader.generate_requests(agent_id=aid, turn=self.load_turn)
            )
        self.sim.init_simulator(requests, self.max_steps)
        self.sim.run()

        state_data = self.sim.get_state()
        return self._get_obs(state_data), {}


def train(ckpt: Optional[Path] = None, run_id: Optional[str] = None) -> None:
    run_id = run_id or os.environ["TRAINING_RUN_ID"]
    run_name = f"{run_id}"

    # Ensure we use the snapshotted config if loading from a checkpoint
    if ckpt is not None:
        snapshot_path = Path(f"results/{run_id}/snapshots/training_config.yaml")
        if snapshot_path.exists():
            print(f"Reloading TRAINING_CONFIG from session snapshot: {snapshot_path}")
            TRAINING_CONFIG.update(snapshot_path)
    agents: Dict[m.AgentId, m.Agent] = {}
    engines: Dict[str, m.LLMEngine] = {}
    for aid in m.AgentId:
        agents[aid] = AgentImpl(aid)

    sim = SimulatorImpl(
        agents=agents,
        engines=engines,
        no_log=True,
    )

    # 1. Initialize the Environment
    raw_env = TrainingMIGResourceEnv(sim)
    monitored_env: Monitor[npt.NDArray[np.float32], int] = Monitor(raw_env)

    if TRAINING_CONFIG.sb3_norm:
        vec_env = DummyVecEnv([lambda: monitored_env])
        if ckpt is not None:
            norm_path = ckpt.with_name(f"{ckpt.stem}_vecnormalize.pkl")
            if norm_path.exists():
                print(f"Loading VecNormalize stats from {norm_path}")
                env = VecNormalize.load(str(norm_path), vec_env)
                env.training = True
                env.norm_reward = True
            else:
                print(
                    f"Warning: {norm_path} not found, initializing fresh VecNormalize"
                )
                env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)
        else:
            env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)
    else:
        env = monitored_env

    policy_kwargs = dict(
        activation_fn=torch.nn.Tanh,  # Tanh is often more stable for PPO
        net_arch=dict(
            # Actor: Policy network layers
            pi=TRAINING_CONFIG.rl_net_arch_pi,
            # Critic: Value network layers (bigger to handle the latency magnitude)
            vf=TRAINING_CONFIG.rl_net_arch_vf,
        ),
    )

    # 2. Define MaskablePPO Hyperparameters
    # Linear LR schedule: decays from lr_max to lr_min over training
    lr_max = TRAINING_CONFIG.rl_lr_max
    lr_min = TRAINING_CONFIG.rl_lr_min

    def lr_schedule(progress_remaining: float) -> float:
        """Linearly decays from lr_max (start) to lr_min (end)."""
        if ckpt is not None:
            return lr_min
        return lr_min + (lr_max - lr_min) * progress_remaining

    model: MaskablePPO
    if ckpt is not None:
        print(f"Loading model from checkpoint: {ckpt}")
        custom_objects: Dict[str, Any] = {
            "learning_rate": lr_schedule,
            "tensorboard_log": f"results/{run_id}/tboards/{run_name}",
            "ent_coef": TRAINING_CONFIG.rl_min_ent_coef,
            "n_steps": TRAINING_CONFIG.rl_n_steps,
            "batch_size": TRAINING_CONFIG.rl_batch_size,
            "n_epochs": TRAINING_CONFIG.rl_n_epochs,
            "gamma": TRAINING_CONFIG.rl_gamma,
            "gae_lambda": TRAINING_CONFIG.rl_gae_lambda,
            "clip_range": TRAINING_CONFIG.rl_clip_range,
            "clip_range_vf": TRAINING_CONFIG.rl_clip_range,
        }
        model = MaskablePPO.load(  # type: ignore
            ckpt,
            env=env,
            device="cuda",
            custom_objects=custom_objects,
        )
        # Ensure tensorboard logger is properly setup for the new run
        model.tensorboard_log = f"results/{run_id}/tboards/{run_name}"
    else:
        model = MaskablePPO(
            "MlpPolicy",
            env,
            policy_kwargs=policy_kwargs,
            verbose=1,
            learning_rate=lr_schedule,
            n_steps=TRAINING_CONFIG.rl_n_steps,
            batch_size=TRAINING_CONFIG.rl_batch_size,
            n_epochs=TRAINING_CONFIG.rl_n_epochs,
            gamma=TRAINING_CONFIG.rl_gamma,
            gae_lambda=TRAINING_CONFIG.rl_gae_lambda,
            clip_range=TRAINING_CONFIG.rl_clip_range,
            clip_range_vf=TRAINING_CONFIG.rl_clip_range,
            ent_coef=TRAINING_CONFIG.rl_ent_coef
            if not TRAINING_CONFIG.rl_enable_ent_coef_schd
            else TRAINING_CONFIG.rl_max_ent_coef,
            device="cuda",
            tensorboard_log=f"results/{run_id}/tboards/{run_name}",
        )

    # 3. Setup Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=5120, save_path=f"results/{run_id}/ckpts/{run_name}"
    )
    log_cleanup_callback = LogCleanupCallback(raw_env.logger)
    cb_list: List[BaseCallback] = [checkpoint_callback, log_cleanup_callback]
    if TRAINING_CONFIG.sb3_norm:
        cb_list.append(
            SaveVecNormalizeCallback(
                save_freq=5120, save_path=f"results/{run_id}/ckpts/{run_name}"
            )
        )
    if TRAINING_CONFIG.rl_enable_ent_coef_schd and ckpt is None:
        cb_list.append(
            EntCoefSchedulerCallback(
                initial_ent_coef=TRAINING_CONFIG.rl_max_ent_coef,
                final_ent_coef=TRAINING_CONFIG.rl_min_ent_coef,
            )
        )
    callbacks = CallbackList(cb_list)

    # 4. Start Training
    print("Training Setup Complete")
    model.learn(  # type: ignore
        total_timesteps=TRAINING_CONFIG.total_timesteps,
        callback=callbacks,
        progress_bar=True,
    )

    # 5. Save the final model
    model.save(f"results/{run_id}/ckpts/{run_name}/ppo_mig_resource_manager")
    if TRAINING_CONFIG.sb3_norm:
        vec_env = model.get_vec_normalize_env()
        if vec_env is not None:
            vec_env.save(
                f"results/{run_id}/ckpts/{run_name}/ppo_mig_resource_manager_vecnormalize.pkl"
            )
    print("Training Complete. Model Saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MIG Resource Manager")
    parser.add_argument(
        "--ckpt",
        type=Path,
        default=None,
        help="Path to checkpoint to continue training",
    )
    args = parser.parse_args()

    run_id = setup_training_environment(ckpt=args.ckpt)
    train(ckpt=args.ckpt, run_id=run_id)
