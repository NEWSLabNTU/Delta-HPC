from typing import Dict, Any, List, Tuple, Optional
import os
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

import src.simulation.models as m
from src.simulation.request_loader import RequestLoader
from src.simulation.agent import AgentImpl
from src.simulation.engine import LLMEngineImpl
from src.simulation.simulator import SimulatorImpl
import src.simulation.utils as utils
from src.training.logger import TrainingLogger
from src.training.config import TRAINING_CONFIG
from src.training.callbacks import (
    SaveVecNormalizeCallback,
    EntCoefSchedulerCallback,
    LogCleanupCallback,
)

from src.simulation.env import BaseMIGResourceEnv


TRAINING_RUN_ID = os.environ.get(
    "TRAINING_RUN_ID", datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3]
)


class TrainingMIGResourceEnv(BaseMIGResourceEnv):
    """
    Concrete Training Environment for MIG Resource Management.
    Handles termination based on max_steps and request replenishment.
    """

    def __init__(self, simulator: m.Simulator, enable_log: bool = True) -> None:
        super().__init__(simulator)
        self.episode_count: int = 0
        self.max_steps: int = TRAINING_CONFIG.episode_length
        self._logger = TrainingLogger(
            log_dir=f"logs/train/{TRAINING_RUN_ID}", enabled=enable_log
        )
        self.request_loader = RequestLoader()

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
        self.request_loader = RequestLoader()

        requests: List[m.Request] = []
        for aid in m.AgentId:
            requests.extend(
                self.request_loader.generate_requests(agent_id=aid, turn=self.load_turn)
            )
        self.sim.init_simulator(requests, self.max_steps)
        self.sim.run()

        state_data = self.sim.get_state()
        return self._get_obs(state_data), {}


def train(ckpt: Optional[Path] = None) -> None:
    phase = TRAINING_CONFIG.phase
    run_name = f"{TRAINING_RUN_ID}_phase_{phase.value}"
    agents: Dict[m.AgentId, m.Agent] = {}
    engines: Dict[str, m.LLMEngine] = {}
    for aid in m.AgentId:
        agents[aid] = AgentImpl(aid)

    for eng_conf in utils.SIM_CONFIG.initial_state:
        mig = m.MIGProfile.from_string(eng_conf["mig"])
        gpu = int(eng_conf["gpu"])
        agent_name = eng_conf["agent"]
        agent = agents[m.AgentId(agent_name)]
        eid = utils.generate_engine_id(gpu, mig.string)

        is_permanent = eng_conf.get("is-permanent", False)
        eng = LLMEngineImpl.create(
            gpu=gpu,
            engine_id=eid,
            owner=agent,
            mig_profile=mig,
            current_time=0.0,
            is_permanent=is_permanent,
        )

        agent.add_engine(eng)
        engines[eid] = eng

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
        return lr_min + (lr_max - lr_min) * progress_remaining

    model: MaskablePPO
    if ckpt is not None:
        print(f"Loading model from checkpoint: {ckpt}")
        custom_objects: Dict[str, Any] = {
            "learning_rate": lr_schedule,
            "tensorboard_log": f"./tboard/actives/{run_name}",
        }
        model = MaskablePPO.load(  # type: ignore
            ckpt,
            env=env,
            device="cuda",
            custom_objects=custom_objects,
        )
        # Ensure tensorboard logger is properly setup for the new run
        model.tensorboard_log = f"./tboard/actives/{run_name}"
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
            tensorboard_log=f"./tboard/actives/{run_name}",
        )

    # 3. Setup Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=5120, save_path=f"./ckpts/{run_name}"
    )
    log_cleanup_callback = LogCleanupCallback(raw_env.logger)
    cb_list: List[BaseCallback] = [checkpoint_callback, log_cleanup_callback]
    if TRAINING_CONFIG.sb3_norm:
        cb_list.append(
            SaveVecNormalizeCallback(save_freq=5120, save_path=f"./ckpts/{run_name}")
        )
    if TRAINING_CONFIG.rl_enable_ent_coef_schd:
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
    model.save(f"./ckpts/{run_name}/ppo_mig_resource_manager")
    if TRAINING_CONFIG.sb3_norm:
        vec_env = model.get_vec_normalize_env()
        if vec_env is not None:
            vec_env.save(
                f"./ckpts/{run_name}/ppo_mig_resource_manager_vecnormalize.pkl"
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

    train(ckpt=args.ckpt)
