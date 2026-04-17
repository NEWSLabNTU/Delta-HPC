from typing import Dict, Any, List, Tuple, Optional, cast
import os
import argparse
from datetime import datetime
from pathlib import Path

import torch
import gymnasium as gym
from gymnasium import spaces
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
from src.training.models import TrainingPhase
from src.training.rewards import compute_reward
from src.training.callbacks import (
    SaveVecNormalizeCallback,
    EntCoefSchedulerCallback,
    LogCleanupCallback,
)


TRAINING_RUN_ID = os.environ.get(
    "TRAINING_RUN_ID", datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3]
)


class MIGResourceEnv(gym.Env[npt.NDArray[np.float32], int]):
    """
    Custom Environment for RL-based MIG Resource Management.
    Follows a fixed-interval Discrete-Time MDP.
    """

    def __init__(self, simulator: m.Simulator, enable_log: bool = True) -> None:
        super(MIGResourceEnv, self).__init__()
        self.sim = simulator
        self.action_space = spaces.Discrete(35)

        # State Space: Flattened dictionary metrics
        history_len = TRAINING_CONFIG.arrival_rate_history_length
        # Agents: 2 agents
        # Per Agent: 4 scalar metrics + history_len + 5*6 (util, latency, q_len, q_trend, run_req) + 1 mig instance + 5 agent_owns_mig + 6 action metrics = 46 + history_len
        per_agent_features = 46 + history_len
        # Global Flags/Budget/Downtime + 8 Agent Ratios + 4 MIG geometry (2 GPUs × 2 agents): 15
        total_features = 2 * per_agent_features + 15

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(total_features,),
            dtype=np.float32,
        )

        # Internal state tracking
        self.current_step: int = 0
        self.max_steps: int = TRAINING_CONFIG.episode_length
        self.load_turn: int = 0
        self.episode_count: int = 0
        self._logger = TrainingLogger(
            log_dir=f"logs/train/{TRAINING_RUN_ID}", enabled=enable_log
        )
        self.request_loader = RequestLoader()
        self._current_action_mask: npt.NDArray[np.bool_] = self.action_masks()
        self.enable_replenish: bool = True

    @property
    def logger(self) -> TrainingLogger:
        return self._logger

    def get_phase_action_mask(self, phase: TrainingPhase) -> npt.NDArray[np.bool_]:
        mask = self.sim.get_action_mask()

        if phase == TrainingPhase.PHASE_1:
            for act_id, action in enumerate(m.ResourceManagerAction):
                if action != m.ResourceManagerAction.NO_ACTION and isinstance(
                    action.value, m.MigAction
                ):
                    mask[act_id] = False
        elif phase == TrainingPhase.PHASE_2:
            # Enable all actions
            pass

        return np.array(mask, dtype=np.bool_)

    def action_masks(self) -> npt.NDArray[np.bool_]:
        self._current_action_mask = self.get_phase_action_mask(TRAINING_CONFIG.phase)
        return self._current_action_mask

    def _get_obs(self, state_data: m.EnvironmentStateData) -> npt.NDArray[np.float32]:
        obs_list: List[float] = []
        agents_ordered = sorted(self.sim.agents.keys(), key=lambda a: a.value)

        metrics = [
            "arrival_rate",
            "predicted_arrival_rate",
            "total_sm_ratio",
            "total_vram_ratio",
        ]

        for aid in agents_ordered:
            # 4 Scalar Metrics
            for metric in metrics:
                data = cast(Dict[m.AgentId, float], state_data[metric])  # type: ignore
                obs_list.append(float(data[aid]))

            # history_len size array
            obs_list.extend(state_data["arrival_rate_history"][aid])

            # 6 KV Cache Utilization (1g, 2g, 3g, 4g, 7g, permanent)
            obs_list.extend(state_data["kv_cache_utilization"][aid])

            # 6 Avg Composite Latency (1g, 2g, 3g, 4g, 7g, permanent) — as percentages
            obs_list.extend(state_data["avg_composite_latency"][aid])

            # 6 Avg Queue Length (1g, 2g, 3g, 4g, 7g, permanent)
            obs_list.extend(state_data["avg_queue_length"][aid])

            # 6 Avg Queue Length Trend (1g, 2g, 3g, 4g, 7g, permanent)
            obs_list.extend(state_data["avg_queue_length_trend"][aid])

            # 6 Avg Running Requests (1g, 2g, 3g, 4g, 7g, permanent)
            obs_list.extend(state_data["avg_running_requests"][aid])

            # 1 MIG Instance count
            n_mig = state_data["n_mig_instance"][aid]
            obs_list.append(float(n_mig))

            # 5 Agent-Owns-MIG: per-profile count (1g, 2g, 3g, 4g, 7g)
            agent_owns_mig = state_data["agent_owns_mig"][aid]
            obs_list.extend([float(x) for x in agent_owns_mig])

            # 6 action metrics
            obs_list.append(float(state_data["last_split"][aid]))
            obs_list.append(float(state_data["last_merge"][aid]))
            obs_list.append(float(state_data["last_give"][aid]))
            obs_list.append(float(state_data["last_receive"][aid]))
            obs_list.append(float(state_data["last_give_amount"][aid]))
            obs_list.append(float(state_data["last_receive_amount"][aid]))

        # Global Metrics: 11 (3 existing + 8 new differences)
        obs_list.append(1.0 if state_data["recovery_flag"] else 0.0)
        obs_list.append(float(state_data["current_budget"]))
        obs_list.append(float(state_data["downtime_ratio"]))

        # Agent Ratios (CODING - RAG)
        obs_list.append(float(state_data.get("agent_arrival_rate_ratio", 0.0)))
        obs_list.append(float(state_data.get("agent_avg_queue_len_ratio", 0.0)))
        obs_list.append(float(state_data.get("agent_avg_running_req_ratio", 0.0)))
        obs_list.append(float(state_data.get("agent_avg_kv_cache_ratio", 0.0)))
        obs_list.append(float(state_data.get("agent_avg_composite_latency_ratio", 0.0)))
        obs_list.append(float(state_data.get("agent_n_mig_ratio", 0.0)))
        obs_list.append(float(state_data.get("agent_vram_ratio", 0.0)))
        obs_list.append(float(state_data.get("agent_sm_ratio", 0.0)))

        # MIG Geometry: 4 values — GPU 0 [coding, rag] and GPU 1 [coding, rag] (pre-normalized)
        mig_geom = state_data["mig_geometry"]
        for gpu_idx in (0, 1):
            for s in mig_geom.get(gpu_idx, [0.0, 0.0]):
                obs_list.append(float(s))

        return np.array(obs_list, dtype=np.float32)

    def _calculate_reward(
        self,
        action: int,
        reqs: Dict[m.AgentId, List[m.Request]],
        current_time: float,
    ) -> float:
        enum_action = list(m.ResourceManagerAction)[action]
        return compute_reward(reqs, enum_action, current_time, agents=self.sim.agents)

    def _is_action_valid(self, action: int) -> bool:
        return self._current_action_mask[action]

    def step(
        self, action: int
    ) -> Tuple[npt.NDArray[np.float32], float, bool, bool, Dict[str, Any]]:
        enum_action = list(m.ResourceManagerAction)[action]
        # 1. Check for Action Validity (Action Masking)
        if not self._is_action_valid(action):
            raise ValueError(f"Invalid action: {action}")

        # 2. Simulate interval logic
        self.sim.handle_resource_manager_trigger(enum_action)
        self.sim.run()
        state_data = self.sim.get_state(self.current_step + 1)
        self._logger.log_step(
            self.current_step,
            enum_action.name,
            state_data["current_budget"],
            state_data["arrival_rate"],
            self.sim.agents,
        )

        # 3. Compute new state and reward
        obs = self._get_obs(state_data)
        reward = self._calculate_reward(
            action, state_data["requests"], self.sim.current_time
        )

        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        truncated = False

        # 4. Replenish requests if necessary
        if self.enable_replenish:
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
        super().reset(seed=seed)
        self.current_step = 0
        self.load_turn = 0
        self.episode_count += 1

        # Reset hardware simulation state
        self.sim.reset()

        # Replenish requests to initialize simulator
        self._logger.start_episode(self.episode_count)
        self.request_loader = RequestLoader()
        requests: List[m.Request] = []
        for aid in m.AgentId:
            requests.extend(
                self.request_loader.generate_requests(agent_id=aid, turn=self.load_turn)
            )
        self.sim.init_simulator(requests, self.max_steps)
        self.sim.run()  # advance to the first action interval

        state_data = self.sim.get_state(self.current_step)
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
    raw_env = MIGResourceEnv(sim)
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
