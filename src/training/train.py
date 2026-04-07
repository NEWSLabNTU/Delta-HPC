import time
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    CallbackList,
    BaseCallback,
)

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
import torch
import numpy.typing as npt
from typing import Dict, Any, List, Tuple, Optional, cast

import src.simulation.models as m
from src.simulation.request_loader import RequestLoader
from src.training.logger import TrainingLogger
from src.training.config import TRAINING_CONFIG
from src.training.rewards import compute_reward
from src.simulation.agent import AgentImpl
from src.simulation.engine import LLMEngineImpl
from src.simulation.simulator import SimulatorImpl
import src.simulation.utils as utils

TIMESTAMP = time.strftime("%Y%m%d_%H%M%S")


class MIGResourceEnv(gym.Env[npt.NDArray[np.float32], int]):
    """
    Custom Environment for RL-based MIG Resource Management.
    Follows a fixed-interval Discrete-Time MDP.
    """

    def __init__(self, simulator: m.Simulator) -> None:
        super(MIGResourceEnv, self).__init__()
        self.sim = simulator
        self.action_space = spaces.Discrete(31)

        # State Space: Flattened dictionary metrics
        history_len = TRAINING_CONFIG.arrival_rate_history_length
        # Agents: 2 agents
        # Per Agent: 4 scalar metrics + history_len + 5*6 (util, latency, q_len, q_trend, run_req) + 1 mig instance + 5 mig geometry + 6 action metrics = 46 + history_len
        per_agent_features = 46 + history_len
        # Global Flags/Budget/Downtime + 8 Agent Ratios: 11
        total_features = 2 * per_agent_features + 11

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
        self._logger = TrainingLogger()
        self.request_loader = RequestLoader(phase=TRAINING_CONFIG.phase)
        self._current_action_mask: List[bool] = list(self.action_masks())

    @property
    def logger(self) -> TrainingLogger:
        return self._logger

    def action_masks(self, phase: Optional[int] = None) -> npt.NDArray[np.bool_]:
        if phase is None:
            phase = TRAINING_CONFIG.phase

        mask = self.sim.get_action_mask()

        if phase == 1:
            for act_id, action in enumerate(m.ResourceManagerAction):
                if action != m.ResourceManagerAction.NO_ACTION and isinstance(
                    action.value, m.VramTransferAction
                ):
                    mask[act_id] = False

        # Cooldown (Per-Agent): disable merging for X steps after a split,
        # and disable splitting for X steps after a merge.
        env_state = self.sim.environment_state
        cooldown_steps = TRAINING_CONFIG.split_merge_cooldown_steps

        # Transfer cooldown constraint:
        # If A transferred a MIG to B recently, NO transfers can happen between A and B
        transfer_blocked = any(
            env_state.get_steps_since(agent.agent_id, "give") < cooldown_steps
            for agent in self.sim.agents.values()
        )

        for act_id, action in enumerate(m.ResourceManagerAction):
            if action == m.ResourceManagerAction.NO_ACTION:
                continue
            val = action.value

            if isinstance(val, m.VramTransferAction):
                if transfer_blocked:
                    mask[act_id] = False
                continue
            assert isinstance(val, m.MigAction)

            aid = val.victim
            if val.action == "split":
                # Cannot split if this agent recently merged
                if env_state.get_steps_since(aid, "merge") < cooldown_steps:
                    mask[act_id] = False
            elif val.action == "merge":
                # Cannot merge if this agent recently split
                if env_state.get_steps_since(aid, "split") < cooldown_steps:
                    mask[act_id] = False

        self._current_action_mask = mask
        return np.array(self._current_action_mask, dtype=np.bool_)

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

            # 5 MIG Geometry
            geometry = state_data["mig_geometry"][aid]
            obs_list.extend([float(x) for x in geometry])

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

        return np.array(obs_list, dtype=np.float32)

    def _calculate_reward(
        self, action: int, reqs: Dict[m.AgentId, List[m.Request]], current_time: float
    ) -> float:
        enum_action = list(m.ResourceManagerAction)[action]
        return compute_reward(reqs, enum_action, current_time)

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
        state_data = self.sim.environment_state.get_state(
            self.sim.current_time,
            self.sim.agents,
            self.sim.engines,
            self.current_step + 1,
        )
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
        self.request_loader = RequestLoader(phase=TRAINING_CONFIG.phase)
        requests: List[m.Request] = []
        for aid in m.AgentId:
            requests.extend(
                self.request_loader.generate_requests(agent_id=aid, turn=self.load_turn)
            )
        self.sim.init_simulator(requests, self.max_steps)
        self.sim.run()  # advance to the first action interval

        state_data = self.sim.environment_state.get_state(
            self.sim.current_time, self.sim.agents, self.sim.engines, self.current_step
        )
        return self._get_obs(state_data), {}


class LogCleanupCallback(BaseCallback):
    def __init__(self, env_logger: TrainingLogger, verbose: int = 0):
        super().__init__(verbose)
        self._env_logger = env_logger

    def _on_step(self) -> bool:
        return True

    def _on_training_end(self) -> None:
        self._env_logger.close()


def train() -> None:
    # 1. Initialize Simulator similar to main.py
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
    env_0: Monitor[npt.NDArray[np.float32], int] = Monitor(raw_env)
    env_1 = DummyVecEnv([lambda: env_0])
    env = VecNormalize(env_1, norm_obs=True, norm_reward=True)

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

    model: MaskablePPO = MaskablePPO(
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
        ent_coef=TRAINING_CONFIG.rl_ent_coef,
        device="cuda",
        tensorboard_log=f"./tboard/{TIMESTAMP}",
    )

    # 3. Setup Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=5120, save_path=f"./ckpts/{TIMESTAMP}"
    )
    log_cleanup_callback = LogCleanupCallback(raw_env.logger)
    callbacks = CallbackList([checkpoint_callback, log_cleanup_callback])

    # 4. Start Training
    print("Training Setup Complete")
    model.learn(total_timesteps=204800, callback=callbacks, progress_bar=True)  # type: ignore

    # 5. Save the final model
    model.save(f"./ckpts/{TIMESTAMP}/ppo_mig_resource_manager")
    print("Training Complete. Model Saved.")


if __name__ == "__main__":
    train()
