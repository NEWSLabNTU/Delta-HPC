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
import torch
from stable_baselines3.common.monitor import Monitor
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

        """
        Action space:
        0. No action
        1. Give 10gb VRAM from A1 to A2
        3. Give 10gb VRAM from A2 to A1
        3. Give 20gb VRAM from A1 to A2
        4. Give 20gb VRAM from A2 to A1
        5. Split A1’s larger MIG
        6. Split A2’s larger MIG
        7. Merge A1’s 2 smaller MIG
        8. Merge A2’s 2 smaller MIG
        """
        self.sim = simulator
        self.action_space = spaces.Discrete(25)

        # State Space: Flattened dictionary metrics
        history_len = TRAINING_CONFIG.arrival_rate_history_length
        # Agents: 2 agents
        # Per Agent: 9 scalar metrics + history_len + 5 KV util + 5 avg latency + 1 mig instance + 5 mig geometry = 24 + history_len
        per_agent_features = 27 + history_len
        # Global Flags/Budget/Downtime/Splits/Merges: 5
        total_features = 2 * per_agent_features + 3

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
            # Action 0 is NO_ACTION
            mask[1] = False
            mask[2] = False
            mask[3] = False
            mask[4] = False

        self._current_action_mask = mask
        return np.array(self._current_action_mask, dtype=np.bool_)

    def _get_obs(self, state_data: m.EnvironmentStateData) -> npt.NDArray[np.float32]:
        obs_list: List[float] = []
        agents_ordered = sorted(self.sim.agents.keys(), key=lambda a: a.value)

        metrics = [
            "arrival_rate",
            "arrival_rate_trend",
            "mig_total_ratio",
            "avg_queue_length",
            "avg_queue_length_trend",
            "avg_running_requests",
            "queue_delta",
            "p99_ttft",
            "avg_tpot",
        ]

        for aid in agents_ordered:
            # 9 Scalar Metrics
            for metric in metrics:
                data = cast(Dict[m.AgentId, float], state_data[metric]) # type: ignore
                obs_list.append(float(data[aid]))

            # history_len size array
            history = cast(Dict[m.AgentId, Tuple[float, ...]], state_data["arrival_rate_history"])
            obs_list.extend(history[aid])

            # 5 KV Cache Utilization
            kv_util = cast(Dict[m.AgentId, Tuple[float, float, float, float, float]], state_data["kv_cache_utilization"])
            obs_list.extend(kv_util[aid])

            # 5 Avg Composite Latency
            latency = cast(Dict[m.AgentId, Tuple[float, float, float, float, float]], state_data["avg_composite_latency"])
            obs_list.extend(latency[aid])

            # 1 MIG Instance count
            n_mig = state_data["n_mig_instance"][aid]
            obs_list.append(float(n_mig))

            # 5 MIG Geometry
            geometry = state_data["mig_geometry"][aid]
            obs_list.extend([float(x) for x in geometry])

            # 2 action counters
            obs_list.append(float(state_data["last_split"][aid]))
            obs_list.append(float(state_data["last_merge"][aid]))

        # Global Metrics: 3
        obs_list.append(1.0 if state_data["recovery_flag"] else 0.0)
        obs_list.append(float(state_data["current_budget"]))
        obs_list.append(float(state_data["downtime_ratio"]))

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
            self.sim.current_time, self.sim.agents, self.sim.engines
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
            self.sim.current_time, self.sim.agents, self.sim.engines
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
    # Track episode rewards and lengths
    env: Monitor[npt.NDArray[np.float32], int] = Monitor(raw_env)

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
    model: MaskablePPO = MaskablePPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        learning_rate=TRAINING_CONFIG.rl_learning_rate,
        n_steps=TRAINING_CONFIG.episode_length,
        batch_size=TRAINING_CONFIG.rl_batch_size,
        n_epochs=TRAINING_CONFIG.rl_n_epochs,
        gamma=TRAINING_CONFIG.rl_gamma,
        gae_lambda=TRAINING_CONFIG.rl_gae_lambda,
        clip_range=TRAINING_CONFIG.rl_clip_range,
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
