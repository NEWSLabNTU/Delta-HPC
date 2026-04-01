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
        self.action_space = spaces.Discrete(9)

        # State Space: Flattened dictionary metrics
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(46,), dtype=np.float32
        )

        # Internal state tracking
        self.current_step: int = 0
        self.max_steps: int = 1024
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

        # 1-7. Agent Metrics, 14 items
        metrics = [
            "arrival_rate",
            "arrival_trend",
            "avg_queue_length",
            "avg_running_requests",
            "queue_delta",
            "p99_ttft",
            "avg_tpot",
        ]
        for metric in metrics:
            data = cast(Dict[m.AgentId, float], state_data[metric])
            for aid in agents_ordered:
                obs_list.append(float(data[aid]))

        # 8. kv_cache_utilization, 15 items
        kv_util = state_data["kv_cache_utilization"]
        for gpu_idx in [0, 1, 2]:
            lst = kv_util.get(gpu_idx, [0.0] * 5)
            obs_list.extend(lst)

        # 9. current_mig_profile
        mig_enc = state_data["current_mig_profile"]
        for gpu_idx in [0, 1, 2]:
            lst = mig_enc[gpu_idx]
            # handle if elements are MIGEncoding or simple ints
            obs_list.extend(
                [
                    float(lst.p1),
                    float(lst.p2),
                    float(lst.p3),
                    float(lst.p4),
                    float(lst.p7),
                ]
            )

        # 10. recovery_flag: 1 item
        obs_list.append(1.0 if state_data["recovery_flag"] else 0.0)

        # 11. current_budget: 1 item
        obs_list.append(float(state_data["current_budget"]))

        return np.array(obs_list, dtype=np.float32)

    def _calculate_reward(
        self, action: int, reqs: Dict[m.AgentId, List[m.Request]]
    ) -> float:
        enum_action = list(m.ResourceManagerAction)[action]
        return compute_reward(reqs, enum_action)

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
            self.request_loader.current_rates,
            self.request_loader.current_pattern,
            self.sim.agents,
        )

        # 3. Compute new state and reward
        obs = self._get_obs(state_data)
        reward = self._calculate_reward(action, state_data["requests"])

        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        truncated = False

        # 4. Replenish requests if necessary
        remain = self.sim.pending_arrival_count
        if remain < 1000:
            max_arr_time = self.sim.latest_arrival_time
            self.load_turn += 1
            new_requests = self.request_loader.generate_requests(
                start_time=max_arr_time, turn=self.load_turn
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
        requests = self.request_loader.generate_requests(turn=self.load_turn)
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

    # 2. Define MaskablePPO Hyperparameters
    model: MaskablePPO = MaskablePPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        n_epochs=5,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
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
