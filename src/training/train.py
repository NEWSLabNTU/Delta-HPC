import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from typing import Dict, Any, Tuple, Optional

from src.simulation.models import *


class MIGResourceEnv(gym.Env[np.ndarray, int]):
    """
    Custom Environment for RL-based MIG Resource Management.
    Follows a 90-second fixed-interval Discrete-Time MDP.
    """

    def __init__(self, simulator: Simulator) -> None:
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
        self.action_space: spaces.Discrete = spaces.Discrete(9)

        # State Space: Flattened dictionary metrics
        self.observation_space: spaces.Box = spaces.Box(
            low=-np.inf, high=np.inf, shape=(35,), dtype=np.float32
        )

        # Internal state tracking
        self.current_step: int = 0
        self.max_steps: int = 1000

    def _get_obs(self) -> np.ndarray:
        state_data = self.sim.environment_state.get_state(
            self.sim.current_time, self.sim.agents, self.sim.engines
        )

        obs_list = []
        agents_ordered = sorted(self.sim.agents.keys())

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
            data = state_data[metric]
            for aid in agents_ordered:
                obs_list.append(float(data.get(aid, 0.0)))

        # 8. kv_cache_utilization, 10 items
        kv_util = state_data["kv_cache_utilization"]
        for gpu_idx in [0, 1, 2]:
            lst = kv_util.get(gpu_idx, [0.0] * 5)
            obs_list.extend(lst)

        # 9. mig_config_encoding, 10 items
        mig_enc = state_data["mig_config_encoding"]
        for gpu_idx in [0, 1, 2]:
            lst = mig_enc.get(gpu_idx, [0] * 5)
            obs_list.extend([float(x) for x in lst])

        # 10. recovery_flag: 1 item
        obs_list.append(1.0 if state_data["recovery_flag"] else 0.0)

        return np.array(obs_list, dtype=np.float32)

    def _calculate_reward(self, action: int, obs: np.ndarray) -> float:
        """
        TODO: Implement the Reward Function logic here.
        """
        reward: float = 0.0
        return reward

    def _is_action_valid(self, action: int) -> bool:
        return self.sim.get_action_mask()[action]

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        # 1. Check for Action Validity (Action Masking)
        if not self._is_action_valid(action):
            # Penalize invalid actions and stay in the same state
            return self._get_obs(), -1.0, False, False, {"invalid_action": True}

        # 2. Simulate interval logic
        # Map integer action index to ResourceManagerAction Enum
        enum_action = list(ResourceManagerAction)[action]
        self.sim.handle_resource_manager_trigger(enum_action)
        self.sim.run()

        # 3. Compute new state and reward
        obs: np.ndarray = self._get_obs()
        reward: float = self._calculate_reward(action, obs)

        self.current_step += 1
        terminated: bool = self.current_step >= self.max_steps
        truncated: bool = False

        return obs, reward, terminated, truncated, {}

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self.current_step = 0
        # Reset hardware to initial MIG profile if in a real environment
        self.sim.reset()
        # TODO: Replanish requests
        return self._get_obs(), {}


def train() -> None:
    # 1. Initialize the Environment
    env: MIGResourceEnv = MIGResourceEnv()

    # 2. Define PPO Hyperparameters
    model: PPO = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
    )

    # 3. Setup Checkpoints
    checkpoint_callback: CheckpointCallback = CheckpointCallback(
        save_freq=5000, save_path="./logs/", name_prefix="mig_rl_model"
    )

    # 4. Start Training
    print("Starting Training...")
    model.learn(total_timesteps=100000, callback=checkpoint_callback)

    # 5. Save the final model
    model.save("ppo_mig_resource_manager")
    print("Training Complete. Model Saved.")


if __name__ == "__main__":
    train()
