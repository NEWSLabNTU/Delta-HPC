import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from typing import Dict, Any, Tuple, Optional
from src.simulation.models import ResourceManagerAction


class MIGResourceEnv(gym.Env[np.ndarray, int]):
    """
    Custom Environment for RL-based MIG Resource Management.
    Follows a 90-second fixed-interval Discrete-Time MDP.
    """

    def __init__(self) -> None:
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
        self.action_space: spaces.Discrete = spaces.Discrete(9)

        # State Space: Placeholder for the 120-second sliding window metrics
        self.observation_space: spaces.Box = spaces.Box(
            low=0, high=1, shape=(20,), dtype=np.float32
        )

        # Internal state tracking
        self.current_step: int = 0
        self.max_steps: int = 1000

    def _get_obs(self) -> np.ndarray:
        """
        TODO: Implement the 120-second sliding window calculation logic here.
        """
        # Dummy observation: Ensuring it matches the observation_space shape and dtype
        obs: np.ndarray = np.random.rand(20).astype(np.float32)
        return obs

    def _calculate_reward(self, action: int, obs: np.ndarray) -> float:
        """
        TODO: Implement the Reward Function logic here.
        """
        reward: float = 0.0
        return reward

    def _is_action_valid(self, action: int) -> bool:
        """
        TODO: Implement Action Masking logic based on current MIG physical layout.
        """
        return True

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        # 1. Check for Action Validity (Action Masking)
        if not self._is_action_valid(action):
            # Penalize invalid actions and stay in the same state
            return self._get_obs(), -1.0, False, False, {"invalid_action": True}

        # 2. Simulate interval logic
        # Map integer action index to ResourceManagerAction Enum
        enum_action = list(ResourceManagerAction)[action]
        self.sim.handle_resource_manager_trigger(enum_action)
        self.sim.run_until()

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
