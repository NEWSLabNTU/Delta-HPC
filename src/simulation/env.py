from typing import Dict, Any, List, Tuple, Optional, cast, TypeVar
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import numpy.typing as npt

import src.simulation.models as m
from src.training.config import TRAINING_CONFIG
from src.training.models import TrainingPhase
from src.training.rewards import compute_reward


ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


class BaseMIGResourceEnv(gym.Env[npt.NDArray[np.float32], int]):
    """
    Common base class for MIG Resource Management environments.
    Handles observation mapping and reward logic.
    """

    def __init__(self, simulator: m.Simulator) -> None:
        super().__init__()
        self.sim = simulator
        self.action_space = spaces.Discrete(len(m.ResourceManagerAction))

        # State Space: Flattened dictionary metrics
        history_len = TRAINING_CONFIG.arrival_rate_history_length
        # Agents: 2 agents
        # Per Agent: 61 + history_len features (Removed n_mig)
        per_agent_features = 61 + history_len
        # Global Metrics: 15 + 30 (one-hot) + 14 (grid) = 59 features
        total_features = 2 * per_agent_features + 59

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(total_features,),
            dtype=np.float32,
        )

        # Basic counters
        self.current_step: int = 0
        self.load_turn: int = 0
        self.episode_count: int = 0
        self._current_action_mask: npt.NDArray[np.bool_] = np.zeros(
            (len(m.ResourceManagerAction),), dtype=np.bool_
        )

    def get_phase_action_mask(self, phase: TrainingPhase) -> npt.NDArray[np.bool_]:
        mask = self.sim.get_action_mask()

        if phase == TrainingPhase.PHASE_1:
            # Phase 1: Only allow pure transfers or self-transitions
            for act_id, action in enumerate(m.ResourceManagerAction):
                if action == m.ResourceManagerAction.NO_ACTION:
                    continue
                val = action.value
                # If it involves a state change, disable it in Phase 1
                if val.target_state_id is not None:
                    if val.target_state_id != self.sim.gpu_current_state[val.gpu_id]:
                        mask[act_id] = False

        return np.array(mask, dtype=np.bool_)

    def action_masks(self) -> npt.NDArray[np.bool_]:
        # Subclasses should override or call get_phase_action_mask with specific phase
        # Default behavior: use active phase from TRAINING_CONFIG
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

            # 7 KV Cache Utilization (1L, 1S, 2g, 3g, 4g, 7g, permanent)
            obs_list.extend(state_data["kv_cache_utilization"][aid])

            # 7 Avg Composite Latency (1L, 1S, 2g, 3g, 4g, 7g, permanent) — as percentages
            obs_list.extend(state_data["avg_composite_latency"][aid])

            # 7 Avg Queue Length (1L, 1S, 2g, 3g, 4g, 7g, permanent)
            obs_list.extend(state_data["avg_queue_length"][aid])

            # 7 Avg Queue Length Trend (1L, 1S, 2g, 3g, 4g, 7g, permanent)
            obs_list.extend(state_data["avg_queue_length_trend"][aid])

            # 7 Avg Running Requests (1L, 1S, 2g, 3g, 4g, 7g, permanent)
            obs_list.extend(state_data["avg_running_requests"][aid])

            # 6 Agent-Owns-MIG: per-profile count (1L, 1S, 2g, 3g, 4g, 7g)
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

        # New: 30 features for mig_profile_id_onehot (15 per GPU)
        onehot = state_data["mig_profile_id_onehot"]
        for gpu_idx in (0, 1):
            obs_list.extend(onehot.get(gpu_idx, [0.0] * 15))

        # New: 14 features for ownership_grid (7 per GPU)
        grid = state_data["ownership_grid"]
        for gpu_idx in (0, 1):
            obs_list.extend([float(x) for x in grid.get(gpu_idx, [0] * 7)])

        return np.array(obs_list, dtype=np.float32)

    def _calculate_reward(
        self,
        action: int,
        reqs: Dict[m.AgentId, List[m.Request]],
        current_time: float,
    ) -> float:
        enum_action = list(m.ResourceManagerAction)[action]
        return compute_reward(reqs, enum_action, current_time, self.sim.gpu_engines)

    def _is_action_valid(self, action: int) -> bool:
        return self._current_action_mask[action]

    def step(
        self, action: int
    ) -> Tuple[npt.NDArray[np.float32], float, bool, bool, Dict[str, Any]]:
        enum_action = list(m.ResourceManagerAction)[action]
        if not self._is_action_valid(action):
            raise ValueError(f"Invalid action: {action}")

        # Core execution
        sim_action = self.sim.map_to_action(enum_action)
        self.sim.handle_resource_manager_trigger(sim_action)
        self.sim.run()

        # Observation and Reward
        state_data = self.sim.get_state()
        obs = self._get_obs(state_data)
        reward = self._calculate_reward(
            action, state_data["requests"], self.sim.current_time
        )

        self.current_step += 1

        # Subclasses will handle termination and replenishment
        return obs, reward, False, False, {}

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[npt.NDArray[np.float32], Dict[str, Any]]:
        # gymnasium setup
        super().reset(seed=seed)
        self.current_step = 0
        self.episode_count += 1

        # Initial zero observation - subclasses should override to provide real obs
        assert self.observation_space.shape is not None
        return np.zeros(self.observation_space.shape, dtype=np.float32), {}
