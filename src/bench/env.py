import numpy as np
import gymnasium as gym
import numpy.typing as npt
from typing import Any, Dict, List, Optional, Tuple

from src.bench.models import BenchMode
from src.bench.config import BENCH_CONFIG
from src.training.train import MIGResourceEnv
import src.simulation.models as m


class BenchMIGResourceEnv(MIGResourceEnv):
    def __init__(
        self,
        simulator: m.Simulator,
        bench_mode: BenchMode,
        requests: List[m.Request],
        init_mode: m.InitialMIGCombination,
    ):
        super().__init__(simulator, enable_log=False)
        self.bench_mode = bench_mode
        self.enable_replenish = False
        self._requests = requests
        self._init_mode = init_mode

        # Overwrite episode length
        self.max_steps = BENCH_CONFIG.benchmark_length

    def action_masks(self) -> npt.NDArray[np.bool_]:
        self._current_action_mask = self.get_phase_action_mask(BENCH_CONFIG.phase)
        return self._current_action_mask

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[npt.NDArray[np.float32], Dict[str, Any]]:
        gym.Env[npt.NDArray[np.float32], int].reset(self, seed=seed)
        self.current_step = 0
        self.load_turn = 0
        self.episode_count += 1

        self.sim.reset(init_mode=self._init_mode)

        # Use cloned pre-built requests to avoid state corruption between trials
        requests = [req.clone() for req in self._requests]

        self.sim.init_simulator(requests, BENCH_CONFIG.benchmark_length)
        self.sim.run()  # advance to the first action interval

        state_data = self.sim.get_state(self.current_step)
        obs = self._get_obs(state_data)

        return obs, {}
