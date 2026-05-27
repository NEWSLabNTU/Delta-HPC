from typing import Any, Dict, List, Literal, Optional, Tuple
import numpy as np
import numpy.typing as npt

import src.share.models as m
from src.share.env import BaseMIGResourceEnv
from src.bench.models import BenchMode
from src.bench.config import BENCH_CONFIG


class BenchMIGResourceEnv(BaseMIGResourceEnv):
    """
    Concrete Evaluation Environment for MIG Resource Management.
    Handles fixed workloads and protects against destructive auto-resets.
    """

    def __init__(
        self,
        simulator: m.Simulator,
        bench_mode: BenchMode,
        requests: List[m.Request],
    ):
        # Heuristic mode bypasses action cooldowns (matches deployment behaviour)
        ignore_cooldowns = bench_mode == BenchMode.BASELINE_HEURISTIC
        super().__init__(simulator, ignore_cooldowns=ignore_cooldowns)
        self.bench_mode = bench_mode
        self._requests = requests
        self._is_initialized = False

        # Overwrite episode length
        self.max_steps = BENCH_CONFIG.benchmark_length

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[npt.NDArray[np.float32], Dict[str, Any]]:
        # If already initialized, we ignore subsequent resets to preserve state
        # for the flush phase (prevents auto-reset from RL wrappers).
        if self._is_initialized:
            state_data = self.sim.get_state()
            return self._get_obs(state_data), {}

        # First reset: Initialize simulation with benchmark workload
        super().reset(seed=seed)
        self._is_initialized = True

        # Determine initial hardware state mode based on bench mode
        _MODE_MAP: Dict[BenchMode, Literal["random", "no_mig", "split_extreme"]] = {
            BenchMode.STATIC_NO_MIG: "no_mig",
            BenchMode.STATIC_SPLIT_EXTREME: "split_extreme",
        }
        initial_state_mode: Literal["random", "no_mig", "split_extreme"] = (
            _MODE_MAP.get(self.bench_mode, "random")
        )

        # Setup Hardware (fixed or random depending on bench mode)
        self.sim.reset(initial_state_mode=initial_state_mode)

        # Setup Workload (cloned to allow trial reuse)
        requests = [req.clone() for req in self._requests]
        self.sim.init_simulator(requests, self.max_steps)
        self.sim.run()

        state_data = self.sim.get_state()
        return self._get_obs(state_data), {}
