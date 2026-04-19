from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import numpy.typing as npt

from src.simulation.env import BaseMIGResourceEnv
from src.bench.models import BenchMode
from src.bench.config import BENCH_CONFIG
import src.simulation.models as m


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
        init_mode: m.InitialMIGCombination
        | Tuple[m.InitialMIGCombination, m.InitialMIGCombination],
    ):
        super().__init__(simulator)
        self.bench_mode = bench_mode
        self._requests = requests
        self._init_mode = init_mode
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

        # Setup Hardware
        self.sim.reset(init_mode=self._init_mode)

        # Setup Workload (cloned to allow trial reuse)
        requests = [req.clone() for req in self._requests]
        self.sim.init_simulator(requests, self.max_steps)
        self.sim.run()

        state_data = self.sim.get_state()
        return self._get_obs(state_data), {}
