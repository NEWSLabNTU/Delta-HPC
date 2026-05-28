"""
src/deploy/config.py

Deployment-time configuration loaded from ``configs/deployment.yaml``.

This module exposes a module-level singleton ``DEPLOY_CONFIG`` so that
``cluster.py`` and ``vllm.py`` can import it without re-reading the file.

Example
-------
::

    from src.deploy.config import DEPLOY_CONFIG

    agent = DEPLOY_CONFIG.gpu_assignment[0]   # "CodingAgent"
    port  = DEPLOY_CONFIG.vllm.base_port      # 8100
"""

from __future__ import annotations

import yaml
from pathlib import Path
from typing import Dict, Union, Tuple
from dataclasses import dataclass

import src.share.models as m
from src.deploy.models import SimulatedGPUConfig
from src.simulation.config import GPU_MIG_PROFILE
from src.bench.models import Workload


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------


@dataclass
class VLLMDeployConfig:
    """vLLM-related deployment parameters.

    Attributes
    ----------
    base_port:
        First TCP port to allocate.  Subsequent slots receive ``base_port + n``.
    script:
        Path to ``launch_vllm.sh`` (relative to repo root).
    compose_file:
        Path to ``docker-compose.yaml`` (relative to repo root).
    health_timeout_s:
        Seconds to wait for a container to pass its ``/health`` check.
    request_timeout_s:
        HTTP timeout (seconds) for inference and metrics calls.
    """

    base_port: int
    script: Path
    compose_file: Path
    health_timeout_s: float
    request_timeout_s: float


@dataclass
class DashboardDeployConfig:
    """Dashboard-related deployment parameters.

    Attributes
    ----------
    host:
        The host to bind the web server to (e.g. "0.0.0.0").
    port:
        The port to run the web server on (e.g. 9000).
    """

    host: str
    port: int


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------


class DeploymentConfig:
    """Reads and exposes ``configs/deployment.yaml``.

    Attributes
    ----------
    gpu_assignment : dict[int, str]
        Maps each physical GPU index to the name of the simulation agent that
        initially occupies it (e.g. ``{0: "CodingAgent", 1: "RAGAgent"}``).
    vllm : VLLMDeployConfig
        vLLM-specific parameters.
    dashboard : DashboardDeployConfig
        Dashboard web server parameters.
    """

    def __init__(self, config_path: Path) -> None:
        with open(config_path, "r") as f:
            data = yaml.safe_load(f)

        self.gpu_assignment: Dict[int, str] = {
            int(k): v for k, v in data["gpu_assignment"].items()
        }

        v = data["vllm"]
        self.vllm = VLLMDeployConfig(
            base_port=int(v["base_port"]),
            script=Path(v["script"]),
            compose_file=Path(v["compose_file"]),
            health_timeout_s=float(v.get("health_timeout_s", 300)),
            request_timeout_s=float(v.get("request_timeout_s", 60)),
        )

        d = data.get("dashboard", {})
        self.dashboard = DashboardDeployConfig(
            host=str(d.get("host", "0.0.0.0")),
            port=int(d.get("port", 9000)),
        )

        self.simulated_gpus: Dict[int, SimulatedGPUConfig] = {
            int(k): v for k, v in data.get("simulated_gpus", {}).items()
        }

        h = data.get("heuristic", {})
        self.heuristic_service_rates = h.get("service_rates", {})
        self._workloads = data["workloads"]
        self.seed = int(data.get("seed", 77))
        self.obs_arrival_rate_divisor: Dict[str, float] = {
            str(k): float(v)
            for k, v in data.get("obs_arrival_rate_divisor", {}).items()
        }

    def get_arrival_rate_divisor(self, agent_id: m.AgentId) -> float:
        """Get the observation arrival rate divisor for the given agent."""
        return self.obs_arrival_rate_divisor.get(agent_id.value, 1.0)

    def get_service_rate(
        self,
        agent_id: m.AgentId,
        mig_profile: Union[m.MIGProfile, m.MIGProfileBase],
        gpu_id: int = 0,
    ) -> float:
        if isinstance(mig_profile, m.MIGProfile):
            hw_prof = next(
                p for p in GPU_MIG_PROFILE[gpu_id] if p.profile_type == mig_profile
            )
        else:
            hw_prof = mig_profile

        gpu_model = hw_prof.gpu_model

        # Structure: service_rates[gpu_model][agent_id][mig_str]
        model_rates = self.heuristic_service_rates.get(gpu_model, {})
        agent_rates = model_rates.get(agent_id.value, {})

        prof_str = hw_prof.string
        original_rate = float(agent_rates.get(prof_str, 0.0))

        match hw_prof.profile_type:
            case m.MIGProfile.MIG_7G:
                factor = 1.0
            case m.MIGProfile.MIG_4G | m.MIGProfile.MIG_3G:
                factor = 0.8
            case (
                m.MIGProfile.MIG_2G
                | m.MIGProfile.MIG_1G_LARGE
                | m.MIGProfile.MIG_1G_SMALL
            ):
                factor = 0.5
            case _:
                raise ValueError(f"Unknown MIG profile type: {hw_prof.profile_type}")

        return original_rate * factor

    def get_rate_range(
        self, workload: Workload, agent_id: m.AgentId
    ) -> Tuple[float, float]:
        cfg = self._workloads[workload.value]["rate"]
        if isinstance(cfg, dict):
            cfg = cfg[agent_id.value]
        return float(cfg[0]), float(cfg[1])

    def get_duration_range(self, workload: Workload) -> Tuple[float, float]:
        cfg = self._workloads[workload.value]["duration"]
        return float(cfg[0]), float(cfg[1])

    @classmethod
    def load(cls, path: Path = Path("configs/deployment.yaml")) -> "DeploymentConfig":
        """Load and return a :class:`DeploymentConfig` from *path*."""
        return cls(path)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

#: Global deployment configuration.  Import and use directly::
#:
#:     from src.deploy.config import DEPLOY_CONFIG
DEPLOY_CONFIG: DeploymentConfig = DeploymentConfig.load()
