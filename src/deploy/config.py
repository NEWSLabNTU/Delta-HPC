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

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import yaml

from src.deploy.models import SimulatedGPUConfig


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
