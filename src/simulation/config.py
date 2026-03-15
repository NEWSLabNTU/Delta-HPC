from dataclasses import dataclass, field
from typing import Dict, Any, List
from pathlib import Path
import yaml
from models import AgentId, ParamDict


@dataclass
class SimulationConfig:
    # simulation.initial_state
    initial_state: List[Dict[str, Any]]
    # simulation.configs: { agent -> { dataset, mig -> { model, restart_time, param } } }
    simulation_configs: Dict[str, Any]
    # model: { model_name -> { generate_path, vllm_config } }
    model: Dict[str, Any]
    # Populated after loading vllm config files
    max_batched_tokens: Dict[str, int] = field(default_factory=dict[str, int])

    @classmethod
    def load(cls, config_path: Path) -> "SimulationConfig":
        with open(config_path, "r") as f:
            data = yaml.safe_load(f)

        return cls(
            initial_state=data["simulation"]["initial_state"],
            simulation_configs=data["simulation"]["configs"],
            model=data["model"],
        )

    def get_model(self, agent: AgentId, mig_profile: str) -> str:
        """Return the model name for a given agent + MIG profile."""
        return self.simulation_configs[agent.value]["mig"][mig_profile]["model"]

    def get_dataset(self, agent: AgentId) -> str:
        """Return the dataset path for a given agent."""
        return self.simulation_configs[agent.value]["dataset"]

    def get_restart_time(self, agent: AgentId, mig_profile: str) -> float:
        """Return the restart time for a given agent + MIG profile."""
        return self.simulation_configs[agent.value]["mig"][mig_profile]["restart_time"]

    def get_prefill_params(self, agent: AgentId, mig_profile: str) -> ParamDict:
        """Return prefill regression params for a given agent + MIG profile."""
        return self.simulation_configs[agent.value]["mig"][mig_profile]["param"][
            "prefill"
        ]

    def get_tpot_params(self, agent: AgentId, mig_profile: str) -> ParamDict:
        """Return tpot regression params for a given agent + MIG profile."""
        return self.simulation_configs[agent.value]["mig"][mig_profile]["param"]["tpot"]

    def get_generate_path(self, model_name: str) -> str:
        """Return the generated JSONL path for a given model."""
        return self.model[model_name]["generate_path"]

    def get_vllm_config_path(self, model_name: str) -> str:
        """Return the vllm config file path for a given model."""
        return self.model[model_name]["vllm_config"]
