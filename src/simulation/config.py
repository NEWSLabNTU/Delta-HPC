from __future__ import annotations

import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Any, Literal

from src.simulation.models import *


@dataclass
class SimulationConfig:
    # simulation.initial_state
    initial_state: List[Dict[str, Any]]
    # simulation.resource-manager
    resource_manager: Dict[Literal["action_interval"], float]
    # simulation.agents: { agent -> { dataset, mig -> { model, restart_time, param } } }
    agents_configs: Dict[str, Any]
    # model: { model_name -> { generate_path, vllm_config } }
    model: Dict[str, Any]
    # Populated after loading vllm config files
    max_batched_tokens: Dict[str, int] = field(default_factory=dict[str, int])

    @classmethod
    def load(cls, config_path: Path) -> SimulationConfig:
        with open(config_path, "r") as f:
            data = yaml.safe_load(f)

        return cls(
            initial_state=data["simulation"]["initial_state"]["engines"],
            resource_manager=data["simulation"]["resource-manager"],
            agents_configs=data["simulation"]["agents"],
            model=data["model"],
        )

    def get_rl_action_interval(self) -> float:
        """Return the action interval."""
        return self.resource_manager["action_interval"]

    def get_model(self, agent: AgentId, mig_profile: MIGProfile) -> str:
        """Return the model name for a given agent + MIG profile."""
        return self.agents_configs[agent.value]["mig"][mig_profile.string]["model"]

    def get_dataset(self, agent: AgentId) -> str:
        """Return the dataset path for a given agent."""
        return self.agents_configs[agent.value]["dataset"]

    def get_restart_time(self, agent: AgentId, mig_profile: MIGProfile) -> float:
        """Return the restart time for a given agent + MIG profile."""
        return self.agents_configs[agent.value]["mig"][mig_profile.string][
            "restart_time"
        ]

    def get_prefill_params(self, agent: AgentId, mig_profile: MIGProfile) -> ParamDict:
        """Return prefill regression params for a given agent + MIG profile."""
        return self.agents_configs[agent.value]["mig"][mig_profile.string]["param"][
            "prefill"
        ]

    def get_tpot_params(self, agent: AgentId, mig_profile: MIGProfile) -> ParamDict:
        """Return tpot regression params for a given agent + MIG profile."""
        return self.agents_configs[agent.value]["mig"][mig_profile.string]["param"][
            "tpot"
        ]

    def get_generate_path(self, model_name: str) -> str:
        """Return the generated JSONL path for a given model."""
        return self.model[model_name]["generate_path"]

    def get_vllm_config_path(self, model_name: str) -> str:
        """Return the vllm config file path for a given model."""
        return self.model[model_name]["vllm_config"]

    def get_kv_cache_gb(self, agent: AgentId, mig_profile: MIGProfile) -> float:
        """Return the kv_cache_GB for a given agent + MIG profile."""
        return self.agents_configs[agent.value]["mig"][mig_profile.string][
            "kv_cache_GB"
        ]

    def get_kv_per_token_kb(self, model_name: str) -> float:
        """Return the kv_per_token_KB for a given model."""
        return self.model[model_name]["kv_per_token_KB"]

    def get_max_kv_cache_tokens(self, agent: AgentId, mig_profile: MIGProfile) -> int:
        """Return the max concurrent KV cache tokens for a given agent + MIG profile."""
        mname = self.get_model(agent, mig_profile)
        kv_cache_gb = self.get_kv_cache_gb(agent, mig_profile)
        kv_per_token_kb = self.get_kv_per_token_kb(mname)
        # 1 GB = 1048576 KB
        return int((kv_cache_gb * 1048576) / kv_per_token_kb)
