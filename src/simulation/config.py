from __future__ import annotations

import random
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Any

import src.simulation.models as m


@dataclass
class SimulationConfig:
    # simulation.initial_state
    initial_state: List[Dict[str, Any]]
    # simulation.agents: { agent -> { dataset, mig -> { model, restart_time, param } } }
    agents_configs: Dict[str, Any]
    # model: { model_name -> { generate_path, vllm_config } }
    model: Dict[str, Any]
    # Populated after loading vllm config files
    max_batched_tokens: Dict[str, int] = field(default_factory=dict[str, int])

    _base_engines: List[Dict[str, Any]] = field(init=False)

    def __post_init__(self):
        # Store the permanent engines from YAML
        self._base_engines = self.initial_state.copy()

    @classmethod
    def load(cls, config_path: Path) -> SimulationConfig:
        with open(config_path, "r") as f:
            data = yaml.safe_load(f)

        return cls(
            initial_state=data["simulation"]["initial_state"]["engines"],
            agents_configs=data["simulation"]["agents"],
            model=data["model"],
        )

    def generate_initial_state(self, mode: str) -> None:
        """
        Initializes the initial_state list based on the requested mode.
        Supported modes: "7g", "2_2_2_1", "random".
        """
        # Start with permanent engines only
        new_state = [e for e in self._base_engines if e.get("is-permanent", False)]

        for gpu in [0, 1]:
            aid = m.AgentId.CODING if gpu == 0 else m.AgentId.RAG
            if mode == "7g":
                combo = m.InitialMIGCombination.C7.value
            elif mode == "2_2_2_1":
                combo = m.InitialMIGCombination.C2_2_2_1.value
            elif mode == "random":
                combo = random.choice(list(m.InitialMIGCombination)).value
            else:
                raise ValueError(f"Unknown initialization mode: {mode}")

            for mig in combo:
                new_state.append(
                    {
                        "gpu": gpu,
                        "mig": mig.string,
                        "agent": aid.value,
                        "is-permanent": False,
                    }
                )
        self.initial_state = new_state

    def get_model(self, agent: m.AgentId, mig_profile: m.MIGProfile) -> str:
        """Return the model name for a given agent + MIG profile."""
        return self.agents_configs[agent.value]["mig"][mig_profile.string]["model"]

    def get_dataset(self, agent: m.AgentId) -> str:
        """Return the dataset path for a given agent."""
        return self.agents_configs[agent.value]["dataset"]

    def get_restart_time(self, agent: m.AgentId, mig_profile: m.MIGProfile) -> float:
        """Return the restart time for a given agent + MIG profile."""
        return self.agents_configs[agent.value]["mig"][mig_profile.string][
            "restart_time"
        ]

    def get_prefill_params(
        self, agent: m.AgentId, mig_profile: m.MIGProfile
    ) -> m.ParamDict:
        """Return prefill regression params for a given agent + MIG profile."""
        return self.agents_configs[agent.value]["mig"][mig_profile.string]["param"][
            "prefill"
        ]

    def get_tpot_params(
        self, agent: m.AgentId, mig_profile: m.MIGProfile
    ) -> m.ParamDict:
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

    def get_kv_cache_gb(self, agent: m.AgentId, mig_profile: m.MIGProfile) -> float:
        """Return the kv_cache_GB for a given agent + MIG profile."""
        return self.agents_configs[agent.value]["mig"][mig_profile.string][
            "kv_cache_GB"
        ]

    def get_kv_per_token_kb(self, model_name: str) -> float:
        """Return the kv_per_token_KB for a given model."""
        return self.model[model_name]["kv_per_token_KB"]

    def get_max_kv_cache_tokens(
        self, agent: m.AgentId, mig_profile: m.MIGProfile
    ) -> int:
        """Return the max concurrent KV cache tokens for a given agent + MIG profile."""
        mname = self.get_model(agent, mig_profile)
        kv_cache_gb = self.get_kv_cache_gb(agent, mig_profile)
        kv_per_token_kb = self.get_kv_per_token_kb(mname)
        # 1 GB = 1048576 KB
        return int((kv_cache_gb * 1048576) / kv_per_token_kb)

    def get_rag_overhead(self) -> float:
        overheads = self.agents_configs[m.AgentId.RAG.value]["search-overhead"]
        assert overheads["model"] == "random"  # support random for now
        min_, max_ = overheads["min"], overheads["max"]
        return random.uniform(min_, max_)
