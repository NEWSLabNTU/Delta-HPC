from __future__ import annotations
from typing import Dict, List, Any, Type

import random
import yaml
from pathlib import Path
from dataclasses import dataclass, field

import importlib.util
import src.simulation.models as m
from src.simulation.mig_matrix import STATE_DEFINITIONS

# Hardware Constants
NUM_MIG_SLICES = 7

# Global Registries for Multi-GPU support
GPU_MIG_PROFILE: Dict[int, Type[m.MIGProfileBase]] = {}
GPU_VALID_COMBINATIONS: Dict[int, Any] = {}
GPU_AGENTS_CONFIG: Dict[int, Dict[str, Any]] = {}


@dataclass
class SimulationConfig:
    # simulation.cluster
    cluster: Dict[int, str]
    # simulation.initial_state.permanent_engines
    initial_state: List[Dict[str, Any]]
    # simulation.initial_state.gpu_initial_agents
    gpu_initial_agents: Dict[int, List[str]]
    # Global model registry: { model_name -> { generate_path, vllm_config } }
    model: Dict[str, Any]
    # Populated after loading vllm config files
    max_batched_tokens: Dict[str, int] = field(default_factory=dict[str, int])

    _base_engines: List[Dict[str, Any]] = field(init=False)

    def __post_init__(self):
        # Store the permanent engines from YAML
        self._base_engines = self.initial_state.copy()

    @property
    def num_managed_gpus(self) -> int:
        permanent_gpus = {e["gpu"] for e in self._base_engines}
        total_gpus = set(self.cluster.keys())
        return len(total_gpus - permanent_gpus)

    @classmethod
    def load(cls, config_path: Path) -> SimulationConfig:

        with open(config_path, "r") as f:
            data = yaml.safe_load(f)

        sim_data = data["simulation"]
        gpu_to_model = sim_data["cluster"]
        agent_defs = sim_data["agents"]

        loaded_modules = {}

        # Load GPU Registries
        for gpu_id_str, model_name in gpu_to_model.items():
            gpu_id = int(gpu_id_str)

            if model_name not in loaded_modules:
                module_path = Path(f"configs/gpus/{model_name}.py")
                spec = importlib.util.spec_from_file_location(
                    f"gpu_mod_{model_name}", module_path
                )
                assert spec is not None and spec.loader is not None, (
                    f"Failed to load module for {model_name}"
                )
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                loaded_modules[model_name] = mod

            mod = loaded_modules[model_name]
            GPU_MIG_PROFILE[gpu_id] = mod.MIG_PROFILE

            # Derive VALID_COMBINATIONS from STATE_DEFINITIONS based on profiles defined in the GPU model class
            supported = {p.profile_type for p in mod.MIG_PROFILE}

            valid_combos = []
            for profiles in STATE_DEFINITIONS.values():
                if all(p in supported for p in profiles):
                    valid_combos.append(profiles)
            GPU_VALID_COMBINATIONS[gpu_id] = valid_combos

            # Reconstruct agent info for THIS gpu_id
            GPU_AGENTS_CONFIG[gpu_id] = {}
            for agent_name, agent_model_cfg in agent_defs.items():
                if model_name in agent_model_cfg:
                    GPU_AGENTS_CONFIG[gpu_id][agent_name] = agent_model_cfg[model_name]

        initial_agents = {
            int(k): v
            for k, v in sim_data["initial_state"]["gpu_initial_agents"].items()
        }

        # Verify that all agents have proper configs on each GPU model listed
        all_models = set(gpu_to_model.values())
        for model_name in all_models:
            for agent_id in m.AgentId:
                if agent_id.value not in agent_defs:
                    raise ValueError(
                        f"Agent {agent_id.value} not found in 'agents' config"
                    )
                if model_name not in agent_defs[agent_id.value]:
                    raise ValueError(
                        f"Agent {agent_id.value} lacks configuration for GPU model {model_name}"
                    )

                # Verify all hardware MIG profiles are supported by the agent
                # We can check this via one of the GPUs that uses this model
                gpu_id = next(
                    gid for gid, mname in gpu_to_model.items() if mname == model_name
                )
                agent_mig_configs = agent_defs[agent_id.value][model_name]["mig"]
                for hw_prof in GPU_MIG_PROFILE[gpu_id]:
                    if hw_prof.string not in agent_mig_configs:
                        raise ValueError(
                            f"Agent {agent_id.value} on {model_name} lacks configuration for MIG profile {hw_prof.string}"
                        )

        return cls(
            cluster=gpu_to_model,
            initial_state=sim_data["initial_state"]["permanent_engines"],
            gpu_initial_agents=initial_agents,
            model=data["model"],
        )

    def generate_initial_state(self) -> None:
        """
        Initializes the initial_state list with a random valid combination for each GPU.
        """
        new_state = [e.copy() for e in self._base_engines]
        for e in new_state:
            e["is-permanent"] = True

        occupied_gpus = {e["gpu"] for e in new_state}

        for gpu_id, agent_names in self.gpu_initial_agents.items():
            if gpu_id in occupied_gpus:
                continue

            # Use hardware-specific valid combinations for randomness
            valid_combos = GPU_VALID_COMBINATIONS[gpu_id]
            combo_logical = random.choice(valid_combos)

            # combo_logical is a tuple of logical MIGProfile members
            for logical_mig in combo_logical:
                # Find hardware string for this logical profile on this GPU
                hw_prof = next(
                    p for p in GPU_MIG_PROFILE[gpu_id] if p.profile_type == logical_mig
                )

                # Pick an agent from the initial assignment that supports this specific profile
                valid_agents = [
                    aname
                    for aname in agent_names
                    if aname in GPU_AGENTS_CONFIG[gpu_id]
                    and hw_prof.string in GPU_AGENTS_CONFIG[gpu_id][aname]["mig"]
                ]
                if not valid_agents:
                    continue

                agent_name = random.choice(valid_agents)

                new_state.append(
                    {
                        "gpu": gpu_id,
                        "mig": hw_prof.string,
                        "agent": agent_name,
                        "is-permanent": False,
                    }
                )
        self.initial_state = new_state

    def generate_no_mig_initial_state(self) -> None:
        """
        Initializes the initial_state with a single 7G MIG instance for each GPU.
        Used by the STATIC_NO_MIG baseline.
        """
        new_state = [e.copy() for e in self._base_engines]
        for e in new_state:
            e["is-permanent"] = True

        occupied_gpus = {e["gpu"] for e in new_state}

        for gpu_id, agent_names in self.gpu_initial_agents.items():
            if gpu_id in occupied_gpus:
                continue

            # Find the 7G hardware profile for this GPU
            hw_prof = next(
                p
                for p in GPU_MIG_PROFILE[gpu_id]
                if p.profile_type == m.MIGProfile.MIG_7G
            )

            # Pick an agent from initial assignment that supports this specific profile
            valid_agents = [
                aname
                for aname in agent_names
                if aname in GPU_AGENTS_CONFIG[gpu_id]
                and hw_prof.string in GPU_AGENTS_CONFIG[gpu_id][aname]["mig"]
            ]
            if not valid_agents:
                continue

            agent_name = random.choice(valid_agents)

            new_state.append(
                {
                    "gpu": gpu_id,
                    "mig": hw_prof.string,
                    "agent": agent_name,
                    "is-permanent": False,
                }
            )
        self.initial_state = new_state

    def generate_split_extreme_initial_state(self) -> None:
        """
        Initializes the initial_state with the most-split valid combination for each GPU
        (the combination with the highest number of MIG slices).
        Used by the STATIC_SPLIT_EXTREME baseline.
        """
        new_state = [e.copy() for e in self._base_engines]
        for e in new_state:
            e["is-permanent"] = True

        occupied_gpus = {e["gpu"] for e in new_state}

        for gpu_id, agent_names in self.gpu_initial_agents.items():
            if gpu_id in occupied_gpus:
                continue

            valid_combos = GPU_VALID_COMBINATIONS[gpu_id]
            # Most-split = combo with the largest number of slices
            combo_logical = max(valid_combos, key=lambda c: len(c))

            for logical_mig in combo_logical:
                hw_prof = next(
                    p for p in GPU_MIG_PROFILE[gpu_id] if p.profile_type == logical_mig
                )

                # Pick an agent from initial assignment that supports this specific profile
                valid_agents = [
                    aname
                    for aname in agent_names
                    if aname in GPU_AGENTS_CONFIG[gpu_id]
                    and hw_prof.string in GPU_AGENTS_CONFIG[gpu_id][aname]["mig"]
                ]
                if not valid_agents:
                    continue

                agent_name = random.choice(valid_agents)

                new_state.append(
                    {
                        "gpu": gpu_id,
                        "mig": hw_prof.string,
                        "agent": agent_name,
                        "is-permanent": False,
                    }
                )
        self.initial_state = new_state

    def get_model(
        self, agent: m.AgentId, mig_profile: m.MIGProfileBase, gpu_id: int = 0
    ) -> str:
        return GPU_AGENTS_CONFIG[gpu_id][agent.value]["mig"][mig_profile.string][
            "model"
        ]

    def get_restart_time(
        self, agent: m.AgentId, mig_profile: m.MIGProfileBase, gpu_id: int = 0
    ) -> float:
        return GPU_AGENTS_CONFIG[gpu_id][agent.value]["mig"][mig_profile.string][
            "restart_time"
        ]

    def get_prefill_params(
        self, agent: m.AgentId, mig_profile: m.MIGProfileBase, gpu_id: int = 0
    ) -> m.ParamDict:
        return GPU_AGENTS_CONFIG[gpu_id][agent.value]["mig"][mig_profile.string][
            "param"
        ]["prefill"]

    def get_tpot_params(
        self, agent: m.AgentId, mig_profile: m.MIGProfileBase, gpu_id: int = 0
    ) -> m.ParamDict:
        return GPU_AGENTS_CONFIG[gpu_id][agent.value]["mig"][mig_profile.string][
            "param"
        ]["tpot"]

    def get_kv_cache_gb(
        self, agent: m.AgentId, mig_profile: m.MIGProfileBase, gpu_id: int = 0
    ) -> float:
        return GPU_AGENTS_CONFIG[gpu_id][agent.value]["mig"][mig_profile.string][
            "kv_cache_GB"
        ]

    def get_generate_path(self, model_name: str) -> str:
        return self.model[model_name]["generate_path"]

    def get_vllm_config_path(self, model_name: str) -> str:
        return self.model[model_name]["vllm_config"]

    def get_kv_per_token_kb(self, model_name: str) -> float:
        return self.model[model_name]["kv_per_token_KB"]

    def get_max_kv_cache_tokens(
        self, agent: m.AgentId, mig_profile: m.MIGProfileBase, gpu_id: int = 0
    ) -> int:
        mname = self.get_model(agent, mig_profile, gpu_id)
        kv_cache_gb = self.get_kv_cache_gb(agent, mig_profile, gpu_id)
        kv_per_token_kb = self.get_kv_per_token_kb(mname)
        return int((kv_cache_gb * 1048576) / kv_per_token_kb)

    def get_rag_overhead(self, gpu_id: int = 1) -> float:
        overheads = GPU_AGENTS_CONFIG[gpu_id][m.AgentId.RAG.value]["search-overhead"]
        min_, max_ = overheads["min"], overheads["max"]
        return random.uniform(min_, max_)
