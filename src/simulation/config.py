from __future__ import annotations
from typing import Dict, List, Any, Type, Literal

import random
import yaml
from pathlib import Path
from dataclasses import dataclass, field
import logging

import importlib.util
import src.share.models as m
from src.share.mig_matrix import STATE_DEFINITIONS

logger = logging.getLogger(__name__)

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
    # datasets: { dataset_name -> local disk path }
    datasets: Dict[str, str] = field(default_factory=dict)
    # Populated after loading vllm config files
    max_batched_tokens: Dict[str, int] = field(default_factory=dict[str, int])

    _base_engines: List[Dict[str, Any]] = field(init=False)

    def __post_init__(self):
        # Store the permanent engines from YAML
        self._base_engines = []
        for e in self.initial_state:
            new_e = e.copy()
            new_e["is-permanent"] = True
            self._base_engines.append(new_e)

    @property
    def num_managed_gpus(self) -> int:
        permanent_gpus = {e["gpu"] for e in self._base_engines}
        total_gpus = set(self.cluster.keys())
        return len(total_gpus - permanent_gpus)

    @property
    def active_agents(self) -> List[m.AgentId]:
        return [m.AgentId(e["agent"]) for e in self._base_engines]

    @property
    def managed_gpus(self) -> List[int]:
        permanent_gpus = {e["gpu"] for e in self._base_engines}
        total_gpus = set(self.cluster.keys())
        return sorted(list(total_gpus - permanent_gpus))

    @classmethod
    def load(
        cls, config_path: Path, use_hardware_detection: bool = False
    ) -> SimulationConfig:

        with open(config_path, "r") as f:
            data = yaml.safe_load(f)

        sim_data = data["simulation"]

        active_agents = list(sim_data.get("agents", {}).keys())
        num_agents = len(active_agents)

        gpu_model = sim_data.get("gpu_model", "A100_40GB")
        gpu_to_model = {i: gpu_model for i in range(num_agents)}
        gpu_to_model[num_agents] = "A100_40GB"

        if use_hardware_detection:
            from src.deploy.mig_controller import MIGController

            try:
                detected = MIGController.detect_mig_gpus()
                gpu_to_model = {}
                for d in detected:
                    gpu_to_model[d.gpu_idx] = d.model_name
                logger.info(
                    "SimulationConfig loaded with hardware override: %s", gpu_to_model
                )
            except Exception as e:
                logger.error("Hardware detection failed: %s", e)
                raise RuntimeError(f"Hardware detection failed: {e}") from e

        agent_defs = sim_data["agents"]

        loaded_modules = {}

        # Load GPU Registries
        for gpu_id, model_name in gpu_to_model.items():
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

            # Derive VALID_COMBINATIONS from STATE_DEFINITIONS.
            # EXCLUSION: We explicitly exclude profiles marked as unsupported by the GPU model class.
            unsupported = mod.MIG_PROFILE.unsupported_profiles()
            supported = {
                p.profile_type
                for p in mod.MIG_PROFILE
                if p.profile_type not in unsupported
            }

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

        gpu_ids = sorted(list(gpu_to_model.keys()))
        if not gpu_ids:
            raise ValueError("No GPUs available in cluster configuration.")

        permanent_gpu = gpu_ids[-1]
        managed_gpus = gpu_ids[:-1]

        initial_agents = {gid: [] for gid in managed_gpus}
        if managed_gpus and active_agents:
            for idx, agent_name in enumerate(active_agents):
                gid = managed_gpus[idx % len(managed_gpus)]
                initial_agents[gid].append(agent_name)

        prof_2g = next((p for p in GPU_MIG_PROFILE[permanent_gpu] if p.profile_type == m.MIGProfile.MIG_2G), None)
        if not prof_2g:
            prof_2g = sorted(GPU_MIG_PROFILE[permanent_gpu], key=lambda x: x.profile_type.value)[0]
            
        permanent_engines = [
            {"agent": agent_name, "gpu": permanent_gpu, "mig": prof_2g.string}
            for agent_name in active_agents
        ]

        # Verify that all agents have proper configs on each GPU model listed
        all_models = set(gpu_to_model.values())
        for model_name in all_models:
            for agent_id in [m.AgentId(a) for a in agent_defs.keys()]:
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
                # Verify hardware MIG profiles that are part of valid combinations
                used_profile_types = {
                    p for combo in GPU_VALID_COMBINATIONS[gpu_id] for p in combo
                }
                for hw_prof in GPU_MIG_PROFILE[gpu_id]:
                    if hw_prof.profile_type in used_profile_types:
                        if hw_prof.string not in agent_mig_configs:
                            raise ValueError(
                                f"Agent {agent_id.value} on {model_name} lacks configuration for MIG profile {hw_prof.string}"
                            )

        return cls(
            cluster=gpu_to_model,
            initial_state=permanent_engines,
            gpu_initial_agents=initial_agents,
            model=data["model"],
            datasets=data["datasets"],
        )

    def _pad_partial_gpu_states(
        self,
        new_state: List[Dict[str, Any]],
        mode: Literal["random", "no_mig", "split_extreme"],
    ) -> None:
        """
        Takes the current permanent engines in `new_state` and checks if they form
        strict subsets of valid GPU hardware states. If so, pads them to a full state.
        Raises ValueError if the combination is physically impossible.
        """
        gpu_to_engines = {}
        for e in new_state:
            gpu_id = e["gpu"]
            gpu_to_engines.setdefault(gpu_id, []).append(e)

        for gpu_id, engines in gpu_to_engines.items():
            logical_profiles = []
            for e in engines:
                mig_str = e["mig"]
                hw_prof = next(
                    p for p in GPU_MIG_PROFILE[gpu_id] if p.string == mig_str
                )
                logical_profiles.append(hw_prof.profile_type)

            logical_profiles_sorted = sorted(
                logical_profiles, key=lambda x: x.value, reverse=True
            )
            valid_combos = GPU_VALID_COMBINATIONS[gpu_id]

            exact_match = False
            for combo in valid_combos:
                if (
                    sorted(list(combo), key=lambda x: x.value, reverse=True)
                    == logical_profiles_sorted
                ):
                    exact_match = True
                    break

            if exact_match:
                continue

            supersets = []
            for combo in valid_combos:
                combo_list = list(combo)
                is_subset = True
                for lp in set(logical_profiles):
                    if logical_profiles.count(lp) > combo_list.count(lp):
                        is_subset = False
                        break
                if is_subset:
                    supersets.append(combo)

            if not supersets:
                raise ValueError(
                    f"GPU {gpu_id} permanent engines {logical_profiles} do not form a valid hardware state subset. "
                    f"Check NVIDIA MIG specifications or STATE_DEFINITIONS."
                )

            if mode == "no_mig":
                chosen_combo = min(supersets, key=lambda c: len(c))
            elif mode == "split_extreme":
                chosen_combo = max(supersets, key=lambda c: len(c))
            else:
                chosen_combo = random.choice(supersets)

            chosen_list = list(chosen_combo)
            for lp in logical_profiles:
                chosen_list.remove(lp)

            agent_names = self.gpu_initial_agents.get(gpu_id, [])
            if not agent_names:
                agent_names = list(GPU_AGENTS_CONFIG[gpu_id].keys())

            for lp in chosen_list:
                hw_prof = next(
                    p for p in GPU_MIG_PROFILE[gpu_id] if p.profile_type == lp
                )
                valid_agents = [
                    aname
                    for aname in agent_names
                    if aname in GPU_AGENTS_CONFIG[gpu_id]
                    and hw_prof.string in GPU_AGENTS_CONFIG[gpu_id][aname]["mig"]
                ]
                if not valid_agents:
                    raise ValueError(
                        f"No valid agent to pad profile {hw_prof.string} on GPU {gpu_id}"
                    )

                agent_name = random.choice(valid_agents)
                new_state.append({
                    "gpu": gpu_id,
                    "mig": hw_prof.string,
                    "agent": agent_name,
                    "is-permanent": True,
                    "is-unused": True,
                })

    def generate_initial_state(self) -> None:
        """
        Initializes the initial_state list with a random valid combination for each GPU.
        """
        new_state = [e.copy() for e in self._base_engines]

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

                new_state.append({
                    "gpu": gpu_id,
                    "mig": hw_prof.string,
                    "agent": agent_name,
                    "is-permanent": False,
                })
        self._pad_partial_gpu_states(new_state, mode="random")
        self.initial_state = new_state

    def generate_no_mig_initial_state(self) -> None:
        """
        Initializes the initial_state with a single 7G MIG instance for each GPU.
        Used by the STATIC_NO_MIG baseline.
        """
        new_state = [e.copy() for e in self._base_engines]

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

            new_state.append({
                "gpu": gpu_id,
                "mig": hw_prof.string,
                "agent": agent_name,
                "is-permanent": False,
            })
        self._pad_partial_gpu_states(new_state, mode="no_mig")
        self.initial_state = new_state

    def generate_split_extreme_initial_state(self) -> None:
        """
        Initializes the initial_state with the most-split valid combination for each GPU
        (the combination with the highest number of MIG slices).
        Used by the STATIC_SPLIT_EXTREME baseline.
        """
        new_state = [e.copy() for e in self._base_engines]

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

                new_state.append({
                    "gpu": gpu_id,
                    "mig": hw_prof.string,
                    "agent": agent_name,
                    "is-permanent": False,
                })
        self._pad_partial_gpu_states(new_state, mode="split_extreme")
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

    def get_dataset_path(self, name: str) -> str:
        """Return the local disk path for a named dataset.

        The key must be an AgentId enum value (e.g. 'CodingAgent', 'RAGAgent').
        """
        if name not in self.datasets:
            raise KeyError(
                f"Dataset '{name}' not found in simulation config. "
                f"Available: {list(self.datasets.keys())}"
            )
        return self.datasets[name]

    def get_generate_path(
        self, agent: m.AgentId, mig_profile: m.MIGProfileBase, gpu_id: int = 0
    ) -> str:
        return GPU_AGENTS_CONFIG[gpu_id][agent.value]["mig"][mig_profile.string][
            "generate_path"
        ]

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
