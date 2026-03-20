from __future__ import annotations
import json
import yaml
from pathlib import Path
from typing import Dict, Tuple, Set

from models import AgentId, MIGProfile
from config import SimulationConfig
from bidict import bidict

type ModelsMapType = Dict[str, Dict[str, Tuple[int, int]]]
type TokensMapType = Dict[AgentId, ModelsMapType]

# Maps (MIGProfile, MIGProfile) <-> Merged MIGProfile
# Always use canonical sorted tuple (by size descending) for lookups
MIG_MERGE_RULES: bidict[Tuple[MIGProfile, MIGProfile], MIGProfile] = bidict(
    {
        (MIGProfile.MIG_4G_20GB, MIGProfile.MIG_3G_20GB): MIGProfile.MIG_7G_40GB,
        (MIGProfile.MIG_2G_10GB, MIGProfile.MIG_1G_10GB): MIGProfile.MIG_3G_20GB,
        (MIGProfile.MIG_2G_10GB, MIGProfile.MIG_2G_10GB): MIGProfile.MIG_4G_20GB,
    }
)


def init_config(base_dir: Path) -> SimulationConfig:
    sim_config = SimulationConfig.load(base_dir / "configs" / "simulation_config.yaml")
    # Load max_num_batched_tokens from each model's vllm config file
    for model_name in sim_config.model:
        filepath = base_dir / sim_config.get_vllm_config_path(model_name)
        with open(filepath, "r") as f:
            data = yaml.safe_load(f)
            if "max_num_batched_tokens" not in data:
                raise ValueError(
                    f"Missing 'max_num_batched_tokens' in {filepath} for model {model_name}"
                )
            sim_config.max_batched_tokens[model_name] = data["max_num_batched_tokens"]
    return sim_config


def init_tokens_map(base_dir: Path) -> TokensMapType:
    # --- TOKENS_MAP ---
    tokens_map: TokensMapType = {}
    for agent_id in AgentId:
        model_map: ModelsMapType = {}
        agent_cfg = SIM_CONFIG.simulation_configs[agent_id.value]
        seen_models: Set[str] = set()
        for mig_cfg in agent_cfg["mig"].values():
            model_name = mig_cfg["model"]
            if model_name in seen_models:
                continue
            seen_models.add(model_name)
            filepath = base_dir / SIM_CONFIG.get_generate_path(model_name)
            req_map = {}
            with open(filepath, "r") as f:
                for line in f:
                    data = json.loads(line)
                    req_map[str(data["id"])] = (
                        data["prompt_tokens"],
                        data["completion_tokens"],
                    )
            model_map[model_name] = req_map
        tokens_map[agent_id] = model_map

    # Check that all models have the same set of request IDs per agent
    for agent_id in AgentId:
        message_id_sets = [
            set(tokens_map[agent_id][model_name].keys())
            for model_name in tokens_map[agent_id]
        ]
        if not all(s == message_id_sets[0] for s in message_id_sets):
            raise ValueError(
                f"Requests not present in all req_maps for agent {agent_id}"
            )

    return tokens_map


base_dir = Path(".")
SIM_CONFIG: SimulationConfig = init_config(base_dir)

# Global token map: { AgentId -> { model_name -> { req_id -> (prompt_tokens, completion_tokens) } } }
TOKENS_MAP: TokensMapType = init_tokens_map(base_dir)
