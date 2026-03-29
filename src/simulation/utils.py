from __future__ import annotations
import json
import yaml
import uuid
from pathlib import Path
from typing import Dict, Tuple, Set

import src.simulation.models as m
from src.simulation.config import SimulationConfig
from src.simulation.mig_rule import MIGProfileRuleImpl

type ModelsMapType = Dict[str, Dict[str, Tuple[int, int]]]
type TokensMapType = Dict[m.AgentId, ModelsMapType]


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
    for agent_id in m.AgentId:
        model_map: ModelsMapType = {}
        agent_cfg = SIM_CONFIG.agents_configs[agent_id.value]
        seen_models: Set[str] = set()
        for mig_cfg in agent_cfg["mig"].values():
            model_name = mig_cfg["model"]
            if model_name in seen_models:
                continue
            seen_models.add(model_name)
            filepath = base_dir / SIM_CONFIG.get_generate_path(model_name)
            req_map = {}
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line)
                    req_map[str(data["id"])] = (
                        data["prompt_tokens"],
                        data["completion_tokens"],
                    )
            model_map[model_name] = req_map
        tokens_map[agent_id] = model_map

    # Check that all models have the same set of request IDs per agent
    for agent_id in m.AgentId:
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

# Global token map: { m.AgentId -> { model_name -> { req_id -> (prompt_tokens, completion_tokens) } } }
TOKENS_MAP: TokensMapType = init_tokens_map(base_dir)

MIG_RULES = MIGProfileRuleImpl()

# Global set to track used engine IDs to ensure uniqueness
USED_EIDS: Set[str] = set()


def generate_engine_id(agent_name: str, gpu: int, mig_str: str) -> str:
    """Generates a globally unique engine ID."""
    while True:
        eid = f"{agent_name}_GPU_{gpu}_{mig_str}_{uuid.uuid4().hex[:4]}"
        if eid not in USED_EIDS:
            USED_EIDS.add(eid)
            return eid
