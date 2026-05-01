from __future__ import annotations
import json
import yaml
import uuid
from pathlib import Path
from typing import Dict, Tuple, Set

import pickle
import src.simulation.models as m
from src.simulation.config import SimulationConfig, GPU_AGENTS_CONFIG
from src.simulation.mig_rule import MIGProfileRuleImpl

ModelsMapType = Dict[str, Dict[str, Tuple[int, int]]]
TokensMapType = Dict[m.AgentId, ModelsMapType]


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


def _load_model_tokens(filepath: Path) -> Dict[str, Tuple[int, int]]:
    """Helper for token map loading."""
    req_map = {}
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            req_map[str(data["id"])] = (
                data["prompt_tokens"],
                data["completion_tokens"],
            )
    return req_map


def init_tokens_map(base_dir: Path, sim_config: SimulationConfig) -> TokensMapType:

    cache_dir = base_dir / ".cache"
    cache_dir.mkdir(exist_ok=True)
    cache_file = cache_dir / "tokens_map.pkl"

    # Use a simple cache for now
    if cache_file.exists():
        with open(cache_file, "rb") as f:
            tokens_map = pickle.load(f)

            # Validate that all currently required models are in the cache
            required_models = set()
            for gpu_id in GPU_AGENTS_CONFIG:
                for agent_cfg in GPU_AGENTS_CONFIG[gpu_id].values():
                    for mig_cfg in agent_cfg["mig"].values():
                        required_models.add(mig_cfg["model"])

            all_cached_models = set()
            for agent_map in tokens_map.values():
                all_cached_models.update(agent_map.keys())

            if required_models.issubset(all_cached_models):
                return tokens_map

    tokens_map: TokensMapType = {}

    # Collect unique (model_name, generate_path) across all GPUs
    model_paths: Dict[str, Path] = {}
    for gpu_id in GPU_AGENTS_CONFIG:
        for agent_name, agent_cfg in GPU_AGENTS_CONFIG[gpu_id].items():
            for mig_cfg in agent_cfg["mig"].values():
                mname = mig_cfg["model"]
                if mname not in model_paths:
                    model_paths[mname] = base_dir / sim_config.get_generate_path(mname)

    # Load tokens map
    m_names = list(model_paths.keys())
    m_paths = [model_paths[n] for n in m_names]
    results = [_load_model_tokens(p) for p in m_paths]
    model_to_map = dict(zip(m_names, results))

    # Reconstruct tokens_map: { agent_id -> { model_name -> req_map } }
    for aid in m.AgentId:
        tokens_map[aid] = {}
        for gpu_id in GPU_AGENTS_CONFIG:
            for agent_name, agent_cfg in GPU_AGENTS_CONFIG[gpu_id].items():
                if agent_name == aid.value:
                    for mig_cfg in agent_cfg["mig"].values():
                        mname = mig_cfg["model"]
                        tokens_map[aid][mname] = model_to_map[mname]

    # Validate IDs consistency per agent (optional, keeping for safety)
    for agent_id in m.AgentId:
        if not tokens_map[agent_id]:
            continue
        message_id_sets = [set(m_map.keys()) for m_map in tokens_map[agent_id].values()]
        if not all(s == message_id_sets[0] for s in message_id_sets):
            raise ValueError(f"Request ID mismatch for agent {agent_id}")

    # Save to cache
    with open(cache_file, "wb") as f:
        pickle.dump(tokens_map, f)

    return tokens_map


base_dir = Path(".")
SIM_CONFIG: SimulationConfig = init_config(base_dir)

# Global token map: { m.AgentId -> { model_name -> { req_id -> (prompt_tokens, completion_tokens) } } }
TOKENS_MAP: TokensMapType = init_tokens_map(base_dir, SIM_CONFIG)

MIG_RULES = MIGProfileRuleImpl()

# Global set to track used engine IDs to ensure uniqueness
USED_EIDS: Set[str] = set()


def generate_engine_id(gpu: int, mig_str: str) -> str:
    """Generates a globally unique engine ID."""
    while True:
        eid = f"GPU_{gpu}_{mig_str}_{uuid.uuid4().hex[:4]}"
        if eid not in USED_EIDS:
            USED_EIDS.add(eid)
            return eid
