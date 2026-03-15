from __future__ import annotations
import json
import yaml
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from config import SimulationConfig

# Global singleton config. Initialised once via init() before anything else runs.
SIM_CONFIG: "SimulationConfig" = None

# Global token map: { AgentId -> { model_name -> { req_id -> (prompt_tokens, completion_tokens) } } }
TOKENS_MAP: dict = {}


def init(base_dir: Path) -> None:
    """Initialise all global singletons. Must be called once at the start of main()."""
    global SIM_CONFIG, TOKENS_MAP

    from config import SimulationConfig
    from models import AgentId

    # --- SIM_CONFIG ---
    SIM_CONFIG = SimulationConfig.load(base_dir / "configs" / "simulation_config.yaml")

    # Load max_num_batched_tokens from each model's vllm config file
    for model_name in SIM_CONFIG.model:
        filepath = base_dir / SIM_CONFIG.get_vllm_config_path(model_name)
        with open(filepath, "r") as f:
            data = yaml.safe_load(f)
            if "max_num_batched_tokens" not in data:
                raise ValueError(
                    f"Missing 'max_num_batched_tokens' in {filepath} for model {model_name}"
                )
            SIM_CONFIG.max_batched_tokens[model_name] = data["max_num_batched_tokens"]

    # --- TOKENS_MAP ---
    tokens_map: dict = {}
    for agent_id in AgentId:
        model_map: dict = {}
        agent_cfg = SIM_CONFIG.simulation_configs[agent_id.value]
        seen_models: set = set()
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

    TOKENS_MAP = tokens_map
