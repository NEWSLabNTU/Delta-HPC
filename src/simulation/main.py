import json
import yaml
from pathlib import Path
from collections import defaultdict
from src.simulation.models import Request, AgentId
from src.simulation.engine import LLMEngine
from src.simulation.agent import Agent
from src.simulation.simulator import Simulator

from datasets import load_from_disk


def load_config(config_path: Path) -> dict:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    required_keys = [
        "agents",
        "generated",
        "vllm_config",
        "regression_params",
        "engines",
    ]
    for key in required_keys:
        if key not in config:
            raise KeyError(f"Missing required config key: '{key}' in {config_path}")

    return config


def load_tokens_map(config: dict, base_dir: Path) -> dict:
    """Read generated JSONL files to build a map of model -> req_id -> (prompt, completion)"""
    tokens_map = defaultdict(dict)

    for agent_type, models in config["generated"].items():
        for model_name, path_str in models.items():
            filepath = base_dir / path_str
            with open(filepath, "r") as f:
                for line in f:
                    data = json.loads(line)
                    tokens_map[model_name][str(data["id"])] = (
                        data["prompt_tokens"],
                        data["completion_tokens"],
                    )

    return tokens_map


def load_requests(config: dict, arrival_interval_sec: float = 0.5) -> list[Request]:
    """
    Loads arriving Request IDs from the huggingface datasets specified in config.
    """
    requests = []

    # Load Coding requests
    coding_ds_path = config["agents"]["coding"]["dataset"]
    coding_ds = load_from_disk(coding_ds_path)
    for row in coding_ds:
        req = Request(id=str(row["id"]), agent_id=AgentId.CODING)
        requests.append(req)

    # Load RAG requests
    rag_ds_path = config["agents"]["rag"]["dataset"]
    rag_ds = load_from_disk(rag_ds_path)
    for row in rag_ds:
        req = Request(id=str(row["id"]), agent_id=AgentId.RAG)
        requests.append(req)

    import random

    random.seed(42)
    random.shuffle(requests)

    for i, req in enumerate(requests):
        req.arrival_time = i * arrival_interval_sec

    return requests


def load_max_batched_tokens(config: dict, base_dir: Path) -> dict:
    mapping = {}
    for model_name, path_str in config["vllm_config"].items():
        filepath = base_dir / path_str
        with open(filepath, "r") as f:
            data = yaml.safe_load(f)
            if "max_num_batched_tokens" not in data:
                raise ValueError(
                    f"Missing 'max_num_batched_tokens' in {filepath} for model {model_name}"
                )
            mapping[model_name] = data["max_num_batched_tokens"]
    return mapping


def load_regression_params(config: dict, base_dir: Path) -> dict:
    """Read inline regression parameters keyed by action, model, and mig."""
    return config["regression_params"]


def main():
    base_dir = Path("/home/yclo/hpc")
    profiling_dir = base_dir / "profiling_results"
    configs_dir = base_dir / "configs"

    print("Loading config...")
    config = load_config(configs_dir / "simulation_config.yaml")

    print("Loading datasets...")
    requests = load_requests(config)
    print(f"Loaded {len(requests)} requests.")

    print("Building token mapping...")
    tokens_map = load_tokens_map(config, base_dir)

    print("Loading regression parameters and configs...")
    params = load_regression_params(config, base_dir)
    batched_tokens_map = load_max_batched_tokens(config, base_dir)

    print("Initializing System Architecture...")
    # Coding agent: A100 (4,2,1) -> (4g.20gb, 2g.10gb, 1g.10gb) -> (7B, 3B, 3B)
    # RAG agent: A100 7g.40gb -> 14B

    coding_agent = Agent(AgentId.CODING)
    rag_agent = Agent(AgentId.RAG)

    engines = {}
    for eng_conf in config["engines"]:
        eid = eng_conf["id"]
        mname = eng_conf["model"]
        mig = eng_conf["mig"]
        size = eng_conf["size"]
        prefill = params["prefill"][mname][mig]
        tpot = params["tpot"][mname][mig]
        max_batch = batched_tokens_map[mname]

        eng = LLMEngine(
            engine_id=eid,
            model_name=mname,
            mig_profile=mig,
            max_num_batched_tokens=max_batch,
            prefill_params=prefill,
            tpot_params=tpot,
        )
        eng.supported_size = size
        engines[eid] = eng

        if eng_conf["agent"] == "coding":
            coding_agent.add_engine(eng)
        else:
            rag_agent.add_engine(eng)

    print("Initializing Simulator...")
    sim = Simulator(
        agents={AgentId.CODING: coding_agent, AgentId.RAG: rag_agent},
        engines=engines,
        tokens_map=tokens_map,
    )

    print("Feeding events...")
    # For a quicker test we limit requests
    sim.add_arrival_events(requests[:50000])

    print("Running simulation...")
    sim.run()

    print("\nSimulation Finished!")
    print(f"Total simulated time: {sim.current_time:.2f} seconds")

    for eid, eng in engines.items():
        print(f"Engine {eid} completed {len(eng.completed_requests)} requests.")
        if eng.completed_requests:
            avg_req_latency = sum(
                r.finish_time - r.arrival_time for r in eng.completed_requests
            ) / len(eng.completed_requests)
            valid_ttfts = [
                r.first_token_time - r.arrival_time
                for r in eng.completed_requests
                if r.first_token_time is not None
            ]
            avg_ttft = sum(valid_ttfts) / len(valid_ttfts) if valid_ttfts else 0.0

            print(f"  Avg latency: {avg_req_latency:.4f}s")
            print(f"  Avg TTFT:    {avg_ttft:.4f}s")


if __name__ == "__main__":
    main()
