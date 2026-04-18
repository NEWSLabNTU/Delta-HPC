import yaml
import copy
import argparse
from pathlib import Path
from datetime import datetime
import itertools
import time
import subprocess
from typing import Dict, Iterator, List, Mapping, Set, Tuple, Any, Union, cast

type Tree = Union[Mapping[str, "Tree"], Any]


def load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_yaml(data: Dict[str, Any], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)


def dict_get(d: Dict[str, Any], keys: Tuple[str, ...]):
    v = d
    for k in keys:
        v = v[k]
    return v


def dict_set(d: Dict[str, Any], keys: Tuple[str, ...], value: Any):
    v = d
    for k in keys[:-1]:
        v = v[k]
    v[keys[-1]] = value


def iter_leaves(
    d: Tree, current_path: Tuple[str, ...] = ()
) -> Iterator[tuple[tuple[str, ...], Any]]:
    if isinstance(d, Mapping):
        d = cast(Mapping[str, Tree], d)
        for k, v in d.items():
            yield from iter_leaves(v, current_path + (k,))
    else:
        yield current_path, d


def main():
    parser = argparse.ArgumentParser(description="Grid search over training config.")
    parser.add_argument(
        "grid_config", type=Path, help="Path to grid search config yaml"
    )
    parser.add_argument(
        "--base-config",
        type=Path,
        default=Path("configs/training_config.yaml"),
        help="Path to base config yaml",
    )
    args = parser.parse_args()

    base_config = load_yaml(args.base_config)
    grid_config = load_yaml(args.grid_config)

    grid_params: Dict[Tuple[str, ...], Any] = {}

    # Validate keys in grid config exist in base config
    # Ensure exact tree format
    for path, values in iter_leaves(grid_config):
        # We expect paths in grid config to correspond to valid paths in base config
        try:
            dict_get(base_config, path)
        except KeyError:
            raise KeyError(
                f"Key path {path} found in grid config but not in base config."
            )

        if not isinstance(values, list):
            raise ValueError(
                f"Value at path {path} in grid config must be a list of values to search over, got {type(values)}"
            )

        grid_params[path] = values

    if not grid_params:
        print("No grid search parameters found in config.")
        return

    paths = list(grid_params.keys())
    value_lists = [grid_params[p] for p in paths]
    combinations = list(itertools.product(*value_lists))

    print(f"Found {len(combinations)} combinations to train.")

    session_name = base_config.get("training", {}).get("tmux_session", "hpc")

    # Ensure tmux session exists
    session_check = subprocess.run(
        ["tmux", "has-session", "-t", session_name], stderr=subprocess.DEVNULL
    )
    if session_check.returncode != 0:
        print(f"Creating tmux session '{session_name}'")
        subprocess.run(["tmux", "new-session", "-d", "-s", session_name])

    seen_ids: Set[str] = set()

    for idx, combo in enumerate(combinations):
        config_copy = copy.deepcopy(base_config)
        for path_idx, path in enumerate(paths):
            dict_set(config_copy, path, combo[path_idx])

        # Generate unique TIMESTAMP id
        while True:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3]
            if timestamp not in seen_ids:
                seen_ids.add(timestamp)
                break
            time.sleep(0.001)

        snapshot_path = Path(f"snapshots/{timestamp}/training_config.yaml")
        save_yaml(config_copy, snapshot_path)

        cmd_str = (
            f"cd /home/yclo/Delta-HPC && source .venv/bin/activate && "
            f"export TRAINING_CONFIG_PATH='{snapshot_path}' && "
            f"export TRAINING_RUN_ID='{timestamp}' && "
            f"python -m src.training.train; bash"
        )

        tmux_cmd: List[str] = [
            "tmux",
            "new-window",
            "-t",
            session_name,
            "-n",
            timestamp,
            cmd_str,
        ]
        print(
            f"[{idx + 1}/{len(combinations)}] Launching training with ID {timestamp} in session {session_name}"
        )
        subprocess.run(tmux_cmd, check=True)


if __name__ == "__main__":
    main()
