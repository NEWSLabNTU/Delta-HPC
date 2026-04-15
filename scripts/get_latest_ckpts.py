import sys
import re
from pathlib import Path
from typing import Dict, Tuple


def get_latest_checkpoints(phase_num: str):
    base_path = Path("ckpts")
    # Pattern to match the specific phase directory
    pattern = f"*_phase_{phase_num}/rl_model_*_steps.zip"

    latest_map: Dict[str, Tuple[int, Path]] = {}

    for p in base_path.glob(pattern):
        # Extract step number using regex: rl_model_5120_steps.zip -> 5120
        match = re.search(r"rl_model_(\d+)_steps", p.name)
        if match:
            steps = int(match.group(1))
            run_id = p.parent.name

            # Group by run_id, keeping only the one with the maximum steps
            if run_id not in latest_map or steps > latest_map[run_id][0]:
                latest_map[run_id] = (steps, p)

    # Sort by the running_id (the directory name) to maintain version order
    sorted_paths = [str(latest_map[k][1]) for k in sorted(latest_map.keys())]
    return " ".join(sorted_paths)


if __name__ == "__main__":
    # Default to phase 2 if no argument is provided
    target_phase = sys.argv[1] if len(sys.argv) > 1 else "2"
    print(get_latest_checkpoints(target_phase), end=" ")
