import yaml
import json
from pathlib import Path

base_dir = Path("/home/yclo/hpc")
config_path = base_dir / "configs/simulation_config.yaml"

with open(config_path, "r") as f:
    config = yaml.safe_load(f)

for action in ["prefill", "tpot"]:
    for model_name, migs in config["regression_params"][action].items():
        for mig_profile, path_str in migs.items():
            if isinstance(path_str, dict):
                continue
            filepath = base_dir / path_str
            with open(filepath, "r") as f:
                data = json.load(f)
            
            # If tpot, rename keys and save back
            if action == "tpot":
                if "c_intercept_s" in data:
                    data["alpha"] = data.pop("c_intercept_s")
                if "d_slope_s" in data:
                    data["beta"] = data.pop("d_slope_s")
                
                with open(filepath, "w") as f:
                    json.dump(data, f, indent=4)
                    
            # In yaml config, store dictionary directly
            migs[mig_profile] = {
                "alpha": data.get("alpha", 0.0),
                "beta": data.get("beta", 0.0)
            }

with open(config_path, "w") as f:
    yaml.dump(config, f, default_flow_style=False, sort_keys=False)

print("Done updating params.")
