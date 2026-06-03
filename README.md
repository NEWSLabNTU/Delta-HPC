# Delta-HPC

**Delta-HPC** is a reinforcement learning framework for dynamically managing NVIDIA Multi-Instance GPU (MIG) resources to serve heterogeneous LLM workloads. It trains an RL agent (Maskable PPO) that learns to split, merge, and transfer MIG slices across GPUs in real time, optimising latency and throughput for concurrent Coding Agent and RAG Agent workloads. The project also includes a discrete-event simulator for training and evaluation, as well as tooling for profiling vLLM performance parameters and benchmarking policies both in simulation and on real hardware.

---

## Table of Contents

1. [Environment Preparation](#0-environment-preparation)
2. [Dataset Preparation](#1-dataset-preparation)
3. [Simulation Configuration](#2-simulation-configuration)
4. [Profiling](#3-profiling)
5. [RL Model Training](#4-rl-model-training)
6. [Benchmarking (Simulation)](#5-benchmarking-simulation)
7. [Benchmarking (Actual Deployment)](#6-benchmarking-actual-deployment)

---

## 0. Environment Preparation

### Prerequisites

| Requirement | Notes |
|---|---|
| **OS** | Ubuntu (22.04 or later recommended) |
| **GPU** | NVIDIA GPU(s) with MIG support (e.g. A100, H100, B200) |
| **MIG mode** | Must be **enabled manually** before using this repo |
| **Python** | ≥ 3.12 (managed via `uv`) |
| **Package manager** | [`uv`](https://github.com/astral-sh/uv) |
| **vLLM Docker image** | Custom build `vllm/vllm-openai:v0.17.0.custom` (see below) |

### Enable MIG Mode

MIG mode must be turned on for each target GPU **before** running any code in this repo. Replace `<GPU_INDEX>` with the actual GPU index (e.g. `0`, `1`, …):

```bash
sudo nvidia-smi -i <GPU_INDEX> -mig 1
# Verify
nvidia-smi -L
```

Repeat for every GPU you intend to manage. A system reboot or driver reload may be required on some machines.

### Python Environment

This project uses [`uv`](https://github.com/astral-sh/uv) as the package manager. Install it first if you haven't:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then create the virtual environment and install all dependencies (including the CUDA-enabled PyTorch build):

```bash
uv sync
```

All `just` recipes automatically prepend `.venv/bin` to `PATH`, so you can run them without explicitly activating the environment. To verify the environment is set up correctly:

```bash
just test-env
```

### vLLM Docker Image

The deployment module launches vLLM as Docker containers (one per MIG slice). The image name is configured in [`docker-compose.yaml`](docker-compose.yaml).

> **The vLLM image is not included in this repository.** You are expected to build or pull a vLLM image that is compatible with your own machine (CUDA version, driver, GPU architecture), and update the `image:` field in `docker-compose.yaml` accordingly.

```yaml
# docker-compose.yaml — change this line to match your image name
image: your-vllm-image-name:tag
```

Refer to the [vLLM documentation](https://docs.vllm.ai) for build instructions. This project targets **vLLM v0.17**.

---

## 1. Dataset Preparation

> **About agents.** This repository currently defines **two agents**: `CodingAgent` (serves coding-assistant LLM requests) and `RAGAgent` (serves retrieval-augmented generation requests). All dataset preparation, LLM configuration, profiling, and training are organised around these two agents.
>
> If you want to add more agents, the following files must be modified:
>
> | File | What to change |
> |---|---|
> | `src/share/models.py` | Add a new value to the `AgentId` enum |
> | `src/share/models.py` | Extend `EnvironmentStateData` with per-agent ratio fields if needed |
> | `src/share/models.py` | Add new `ResourceManagerAction` entries for the new GPU |
> | `src/simulation/environment_state.py` | Update observation construction for the new agent |
> | `src/training/config.py` | Add workload / request-rate configuration for the new agent |
> | `configs/simulation_config.yaml` | Add the new agent block under `simulation.agents` |
> | `configs/deployment.yaml` | Assign the new agent to a GPU slot |
>
> **Note:** Adding more than two agents has not been tested. Proceed with caution and expect to debug edge cases in the simulator and RL environment.

Datasets are stored under `assets/` and must be preprocessed before use. Two datasets are required: one for the **Coding Agent** workload and one for the **RAG Agent** workload.

### Coding Agent Dataset — Code Feedback

Download URL: [Crystalcareai/Code-feedback-sharegpt-renamed](https://huggingface.co/datasets/Crystalcareai/Code-feedback-sharegpt-renamed)

The raw dataset uses multi-round conversations. Preprocess it so that each conversation round becomes an independent row:

```bash
python -m src.dataset.coder_preprocess
# Output saved to: assets/processed_code_feedback/
```

### RAG Agent Dataset — RAG Dataset 12k

Download URL: [neural-bridge/rag-dataset-12000](https://huggingface.co/datasets/neural-bridge/rag-dataset-12000)

Convert the dataset (both train and test splits) to ShareGPT format:

```bash
python -m src.dataset.rag_convert_to_sharegpt \
    --hf-path assets/rag-dataset-sharegpt
# Output saved to: assets/rag-dataset-sharegpt/
```

---

## 2. Simulation Configuration

Before profiling, you must decide which LLMs will run on each MIG profile and prepare their configuration files. This is split across three types of files:

1. **`configs/<model_name>.yaml`** — vLLM server configuration for each LLM.
2. **`configs/gpus/<GPU_MODEL>.py`** — defines the MIG profile set for a GPU model (e.g. A100 40 GB, B200).
3. **`configs/simulation_config.yaml`** — maps models to MIG profiles and holds measured hardware parameters (filled in after profiling).

---

### 2.1 vLLM Model Config (`configs/<model_name>.yaml`)

Each LLM needs a YAML file consumed by the vLLM server. Create one file per model you want to serve, using the naming convention `configs/<model_name>.yaml` (underscores replace dots/hyphens as needed). The file is passed directly to vLLM via `--config`.

**Template:**

```yaml
# configs/qwen2_5-7b-instruct.yaml
model: "Qwen/Qwen2.5-7B-Instruct"      # HuggingFace model ID (or local path)
max_model_len: 32767                    # Maximum sequence length (tokens)
max_num_batched_tokens: 4096            # Max tokens processed per iteration
gpu_memory_utilization: 0.9            # Fraction of GPU memory vLLM may use
```

> Tune `max_model_len` and `max_num_batched_tokens` down on smaller MIG slices if vLLM fails to start due to insufficient KV cache.

---

### 2.2 GPU Profile Definition (`configs/gpus/<GPU_MODEL>.py`)

Each GPU model that appears in your cluster must have a corresponding Python file under `configs/gpus/` that enumerates its supported MIG profiles. The file name (without `.py`) is the key used in `simulation_config.yaml`.

**Example — A100 40 GB (`configs/gpus/A100_40GB.py`):**

```python
from src.share.models import MIGProfileBase, MIGProfile, ProfileInfo


class MIGProfileA100(MIGProfileBase):
    # ProfileInfo(compute_slices, memory_GB, logical_profile_type)
    MIG_7G_40GB = ProfileInfo(7, 40, MIGProfile.MIG_7G)
    MIG_4G_20GB = ProfileInfo(4, 20, MIGProfile.MIG_4G)
    MIG_3G_20GB = ProfileInfo(3, 20, MIGProfile.MIG_3G)
    MIG_2G_10GB = ProfileInfo(2, 10, MIGProfile.MIG_2G)
    MIG_1G_10GB = ProfileInfo(1, 10, MIGProfile.MIG_1G_LARGE)

    @property
    def gpu_model(self) -> str:
        return "A100_40GB"

    @classmethod
    def unsupported_profiles(cls):
        return []  # list any MIGProfile variants not supported on this GPU


MIG_PROFILE = MIGProfileA100
```

The `string` representation of each profile (e.g. `"2g.10gb"`) is derived automatically from the `ProfileInfo` compute-slice and memory values and is the key used throughout `simulation_config.yaml`.

---

### 2.3 Simulation Config (`configs/simulation_config.yaml`)

This is the central configuration file. The `model` section and the cluster/agent structure should be filled in now (before profiling). The measured parameters (`kv_cache_GB`, `restart_time`, and `param.*`) are filled in **after** running the profiling steps in chapter 3.

#### `datasets` — local dataset paths

This section is **required**. It tells the runtime where to find the preprocessed datasets on disk (the outputs from [§1 Dataset Preparation](#1-dataset-preparation)).

```yaml
datasets:
  CodingAgent: assets/processed_code_feedback   # Output of src.dataset.coder_preprocess
  RAGAgent: assets/rag-dataset-sharegpt         # Output of src.dataset.rag_convert_to_sharegpt
```

> [!IMPORTANT]
> The paths must match the `--hf-path` argument you used when running the dataset preprocessing scripts. If you changed the default output paths, update these values accordingly.

#### `model` — per-model global settings

```yaml
model:
  <model-name>:                          # Must match the key used in agents below
    generate_path: profiling_results/generated/<generated>.jsonl
                                         # Path to the JSONL file produced by profile-generate (chapter 3)
                                         # (used to replay realistic output lengths in simulation)
    kv_per_token_KB: <value>             # KV cache consumed per token, in kilobytes (see §2.3.1)
    vllm_config: configs/<model_name>.yaml  # Path to the vLLM YAML config above
```

#### `simulation` — cluster layout and per-agent, per-MIG parameters

```yaml
simulation:
  cluster:
    <gpu_index>: <GPU_MODEL>             # e.g. 0: A100_40GB
    ...

  initial_state:
    gpu_initial_agents:
      <gpu_index>:
        - <AgentName>                    # Which agent occupies this GPU initially (currently only support one GPU for one agent)
    permanent_engines:                   # Fixed engines not managed by the RL agent
      - gpu: <gpu_index>
        mig: <mig_profile_string>        # e.g. 2g.10gb
        agent: <AgentName>

  agents:
    <AgentName>:                         # e.g. CodingAgent, RAGAgent
      <GPU_MODEL>:                       # e.g. A100_40GB
        mig:
          <mig_profile_string>:          # e.g. 1g.10gb, 2g.10gb, 7g.40gb
            model: <model-name>          # Which LLM runs on this profile
            kv_cache_GB: <value>         # Measured available KV cache (see §2.3.2 — fill after profiling)
            restart_time: <seconds>      # Measured restart time (see §2.3.3 — fill after profiling)
            param:
              prefill:                   # Fill after running profile-prefill (chapter 3)
                alpha: <seconds>
                beta: <seconds_per_token>
                sigma: <seconds>
              tpot:                      # Fill after running profile-tpot (chapter 3)
                alpha: <seconds>
                beta: <seconds_per_request>
                sigma: <seconds>
```

---

#### 2.3.1 Estimating `kv_per_token_KB`

This value is the KV cache memory consumed per **output token** for a given model, in **kilobytes**. It depends on the model architecture, not on the MIG profile. Use the following formula:

```
kv_per_token_KB = 2 × num_layers × num_kv_heads × head_dim × bytes_per_element / 1024
```

Where:
- `2` accounts for both the K and V tensors
- `bytes_per_element` = 2 for FP16/BF16, 1 for FP8
- All values are available in the model's `config.json` on HuggingFace

**Example — Qwen2.5-7B-Instruct (BF16):**

| Parameter | Value |
|---|---|
| `num_hidden_layers` | 28 |
| `num_key_value_heads` | 4 |
| `head_dim` (`hidden_size / num_attention_heads`) | 128 |
| `bytes_per_element` | 2 (BF16) |

```
kv_per_token_KB = 2 × 28 × 4 × 128 × 2 / 1024 ≈ 56 KB/token
```

---

#### 2.3.2 Measuring `kv_cache_GB`

This is the amount of GPU memory actually available for the KV cache on a specific (model, MIG profile) combination *after* vLLM loads the model weights. Measure it from vLLM's startup logs after starting the model on the target MIG slice:

1. Start vLLM on the target MIG slice with the desired model config.
2. Search the startup logs for a line like:
   ```
   GPU KV cache size: 6.45 GiB
   ```
   Read the value directly from this line.

Alternatively, you can compute it from the `# GPU blocks` line:
```
# GPU blocks: 1234, # CPU blocks: 0
kv_cache_GB = gpu_blocks × block_size × kv_per_token_KB / 1024 / 1024
```
where `block_size` is vLLM's block size in tokens (default: **16**).

---

#### 2.3.3 Measuring `restart_time`

This is the wall-clock time (in seconds) for a vLLM container to shut down and restart on a given MIG slice. It is used by the simulator to model the downtime cost of MIG reconfigurations.

Measure it by timing a full down-then-up cycle using `scripts/launch_vllm.sh`:

```bash
# scripts/launch_vllm.sh <MIG_UUID> <MODEL_ID> <PORT> <up|down|logs>
time scripts/launch_vllm.sh <MIG_UUID> <MODEL_ID> <PORT> down \
  && time scripts/launch_vllm.sh <MIG_UUID> <MODEL_ID> <PORT> up
```

Record the total elapsed time. Typical values are in the range of 60–90 seconds depending on model size and MIG memory.

---

## 3. Profiling

With the LLM-to-MIG-profile assignments decided in chapter 2, this step measures the actual latency behaviour of each (model, MIG profile) combination and fits linear models for **prefill** (TTFT) and **decoding** (TPOT). The fitted parameters are then written back into `configs/simulation_config.yaml` under the appropriate `param` fields.

Profiling must be run for **every model listed under the `model:` section of `configs/simulation_config.yaml`**, once per MIG profile it is assigned to.

### Scripts

| Script | Entry point | Purpose |
|---|---|---|
| `generate.py` | `src.profiling.generate` | Query a live vLLM server with dataset prompts and save token usage + responses |
| `prefill.py` | `src.profiling.prefill` | Fit a linear model `TTFT = α + β·x + ε` to prefill latency data |
| `tpot.py` | `src.profiling.tpot` | Fit a linear model `ITL = α + β·N + ε` to decoding latency data across concurrency sweeps |

### Step 1 — Generate responses

> **Time Warning:** Generating offline responses for thousands of requests can take **several hours to days** depending on the dataset size, model size, and hardware speed.

Start a vLLM server externally (e.g. via Docker), then run:

```bash
just profile-generate <PORT> <MODEL_NAME> <DATASET_DIR> <OUTPUT_DIR>
# Example:
just profile-generate 8003 qwen2.5-3b-instruct assets/rag-dataset-sharegpt profiling_results/generated
```

### Step 2 — Extract prefill parameters

The input is a `benchmark_detailed_results.json` file produced by a vLLM benchmark tool (e.g. `vllm benchmark_serving` with `--save-detailed-results`):

```bash
just profile-prefill <INPUT_JSON> <OUTPUT_DIR>
# Example:
just profile-prefill profiling_results/raw/prefill-1g.10gb.json profiling_results/prefill
```

### Step 3 — Extract TPOT parameters

Run a concurrency sweep and collect one JSON per concurrency level. Then:

```bash
just profile-tpot <INPUT_DIR> <OUTPUT_DIR>
# Example:
just profile-tpot profiling_results/raw/tpot-1g.10gb/ profiling_results/tpot
```

### Output Directory Structure

After running the profiling pipeline, the `profiling_results/` directory will look like:

```
profiling_results/
├── generated/                         # Step 1 outputs
│   └── <model>-port-<port>-<dataset>-generated.jsonl
├── prefill/                           # Step 2 outputs
│   ├── <benchmark_name>-param.json    # Fitted α, β, σ parameters
│   └── <benchmark_name>-plot.png      # Scatter plot with regression line
└── tpot/                              # Step 3 outputs
    ├── <benchmark_name>-param.json    # Fitted α, β, σ parameters
    └── <benchmark_name>-plot.png      # Scatter plot with regression line
```

Each `-param.json` file contains the following fields:

```json
{
    "alpha": 0.03,
    "beta": 0.00011,
    "sigma": 0.029,
    "r_squared": 0.97,
    "unit_alpha": "seconds",
    "unit_beta": "seconds_per_token",
    "model_formula": "y = beta * x + alpha + N(0, sigma^2)"
}
```

Then copy the parameter values into `configs/simulation_config.yaml` under the `param` fields of the appropriate MIG profile entry (see §2.3).

---

## 4. RL Model Training

> **Time Warning:** Reinforcement learning requires simulating hundreds of thousands of environment steps. A full training run from scratch may take **several days**. Consider running it in the background using `tmux` or `screen`.

The RL agent is trained inside a discrete-event simulator. The main training code is in `src/training/` and the simulator logic is in `src/simulation/`.

### Overview

- **Algorithm**: Maskable PPO (`sb3-contrib`) — action masking prevents the agent from selecting physically impossible MIG reconfigurations.
- **Environment**: `TrainingMIGResourceEnv` wraps the simulator and drives it step-by-step, generating synthetic LLM request workloads.
- **Simulator** (`src/simulation/`): A discrete-event engine that models vLLM engines on MIG slices, request queueing, prefill/decoding latency (using the profiled parameters), and MIG reconfiguration overheads.
- **Config**: Edit `configs/training_config.yaml` to adjust hyperparameters, episode length, reward shaping, etc.
- **Cluster**: The `training.cluster` field in `configs/training_config.yaml` specifies which GPU model to simulate (e.g. `A100_40GB`). Note that this cluster configuration will be copied to `configs/simulation_config.yaml` automatically when training starts (via `src/training/train.py`). (A similar sync occurs from `bench_config.yaml` or a snapshot during benchmarking via `src/bench/main.py`).

### Train from Scratch

Set the GPU index in the justfile (edit `gpu := ""` to e.g. `gpu := "0"`) then run:

```bash
just train
```

### Resume from Checkpoint

```bash
just train <PATH_TO_CHECKPOINT.zip>
# Example:
just train results/20250501-120000-000/ckpts/20250501-120000-000/ppo_mig_resource_manager_5120_steps.zip
```

### Grid Search

To sweep over hyperparameter configurations defined in a YAML file:

```bash
just grid-search configs/grid_search.yaml
```

### Output Directory Structure

Each training run produces a timestamped directory under `results/`:

```
results/
└── <run_id>/                          # e.g. 20250501-120000-000
    ├── ckpts/
    │   └── <run_id>/
    │       ├── ppo_mig_resource_manager_<N>_steps.zip        # Periodic checkpoints
    │       ├── ppo_mig_resource_manager_<N>_steps_vecnormalize.pkl
    │       ├── ppo_mig_resource_manager.zip                  # Final model
    │       └── ppo_mig_resource_manager_vecnormalize.pkl     # Final VecNormalize stats
    ├── logs/
    │   └── train/                     # Per-episode step logs (JSONL)
    ├── snapshots/
    │   └── training_config.yaml       # Snapshot of the config used for this run
    └── tboards/
        └── <run_id>/                  # TensorBoard event files
```

Monitor training progress:

```bash
tensorboard --logdir results
```

---

## 5. Benchmarking (Simulation)

Evaluate one or more RL checkpoints (and optional baselines) in the simulator. The benchmarking code is in `src/bench/`.

### Available Policies

| Policy / flag | Description |
|---|---|
| `--ckpt <path>` | RL checkpoint(s) to evaluate |
| `--bl static_no_mig` | Baseline: single 7G instance per GPU (no MIG splitting) |
| `--bl static_split_extreme` | Baseline: maximum MIG splitting |
| `--bl heuristic` | Baseline: rule-based heuristic agent |
| `--bl all` | Run all three baselines |

### Run

```bash
# Evaluate a specific checkpoint
just bench results/<run_id>/ckpts/<run_id>/ppo_mig_resource_manager.zip

# Evaluate multiple checkpoints
just bench <ckpt1.zip> <ckpt2.zip>

# Evaluate baselines only
just bench-bl all

# Evaluate latest checkpoints + all baselines (uses scripts/get_latest_ckpts.py)
just bench-all
```

### Output Directory Structure

Results are written under the corresponding `results/<run_id>/bench/` directory:

```
results/
└── <run_id>/
    └── bench/
        ├── results_<run_name>.txt     # Printed metrics table (TTFT, TPOT, queue length, MIG usage, …)
        └── figs/
            └── <run_name>/
                ├── split.png          # Workload timeline annotated with Split events
                ├── merge.png          # Workload timeline annotated with Merge events
                └── transfer.png       # Workload timeline annotated with Transfer events
```

---

## 6. Benchmarking (Actual Deployment)

Deploy and benchmark a policy on real hardware. The deployment code is in `src/deploy/`. This module:

1. Configures physical MIG instances on the GPUs.
2. Launches one vLLM Docker container per MIG slice (via `docker-compose`).
3. Dispatches real LLM requests according to a configurable workload pattern.
4. Runs the policy agent (RL, heuristic, or static) in a control loop.
5. Exposes a live web dashboard for monitoring.

> **Note**: This requires `sudo` because MIG reconfiguration (via `nvidia-smi`) needs root privileges. The justfile recipe uses `sudo .venv/bin/python3` to avoid root-owned file artefacts while still having the required permissions.

### Configuration

Edit `configs/deployment.yaml` to declare which GPUs are managed and which are reserved for permanent engines. Edit `configs/bench_config.yaml` to configure the workload pattern.

### Run

```bash
# Run with an RL checkpoint
just deploy-bench <PATH_TO_CHECKPOINT.zip> <DURATION_SECONDS>
# Example:
just deploy-bench results/20250501-120000-000/ckpts/20250501-120000-000/ppo_mig_resource_manager.zip $((12*60*60))

# Run with the rule-based heuristic
just deploy-bench heuristic $((12*60*60))

# Run with a static single-instance (no MIG) baseline
just deploy-bench static-7g $((12*60*60))

# Run with a static maximally-split baseline
just deploy-bench static-2g $((12*60*60))
```

Adjust the logging verbosity with the optional third argument (default `INFO`):

```bash
just deploy-bench heuristic $((12*60*60)) DEBUG
```

### Cleanup

After a run (or after a crash), clean up any lingering Docker containers and MIG instances:

```bash
just clean-deploy
```

### Dashboard

While a deployment benchmark is running, a live web dashboard is served (host and port configured in `configs/deployment.yaml`). Use the `watch` command to monitor the dashboard:

```bash
watch -n1 "curl -s http://localhost:9000"  # Dashboard port is configurable in configs/deployment.yaml
```

---

## 7. Developer & Utility Commands

There are a few extra `just` recipes included for development and debugging:

### `mock-train`
Runs a mock version of the simulator with a random action policy and no Stable Baselines overhead (`python -m src.simulation.main`). This is useful to rapidly debug the discrete-event logic or verify that metrics are tracked correctly before kicking off a multi-day training run.

```bash
just mock-train
```

### `clean-logs`
Deletes all generated log files and output from mock runs.

```bash
just clean-logs
```

### `lint`
Runs `ruff` over the `src/` directory to automatically fix and format python code.

```bash
just lint
```
