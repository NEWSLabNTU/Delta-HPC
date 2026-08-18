# Scripts

| Script | Description | Used by |
| --- | --- | --- |
| `bench_serve_prefill.sh` | Runs a `vllm bench serve` prefill sweep (1024-token inputs, 1-token outputs) against a served model and saves results under `profiling_results/prefill/`. | — |
| `bench_serve_tpot.sh` | Sweeps client concurrency from 1 to 30 with decode-heavy requests (512-token outputs) to profile time-per-output-token, saving one result file per concurrency level. | — |
| `cleanup_all.sh` | Tears down all `vllm-*` Docker Compose projects (or orphaned containers) and destroys every MIG instance on MIG-enabled GPUs. | `just clean-deploy` |
| `count_bench_requests.py` | Replays `RequestLoader.generate_requests` from the config files alone to report how many requests `just bench` will generate per agent, without loading any dataset or model. | — |
| `get_latest_ckpts.py` | Prints a space-separated list of the highest-step checkpoint (`rl_model_*_steps.zip`) for each run under `results/`. | `just bench-all` |
| `launch_vllm.sh` | Brings a vLLM container `up`, `down`, or tails its `logs` for a given MIG UUID, model ID, and host port via Docker Compose. | — |
| `mps.sh` | Starts the NVIDIA CUDA MPS control daemon. | — |
