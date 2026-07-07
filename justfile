# Activate venv

export PATH := ".venv/bin:" + env("PATH")

# Replace with your GPU id

gpu := "MIG-9ac89393-4759-5270-8347-c8e1b56d4df7"

test-env:
    which python

# Profiling: extract prefill latency parameters (alpha, beta, sigma) from a benchmark JSON
profile-prefill input output_dir:
    python -m src.profiling.prefill --input {{ input }} --output-dir {{ output_dir }}

# Profiling: extract TPOT (decoding) parameters from a directory of concurrency-sweep JSONs
profile-tpot input_dir output_dir:
    python -m src.profiling.tpot --input-dir {{ input_dir }} --output-dir {{ output_dir }}

# Profiling: generate model responses from a dataset via a running vLLM server
profile-generate port model dataset_dir output_dir:
    python -m src.profiling.generate --port {{ port }} --model {{ model }} --dataset-dir {{ dataset_dir }} --output-dir {{ output_dir }}

bench +ckpts:
    python -m src.bench.main --ckpt {{ ckpts }}

bench-bl +bl:
    python -m src.bench.main --bl {{ bl }}

bench-all:
    #!/usr/bin/env bash
    set -euo pipefail
    ckpts=$(python scripts/get_latest_ckpts.py)
    set -x
    python -m src.bench.main --bl all --ckpt $ckpts

deploy-bench policy duration_s log_level="INFO":
    sudo .venv/bin/python3 -m src.deploy.main --policy {{ policy }} --duration {{ duration_s }} --log-level {{ log_level }}

mock-train:
    TRAINING_RUN_ID=$(date +%Y%m%d-%H%M%S-000) python -m src.simulation.main > test.log

train ckpt="":
    CUDA_VISIBLE_DEVICES={{ gpu }} python -m src.training.train {{ if ckpt != "" { "--ckpt " + ckpt } else { "" } }}

grid-search conf:
    CUDA_VISIBLE_DEVICES={{ gpu }} python -m src.training.grid_search {{ conf }}

clean-logs:
    rm -f logs/*.jsonl logs/*.log test.log

clean-deploy:
    ./scripts/cleanup_all.sh

lint:
    ruff check src --fix && ruff format src
