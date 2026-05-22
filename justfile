# Activate venv

export PATH := ".venv/bin:" + env("PATH")

# Replace with your GPU id

gpu := "1"

test-env:
    which python

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

deploy-bench ckpt duration_s:
    sudo .venv/bin/python3 -m src.deploy.main --ckpt {{ ckpt }} --duration {{ duration_s }}

mock-train:
    python -m src.simulation.main > test.log

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
