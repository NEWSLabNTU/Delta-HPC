# Activate venv

export PATH := ".venv/bin:" + env("PATH")

# Replace with your GPU id

gpu := "MIG-9ac89393-4759-5270-8347-c8e1b56d4df7"

test-env:
    which python

bench +ckpts:
    python -m src.bench.main --ckpt {{ ckpts }}

bench-bl +bl:
    python -m src.bench.main --bl {{ bl }}

bench-all phase="2":
    #!/usr/bin/env bash
    set -euo pipefail
    ckpts=$(python scripts/get_latest_ckpts.py {{ phase }})
    set -x
    python -m src.bench.main --bl --ckpt $ckpts

mock-train phase="2":
    python -m src.simulation.main --phase {{ phase }} > test.log

train ckpt="":
    CUDA_VISIBLE_DEVICES={{ gpu }} python -m src.training.train {{ if ckpt != "" { "--ckpt " + ckpt } else { "" } }}

grid-search conf:
    CUDA_VISIBLE_DEVICES={{ gpu }} python -m src.training.grid_search {{ conf }}

clean:
    rm logs/*.jsonl logs/*.log test.log

lint:
    ruff check src --fix && ruff format src
