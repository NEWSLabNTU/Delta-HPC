#!/bin/bash
# [Usage] ./script/bench_serve_prefill.sh [MODEL_NAME] [MIG_NAME] [PORT] 

MODEL_ORG="Qwen"
MODEL_NAME=$1   # e.g. "Qwen2.5-Coder-14B-Instruct"
MODEL_ID="${MODEL_ORG}/${MODEL_NAME}"
MIG_NAME=$2     # e.g. "MIG-7g-40gb"
PORT=$3         # e.g. 8014
TIMESTAMP=$(date +"%Y%m%d-%H%M%S")

vllm bench serve \
--model ${MODEL_ID} \
--dataset-name random \
--random-input-len 1024 \
--random-output-len 1 \
--random-range-ratio 0.9 \
--num-prompts 1000 \
--request-rate 0.4 \
--port ${PORT} \
--save-result \
--save-detailed \
--result-dir "profiling_results/prefill/coder/${MIG_NAME}/bench_details" \
--result-filename "prefill-${MODEL_NAME}-${MIG_NAME}-${TIMESTAMP}.json"
