#!/bin/bash
# [Usage] ./script/bench_serve_prefill.sh [MODEL_NAME] [MIG_NAME] [PORT] [MODE]

MODEL_ID=$1  # e.g. "Qwen/Qwen2.5-Coder-14B-Instruct"
MODEL_NAME="${MODEL_ID##*/}"  # extract "Qwen2.5-Coder-14B-Instruct"
MIG_NAME=$2     # e.g. "MIG-7g-40gb"
PORT=$3         # e.g. 8014
MODE=$4         # rag | coder
TIMESTAMP=$(date +"%Y%m%d-%H%M%S")

INPUT_LEN=1024
NUM_PROMPT=1000
REQ_RATE=0.4

vllm bench serve \
    --model ${MODEL_ID} \
    --dataset-name random \
    --random-input-len ${INPUT_LEN} \
    --random-output-len 1 \
    --random-range-ratio 0.9 \
    --num-prompts ${NUM_PROMPT} \
    --request-rate ${REQ_RATE} \
    --port ${PORT} \
    --save-result \
    --save-detailed \
    --result-dir "profiling_results/prefill/${MODE}/${MIG_NAME}/bench_details" \
    --result-filename "prefill-${MODEL_NAME}-${MIG_NAME}-${TIMESTAMP}.json"
