#!/bin/bash

MODEL_ID="Qwen/Qwen2.5-Coder-14B-Instruct"
PORT=8014

vllm bench serve \
--model ${MODEL_ID} \
--dataset-name random \
--random-input-len 1024 \
--random-output-len 1 \
--random-range-ratio 0.9 \
--num-prompts 1000 \
--request-rate 0.4 \
--save-result --save-detailed \
--label "profile_prefill_${MODEL_ID}" \
--port ${PORT}
