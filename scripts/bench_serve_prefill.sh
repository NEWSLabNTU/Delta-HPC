#!/bin/bash

vllm bench serve \
--model Qwen/Qwen2.5-Coder-14B-Instruct \
--dataset-name random \
--random-input-len 1024 \
--random-output-len 1 \
--random-range-ratio 0.9 \
--num-prompts 1000 \
--request-rate 0.4 \
--save-result --save-detailed \
--label "profile_prefill_time_1024" \
--port 8014
