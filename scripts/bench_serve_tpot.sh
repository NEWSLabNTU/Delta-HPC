#!/bin/bash

# Configuration
RESULT_DIR=$1
if [[ -z $RESULT_DIR ]]; then
    echo "[Usage] ./script/bench_serve_tpot.sh [RESULT_DIR]"
    exit 1
fi
if [[ ! -d $RESULT_DIR ]]; then
    mkdir -p $RESULT_DIR
fi

MODEL="Qwen/Qwen2.5-Coder-14B-Instruct"
PORT=8014

echo "Starting TPOT Profiling Sweep"

# We iterate concurrency (N) from 1 to 30
for N in {1..30}
do
    # Calculate num_requests: ~~
    # Logic: Lower concurrency has fewer prompts, higher has more.
    # N=1  -> ~20 prompts
    # N=30 -> ~80 prompts
    # Linear interpolation: num = 20 + ((N-1) * (60/29))
    NUM_PROMPTS=$(( 20 + ((N-1) * 60 / 29) ))
    
    LABEL="profile_tpot_concurrency_$N"
    RESULT_FILENAME="$LABEL-$MODEL_NAME.json"
    
    echo "-------------------------------------------------------"
    echo "Target Concurrency: $N | Total Prompts: $NUM_PROMPTS"
    echo "-------------------------------------------------------"

    # forces the client to maintain exactly N requests in flight.
    vllm bench serve \
        --model "$MODEL" \
        --dataset-name random \
        --random-input-len 2 \
        --random-output-len 512 \
        --random-range-ratio 0.3 \
        --num-prompts "$NUM_PROMPTS" \
        --request-rate inf \
        --max-concurrency "$N" \
        --save-result \
        --save-detailed \
        --result-dir "$RESULT_DIR" \
        --result-filename "$RESULT_FILENAME" \
        --label "$LABEL" \
        --port "$PORT"

    # Brief sleep to allow the engine to settle before the next run
    sleep 2
done

echo "Sweep complete. Results saved with label profile_tpot_*"
