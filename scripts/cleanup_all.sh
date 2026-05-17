#!/usr/bin/env bash
# scripts/cleanup_all.sh
# Standalone cleanup script for vLLM containers and MIG instances.
# No dependencies on the Delta-HPC python codebase.

set -eo pipefail

echo "--------------------------------------------------------"
echo "Starting Standalone Deployment Cleanup"
echo "--------------------------------------------------------"

# 1. Kill vLLM Docker projects via docker compose down
echo "[1/2] Cleaning up vLLM containers via docker compose..."

# Get project names starting with vllm-
# We use 'docker compose ls' to find projects managed by compose
PROJECTS=$(docker compose ls --format json | grep -oP '"Name":"vllm-[^"]+"' | cut -d'"' -f4 || true)

if [ -n "$PROJECTS" ]; then
    echo "Found projects: $PROJECTS"
    for proj in $PROJECTS; do
        echo "Shutting down project: $proj"
        # We assume the docker-compose.yaml is in the project root (where this script is likely called from)
        # or we just rely on the project name if it's already registered.
        # launch_vllm.sh uses the local docker-compose.yaml
        docker compose -p "$proj" down || true
    done
    echo "All vLLM projects shut down."
else
    # Fallback: Check for orphaned containers starting with vllm-
    echo "No active compose projects found. Checking for orphaned vllm- containers..."
    ORPHANS=$(docker ps -a --format '{{.Names}}' | grep '^vllm-' || true)
    if [ -n "$ORPHANS" ]; then
        echo "Removing orphaned containers: $ORPHANS"
        docker stop $ORPHANS >/dev/null 2>&1 || true
        docker rm $ORPHANS >/dev/null 2>&1 || true
    else
        echo "No vLLM containers found."
    fi
fi

# 2. Destroy MIG instances using nvidia-smi
echo "[2/2] Destroying all MIG instances..."

# Get all GPU indices that have MIG mode enabled
GPU_INDICES=$(nvidia-smi --query-gpu=index,mig.mode.current --format=csv,noheader,nounits | grep 'Enabled' | cut -d',' -f1 || true)

if [ -n "$GPU_INDICES" ]; then
    for idx in $GPU_INDICES; do
        echo "GPU $idx: Destroying all GPU instances (MIG partitions)..."
        # -dgi destroys all GPU instances on the specified device
        if sudo nvidia-smi mig -dci -ci 0 && sudo nvidia-smi mig -dgi; then
            echo "GPU $idx: Cleanup successful."
        else
            echo "GPU $idx: Cleanup failed (check if processes are still using the MIG devices)."
        fi
    done
else
    echo "No MIG-enabled GPUs detected."
fi

echo "--------------------------------------------------------"
echo "Cleanup Finished."
echo "--------------------------------------------------------"
