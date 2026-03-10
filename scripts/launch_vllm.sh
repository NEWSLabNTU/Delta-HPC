#!/bin/bash
# Usage: ./launch.sh <MIG_UUID> <MODEL_ID> <PORT> <up|down|logs>

set -e

if [ "$#" -ne 4 ]; then
  echo "Usage: $0 <MIG_UUID> <MODEL_ID> <PORT> <up|down|logs>"
  exit 1
fi

MIG_UUID="$1"
MODEL_ID="$2"
HOST_PORT="$3"
ACTION="$4"

export MIG_UUID
export MODEL_ID
export HOST_PORT
export MIG_SHORT="${MIG_UUID:4:4}"

PROJECT_NAME="vllm-${MODEL_ID}-${MIG_SHORT}"

case "$ACTION" in
  up)
    docker compose -p "$PROJECT_NAME" up -d
    ;;
  down)
    docker compose -p "$PROJECT_NAME" down
    ;;
  logs)
    docker compose -p "$PROJECT_NAME" logs -f -t --no-log-prefix -n 30
    ;;
  *)
    echo "Invalid action: $ACTION"
    exit 1
    ;;
esac

