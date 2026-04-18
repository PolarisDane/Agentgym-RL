#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_PORT="${ENV_PORT:-36001}"
ENV_ADDR="http://127.0.0.1:${ENV_PORT}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
MODEL_PATH="${MODEL_PATH:-/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/ziyu/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct}"
WANDB_MODE="${WANDB_MODE:-offline}"
PROJECT_NAME="${PROJECT_NAME:-agentgym-alfworld}"

export HF_HUB_OFFLINE=1
export WANDB_MODE=offline

RUN_TS="$(date -u +%Y%m%d_%H%M%S)"
EXP_NAME="${EXP_NAME:-alfworld_grpo_qwen2.5_3b_wm_clip_${RUN_TS}}"

ENV_SESSION="alfworld_env_${ENV_PORT}"
TRAIN_SESSION="alfworld_grpo_train"
ENV_LOG="${ROOT}/runlogs/${ENV_SESSION}_${RUN_TS}.log"
TRAIN_LOG="${ROOT}/runlogs/${EXP_NAME}/train.log"

mkdir -p "${ROOT}/runlogs/${EXP_NAME}"

if tmux has-session -t "${ENV_SESSION}" 2>/dev/null; then
  tmux kill-session -t "${ENV_SESSION}"
fi
if tmux has-session -t "${TRAIN_SESSION}" 2>/dev/null; then
  tmux kill-session -t "${TRAIN_SESSION}"
fi

echo "Starting AlfWorld Environment Service..."
tmux new-session -d -s "${ENV_SESSION}" \
  "cd ${ROOT} && HOST=127.0.0.1 PORT=${ENV_PORT} LOG_PATH=${ENV_LOG} bash ${ROOT}/scripts/run_alfworld_env_service.sh"

echo "Waiting for service to become healthy at ${ENV_ADDR}..."
for _ in $(seq 1 60); do
  if curl --noproxy '*' -sf "${ENV_ADDR}/" >/dev/null; then
    break
  fi
  sleep 2
done

if ! curl --noproxy '*' -sf "${ENV_ADDR}/" >/dev/null; then
  echo "AlfWorld service failed to start."
  exit 1
fi

echo "Starting GRPO Training..."
tmux new-session -d -s "${TRAIN_SESSION}" \
  "cd ${ROOT} && CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} ENV_ADDR=${ENV_ADDR} MODEL_PATH=${MODEL_PATH} WANDB_MODE=${WANDB_MODE} PROJECT_NAME=${PROJECT_NAME} EXP_NAME=${EXP_NAME} LOG_PATH=${TRAIN_LOG} bash ${ROOT}/scripts/run_alfworld_grpo_train.sh"

echo "--------------------------------------------------"
echo "AlfWorld Training Launched!"
echo "Environment Session: ${ENV_SESSION}"
echo "Training Session:    ${TRAIN_SESSION}"
echo "Training Log:        ${TRAIN_LOG}"
echo "--------------------------------------------------"
