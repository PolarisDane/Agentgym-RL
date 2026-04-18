#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_SH="${CONDA_SH:-/opt/conda/etc/profile.d/conda.sh}"
ALFWORLD_ENV="${ALFWORLD_ENV:-/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/cy/conda_envs/agentenv-alfworld}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-36001}"
LOG_PATH="${LOG_PATH:-}"

export HF_HUB_OFFLINE=1
export WANDB_MODE=offline

if [[ -n "${LOG_PATH}" ]]; then
  mkdir -p "$(dirname "${LOG_PATH}")"
  exec >"${LOG_PATH}" 2>&1
fi

source "${CONDA_SH}"
set +u
conda activate "${ALFWORLD_ENV}"
set -u

export ALFWORLD_DATA="${ALFWORLD_DATA:-/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/ziyu/.cache/alfworld}"
export NO_PROXY="${NO_PROXY:-127.0.0.1,localhost}"
export no_proxy="${no_proxy:-127.0.0.1,localhost}"

cd "${ROOT}"
exec env \
  -u http_proxy -u https_proxy -u all_proxy \
  -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
  NO_PROXY="${NO_PROXY}" \
  no_proxy="${no_proxy}" \
  ALFWORLD_DATA="${ALFWORLD_DATA}" \
  alfworld --host "${HOST}" --port "${PORT}"
