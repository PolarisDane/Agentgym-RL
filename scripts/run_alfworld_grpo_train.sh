#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAIN_CODE_DIR="${ROOT}/AgentGym-RL"
CONDA_SH="${CONDA_SH:-/opt/conda/etc/profile.d/conda.sh}"
TRAIN_ENV="${TRAIN_ENV:-/mnt/tidal-alsh-share2/usr/wangshanyong/conda_envs/agentgym-rl-h20}"
MODEL_PATH="${MODEL_PATH:-/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/ziyu/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct}"
TASK_NAME="alfworld"

export HF_HUB_OFFLINE=1

ENV_ADDR_HOST="${ENV_ADDR_HOST:-127.0.0.1}"
BASE_PORT="${BASE_PORT:-36001}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
IFS=',' read -r -a GPU_ARRAY <<< "${CUDA_VISIBLE_DEVICES}"
NUM_GPUS="${#GPU_ARRAY[@]}"

# Automatically construct comma-separated list of environment addresses
ENV_ADDR_LIST=""
for i in $(seq 0 $((NUM_GPUS - 1))); do
  PORT=$((BASE_PORT + i))
  ADDR="http://${ENV_ADDR_HOST}:${PORT}"
  if [[ -z "${ENV_ADDR_LIST}" ]]; then
    ENV_ADDR_LIST="${ADDR}"
  else
    ENV_ADDR_LIST="${ENV_ADDR_LIST},${ADDR}"
  fi
done
ENV_ADDR="${ENV_ADDR:-${ENV_ADDR_LIST}}"
echo "Using ENV_ADDR: ${ENV_ADDR}"

WANDB_MODE="${WANDB_MODE:-online}"
WANDB_API_KEY="${WANDB_API_KEY:-}"
PROJECT_NAME="${PROJECT_NAME:-agentgym-alfworld}"

KL_COEF="${KL_COEF:-0.001}"
ENTROPY_COEF="${ENTROPY_COEF:-0.001}"
POLICY_LR="${POLICY_LR:-1e-6}"
ROLLOUT_N="${ROLLOUT_N:-8}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-8}"
PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"
PPO_EPOCHS="${PPO_EPOCHS:-1}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-2}"
MAX_ROUNDS="${MAX_ROUNDS:-20}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-4096}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_TOKENS_PER_TURN="${MAX_TOKENS_PER_TURN:-512}"
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.80}"
SAVE_FREQ="${SAVE_FREQ:-50}"

ENABLE_ERC="${ENABLE_ERC:-0}"
ERC_MU_BASE="${ERC_MU_BASE:-1.0}"
ERC_MU_EXP="${ERC_MU_EXP:-1.5}"
ERC_ETA_WM="${ERC_ETA_WM:-2.0}"
ERC_LAMBDA_WM="${ERC_LAMBDA_WM:-1.0}"
ERC_CLIPPING_TYPE="${ERC_CLIPPING_TYPE:-global}"
ERC_CLIPPING_METHOD="${ERC_CLIPPING_METHOD:-add}"
ERC_MOMENTUM="${ERC_MOMENTUM:-0.8}"
WMLOSS_ADD_COEF="${WMLOSS_ADD_COEF:-0.5}"
WMLOSS_ADD_COEF_END="${WMLOSS_ADD_COEF_END:-0}"
WMLOSS_ADD_HORIZON="${WMLOSS_ADD_HORIZON:-0}"
WMLOSS_ADD_USE_ENTROPY="${WMLOSS_ADD_USE_ENTROPY:-True}"
WMLOSS_ADD_USE_EMA="${WMLOSS_ADD_USE_EMA:-False}"
WMLOSS_ADD_USE_GROUPED="${WMLOSS_ADD_USE_GROUPED:-False}"
WMLOSS_ADD_USE_REF_BASELINE="${WMLOSS_ADD_USE_REF_BASELINE:-False}"
WMLOSS_ADD_ONLY_FAILED="${WMLOSS_ADD_ONLY_FAILED:-True}"
WMLOSS_ADD_USE_PI_WEIGHT="${WMLOSS_ADD_USE_PI_WEIGHT:-True}"
WMLOSS_ADD_BASELINE="${WMLOSS_ADD_BASELINE:-ema}"

ERC_ENABLE_VALUE="False"
if [[ "${ENABLE_ERC}" == "1" ]]; then
  ERC_ENABLE_VALUE="True"
fi

WMC_COEFF="${WMC_COEFF:-0.01}"
WMC_TYPE="${WMC_TYPE:-fixed}"
WMC_START_COEFF="${WMC_START_COEFF:-0.001}"
WMC_END_COEFF="${WMC_END_COEFF:-0.0}"
WMC_HORIZON="${WMC_HORIZON:-100}"
WMC_POWER="${WMC_POWER:-2}"
WMC_CUTOFF_STEP="${WMC_CUTOFF_STEP:-50}"

WM_ENABLE="${WM_ENABLE:-False}"
WM_LOSS_PI_DEDUP="${WM_LOSS_PI_DEDUP:-True}"

WM_ENV_PREDICT_PROMPT="${WM_ENV_PREDICT_PROMPT:-null}"
WM_MAX_LENGTH="${WM_MAX_LENGTH:-4096}"
WM_MAX_SAMPLES_PER_TRAJECTORY="${WM_MAX_SAMPLES_PER_TRAJECTORY:-null}"
WM_MIN_ENV_TOKENS="${WM_MIN_ENV_TOKENS:-1}"

EXP_NAME="${EXP_NAME:-alfworld_grpo_qwen2.5_3b_$(date -u +%Y%m%d_%H%M%S)}"
CKPT_DIR="${CKPT_DIR:-${ROOT}/checkpoints/${EXP_NAME}}"
RUN_DIR="${RUN_DIR:-${ROOT}/runlogs/${EXP_NAME}}"
ROLLOUT_LOG_DIR="${ROLLOUT_LOG_DIR:-${RUN_DIR}/rollout_logs}"
TRAIN_FILE="${TRAIN_FILE:-${ROOT}/AgentItemId/train/alfworld_train.json}"
LOG_PATH="${LOG_PATH:-}"

mkdir -p "${CKPT_DIR}" "${RUN_DIR}" "${ROLLOUT_LOG_DIR}"
if [[ -n "${LOG_PATH}" ]]; then
  mkdir -p "$(dirname "${LOG_PATH}")"
  exec >"${LOG_PATH}" 2>&1
fi

source "${CONDA_SH}"
set +u
conda activate "${TRAIN_ENV}"
set -u

python "${ROOT}/scripts/prepare_alfworld_grpo_splits.py"

export NO_PROXY="${NO_PROXY:-127.0.0.1,localhost}"
export no_proxy="${no_proxy:-127.0.0.1,localhost}"

cd "${TRAIN_CODE_DIR}"

# CPU affinity wrapper (limits how many CPUs Ray sees → avoids prestart worker storm)
TASKSET_PREFIX=""
if [[ -n "${CPU_AFFINITY:-}" ]]; then
  TASKSET_PREFIX="taskset -c ${CPU_AFFINITY}"
fi

exec ${TASKSET_PREFIX} env \
  -u http_proxy -u https_proxy -u all_proxy \
  -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
  NO_PROXY="${NO_PROXY}" \
  no_proxy="${no_proxy}" \
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  VLLM_USE_MODELSCOPE=0 \
  VLLM_WORKER_MULTIPROC_METHOD=spawn \
  VLLM_ATTENTION_BACKEND=FLASH_ATTN \
  HYDRA_FULL_ERROR=1 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  WANDB_MODE="${WANDB_MODE}" \
  WANDB_API_KEY="${WANDB_API_KEY}" \
  RAY_TMPDIR="${RAY_TMPDIR:-/tmp/ray}" \
  python -m verl.agent_trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.rounds_ctrl.type=fixed \
    algorithm.rounds_ctrl.rounds="${MAX_ROUNDS}" \
    data.train_file="${TRAIN_FILE}" \
    data.train_batch_size="${TRAIN_BATCH_SIZE}" \
    data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
    data.max_response_length="${MAX_RESPONSE_LENGTH}" \
    actor_rollout_ref.agentgym.task_name="${TASK_NAME}" \
    actor_rollout_ref.agentgym.env_addr="'${ENV_ADDR}'" \
    actor_rollout_ref.agentgym.timeout=2400 \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef="${KL_COEF}" \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=${ENTROPY_COEF} \
    actor_rollout_ref.actor.ppo_epochs="${PPO_EPOCHS}" \
    actor_rollout_ref.actor.optim.lr="${POLICY_LR}" \
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU}" \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.dtype=bfloat16 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.load_format=dummy_dtensor \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEMORY_UTILIZATION}" \
    actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
    actor_rollout_ref.rollout.max_model_len="${MAX_MODEL_LEN}" \
    actor_rollout_ref.rollout.max_tokens="${MAX_TOKENS_PER_TURN}" \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.rollout_log_dir="${ROLLOUT_LOG_DIR}" \
    algorithm.kl_ctrl.kl_coef="${KL_COEF}" \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXP_NAME}" \
    trainer.default_local_dir="${CKPT_DIR}" \
    trainer.save_freq="${SAVE_FREQ}" \
    trainer.total_epochs="${TOTAL_EPOCHS}" \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node="${NUM_GPUS}" \
    wmc_erc.enable="${ERC_ENABLE_VALUE}" \
    wmc_erc.mu_base="${ERC_MU_BASE}" \
    wmc_erc.mu_exp="${ERC_MU_EXP}" \
    wmc_erc.eta_wm="${ERC_ETA_WM}" \
    wmc_erc.lambda_wm="${ERC_LAMBDA_WM}" \
    wmc_erc.clipping_type="${ERC_CLIPPING_TYPE}" \
    wmc_erc.clipping_method="${ERC_CLIPPING_METHOD}" \
    wmc_erc.momentum="${ERC_MOMENTUM}" \
    +wmc_erc.wmloss_add_coef="${WMLOSS_ADD_COEF}" \
    +wmc_erc.wmloss_add_use_entropy="${WMLOSS_ADD_USE_ENTROPY}" \
    +wmc_erc.wmloss_add_use_ema="${WMLOSS_ADD_USE_EMA}" \
    +wmc_erc.wmloss_add_use_grouped="${WMLOSS_ADD_USE_GROUPED}" \
    +wmc_erc.wmloss_add_use_ref_baseline="${WMLOSS_ADD_USE_REF_BASELINE}" \
    +wmc_erc.wmloss_add_only_failed="${WMLOSS_ADD_ONLY_FAILED}" \
    +wmc_erc.wmloss_add_use_pi_weight="${WMLOSS_ADD_USE_PI_WEIGHT}" \
    +wmc_erc.wmloss_add_baseline="${WMLOSS_ADD_BASELINE}" \
    +wmc_erc.wmloss_add_coef_end="${WMLOSS_ADD_COEF_END}" \
    +wmc_erc.wmloss_add_horizon="${WMLOSS_ADD_HORIZON}" \
    actor_rollout_ref.actor.world_model_coeff="${WMC_COEFF}" \
    actor_rollout_ref.actor.world_model.enable="${WM_ENABLE}" \
    actor_rollout_ref.actor.world_model.env_predict_prompt="${WM_ENV_PREDICT_PROMPT}" \
    actor_rollout_ref.actor.world_model.max_length="${WM_MAX_LENGTH}" \
    actor_rollout_ref.actor.world_model.max_samples_per_trajectory="${WM_MAX_SAMPLES_PER_TRAJECTORY}" \
    actor_rollout_ref.actor.world_model.min_env_tokens="${WM_MIN_ENV_TOKENS}" \
    +actor_rollout_ref.actor.wm_loss_pi_dedup="${WM_LOSS_PI_DEDUP}" \
    algorithm.world_model_coeff_ctrl.type="${WMC_TYPE}" \
    algorithm.world_model_coeff_ctrl.start_coeff="${WMC_START_COEFF}" \
    algorithm.world_model_coeff_ctrl.end_coeff="${WMC_END_COEFF}" \
    algorithm.world_model_coeff_ctrl.horizon="${WMC_HORIZON}" \
    algorithm.world_model_coeff_ctrl.power="${WMC_POWER}" \
    algorithm.world_model_coeff_ctrl.cutoff_step="${WMC_CUTOFF_STEP}"
