#!/usr/bin/env bash
set -euo pipefail

# 单机 4 卡 yolov8 + bitscom DDP 冒烟/训练脚本（u62 本地）
# 用法：conda activate trj-test 后直接运行，参数可用环境变量覆盖。
#   NPROC_PER_NODE=4 STEPS=20 METHOD=bitscom BITWIDTH=4 ./0_corex_bitscom_single_node.sh

export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-DETAIL}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"

NPROC="${NPROC_PER_NODE:-4}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"
STEPS="${STEPS:-20}"
METHOD="${METHOD:-bitscom}"
BITWIDTH="${BITWIDTH:-4}"
RUN_NAME="${RUN_NAME:-bitscom_single}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."   # 回到 udtca 根目录，脚本里用的是相对路径

EXTRA_ARGS=()
if [[ "${NO_EVAL:-1}" == "1" ]]; then
    EXTRA_ARGS+=(--no-eval)
fi

torchrun \
  --nproc_per_node="$NPROC" \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr="$MASTER_ADDR" \
  --master_port="$MASTER_PORT" \
  experiments/yolov8/run_yolov8_ddp_baseline.py \
  --task detect \
  --model yolov8n.pt \
  --data experiments/yolov8/holes_v3.yaml \
  --imgsz 640 \
  --steps "$STEPS" \
  --optimizer adamw \
  --lr 0.001 \
  --weight-decay 0.01 \
  --grad-clip 1.0 \
  --run-name "$RUN_NAME" \
  --method "$METHOD" \
  --bitwidth "$BITWIDTH" \
  "${EXTRA_ARGS[@]}"
