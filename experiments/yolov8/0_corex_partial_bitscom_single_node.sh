#!/usr/bin/env bash
set -euo pipefail

# 单机 4 卡 yolov8 partial-sync（部分梯度同步）+ bitscom 低比特量化脚本（u62 本地）
# 用法：conda activate trj-test 后直接运行，参数可用环境变量覆盖。
#   NPROC_PER_NODE=4 SYNC_INTERVAL=4 MICRO_STEPS=100 COMM_BACKEND=bitscom BITWIDTH=4 \
#     ./0_corex_partial_bitscom_single_node.sh

export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-DETAIL}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"

NPROC="${NPROC_PER_NODE:-4}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29501}"
SYNC_INTERVAL="${SYNC_INTERVAL:-4}"
MICRO_STEPS="${MICRO_STEPS:-100}"
COMM_BACKEND="${COMM_BACKEND:-bitscom}"
BITWIDTH="${BITWIDTH:-4}"
RUN_NAME="${RUN_NAME:-partial_bitscom}"

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
  experiments/yolov8/run_yolov8_ddp_partial_sync.py \
  --task detect \
  --model yolov8n.pt \
  --data experiments/yolov8/holes_v3.yaml \
  --imgsz 640 \
  --sync-interval "$SYNC_INTERVAL" \
  --micro-steps "$MICRO_STEPS" \
  --comm-backend "$COMM_BACKEND" \
  --bitwidth "$BITWIDTH" \
  --optimizer adamw \
  --lr 0.001 \
  --weight-decay 0.01 \
  --grad-clip 5.0 \
  --grad-clip-steps 5 \
  --run-name "$RUN_NAME" \
  "${EXTRA_ARGS[@]}"
