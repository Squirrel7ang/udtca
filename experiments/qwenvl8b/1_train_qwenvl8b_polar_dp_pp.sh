#!/usr/bin/env bash
set -euo pipefail

# Node 1 of 2. 创新路径：POLAR + bitscom。
# MASTER_ADDR / MASTER_PORT 必须和 node 0 完全一致。
# PP=8、TP=1、DP=4。

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
MASTER_ADDR="${MASTER_ADDR:-10.31.10.62}"
MASTER_PORT="${MASTER_PORT:-29510}"
NNODES="${NNODES:-2}"
NPROC_PER_NODE="${NPROC_PER_NODE:-16}"
NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-ens1f0}"
NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"

export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-0}"
export NCCL_SOCKET_IFNAME
export NCCL_IB_DISABLE

# ssh / conda 起的都是非交互非登录 shell，不会 source ~/.bashrc，HF 镜像要显式导出
unset http_proxy HTTP_PROXY https_proxy HTTPS_PROXY all_proxy ALL_PROXY NO_PROXY no_proxy
export HF_ENDPOINT=https://hf-mirror.com

torchrun \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --nnodes="${NNODES}" \
  --node_rank=1 \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  "${SCRIPT_DIR}/run_qwenvl8b_polar_dp_pp.py" \
  --model-name Qwen/Qwen3-VL-8B-Instruct \
  --pp-size 8 \
  --tp-size 1 \
  --micro-batches 8 \
  --per-device-batch-size 8 \
  --seq-len 256 \
  --images-per-sample 1 \
  --image-grid-h 4 \
  --image-grid-w 4 \
  --comm-timing 4 \
  --max-steps 200 \
  --lr 2e-4 \
  --using-polar true \
  --run-label polar_bitscom_qwenvl8b \
  --polar-hook ef_lowmem \
  --polar-bucket-numel 64000000 \
  --polar-max-inflight-buckets 4 \
  --method bitscom \
  --bitwidth 4
