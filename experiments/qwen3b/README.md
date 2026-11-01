# Qwen3B 2DP x 8PP x 2TP Experiments

Two-node **Qwen2.5-3B-Instruct** experiments, structural copy of
`experiments/qwen14b/`. 3B is the small stand-in we use to iterate quickly
(one step is seconds rather than minutes on 14B) while keeping the exact same
PP / TP / DP code path.

- 2 nodes x 16 GPUs = 32 GPUs
- DP = 2 across nodes
- PP = 8
- TP = 2
- 1F1B pipeline schedule
- micro-batches = 32
- per-device batch size = 32
- sequence length = 256
- node-local token cache under `experiments/qwen3b/cache`

Only `LOCAL_RANK=0` on each node contacts Hugging Face to warm the
tokenizer/config cache and materialize the token-block dataset cache. Other
local ranks wait at a distributed barrier and then read local files.

## 和 14B 的差别（只有这些）

| 项 | qwen14b | qwen3b |
|---|---|---|
| 模型 | `Qwen/Qwen2.5-14B-Instruct` | `Qwen/Qwen2.5-3B-Instruct` |
| 层数 | 48 | 36 |
| attention | 40 头 / 8 KV 头 | 16 头 / 2 KV 头（GQA） |
| token 缓存 | `experiments/qwen14b/cache` | `experiments/qwen3b/cache` |
| ckpt 目录 | `checkpoints/qwen2_5_14b_instruct` | `checkpoints/qwen2_5_3b_instruct` |
| MASTER_PORT | 29500 / 29501 | 29520 / 29521 |

两条 3B 特有的硬约束（入口脚本会校验）：

- **TP 只能取 1 或 2**：GQA 只有 2 个 KV 头，TP>2 时每张卡分不到整 KV 头，
  启动阶段直接报错。
- **PP 不要求整除层数**：36 层按 stage 均分，余数从前几个 stage 摊，
  pp=8 时前 4 个 stage 各 5 层、后 4 个各 4 层；pp=4 时每 stage 9 层。

其余（入口脚本、POLAR/bitscom 参数、baseline 逻辑）跟 14B 一字不差。

## 首跑前：模型要在本地 HF 缓存里

入口固定 `local_files_only=True`，两台节点各拉一次（约 6.2 GB）：

```bash
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download Qwen/Qwen2.5-3B-Instruct
```

## POLAR + bitscom

This is the innovation path:

- `--using-polar true`
- POLAR hook = `ef_lowmem`
- bitscom DP communication = 4-bit
- run label = `polar_bitscom_1f1b_tp`

```bash
# node 0 (powerone1)
bash experiments/qwen3b/0_train_qwen3b_polar_dp_pp_tp.sh

# node 1 (powerone2)
bash experiments/qwen3b/1_train_qwen3b_polar_dp_pp_tp.sh
```

## Dense DP Baseline

Same 1F1B + TP topology, POLAR and bitscom off:

- `--using-polar false`
- dense DP gradient averaging after each stage finishes 1F1B backward
- run label = `baseline_ddp_1f1b_tp`

```bash
# node 0
bash experiments/qwen3b/0_train_qwen3b_baseline_ddp_1f1b_tp.sh

# node 1
bash experiments/qwen3b/1_train_qwen3b_baseline_ddp_1f1b_tp.sh
```

Override launch address/port if needed:

```bash
MASTER_ADDR=<node0_ip> MASTER_PORT=11234 bash experiments/qwen3b/0_train_qwen3b_polar_dp_pp_tp.sh
MASTER_ADDR=<node0_ip> MASTER_PORT=11234 bash experiments/qwen3b/1_train_qwen3b_polar_dp_pp_tp.sh
```

要换卡数/并行度时不用改这些脚本，直接用 `udtca-config/qwen3b/` 下的启动器
（`run_docker.py --gpus/--pp/--tp`、`run_docker_single_node.sh`），见那边的 README。
