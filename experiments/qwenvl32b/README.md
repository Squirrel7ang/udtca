# QwenVL32B 4DP x 8PP Experiments

[`experiments/qwenvl8b`](../qwenvl8b/) 的 32B 版本，模型是
[`Qwen/Qwen3-VL-32B-Instruct`](https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct)。

> 注意：通义**没有** `Qwen/Qwen-VL-32B-Instruct` 这个仓库（HF API 返回 401）。
> 32B 档只有 `Qwen/Qwen3-VL-32B-Instruct` 和 `Qwen/Qwen2.5-VL-32B-Instruct`。
> 这里用前者，因为它和 8B 脚本是同一套 `Qwen3VLForConditionalGeneration`
> 结构（同一个 visual tower、同样的 deepstack），代码可以直接复用；
> 2.5 是另一套 `Qwen2_5_VLForConditionalGeneration`，得重写一遍。

拓扑和 8B **完全一致**（两个节点 x 16 卡 = 32 卡）：

- DP = 4
- PP = 8
- TP = 1
- 1F1B pipeline schedule
- micro-batches = 8
- per-device batch size = 8
- sequence length = 256
- 每样本 1 张图，patch 网格 4x4（即 64x64 像素，patch_size=16）

## 和 8B 脚本的全部差别

代码是 8B 那份逐行复制的，只动了四处：

| | 8B | 32B |
| --- | --- | --- |
| 模型 | `Qwen3-VL-8B-Instruct` | `Qwen3-VL-32B-Instruct` |
| 语言层数 | 36 | 64 |
| hidden | 4096 | 5120 |
| 优化器 | AdamW | **SGD**（原因见下，唯一一个实质性改动） |
| 端口 | 29510 / 29511 | 29520 / 29521 |

新增了一个 `--optimizer {adamw,sgd}` 开关（默认 `adamw`，和 wrapper 默认值一致），
`PolarParallel.__init__` 本来就收这个参数，只是 8B 脚本没暴露出来。

## 为什么必须用 SGD

本机单卡 **32 GiB**，PP 只在 stage 之间切参数，DP 是复制的，所以每卡显存
≈ `stage 参数量 × 每参数字节数`。每参数字节数：

| 优化器 | 权重(bf16) | 梯度(bf16) | 优化器状态 | 合计 |
| --- | --- | --- | --- | --- |
| AdamW | 2 | 2 | 2 份 fp32 = 8 | **12** |
| SGD | 2 | 2 | 无（wrapper 没传 momentum，不留动量缓冲） | **4** |

实测参数量（从 HF config 在 meta 上建模型数出来的）：

| | 8B | 32B |
| --- | --- | --- |
| 视觉塔 | 0.576B | 0.595B |
| `embed_tokens` | 0.622B | 0.778B |
| 单层 decoder | 0.193B | 0.488B |
| 合计 | 8.77B | 33.36B |

PP=8 时每个 stage 的显存：

| stage | 参数 | AdamW(12B/参数) | SGD(4B/参数) |
| --- | --- | --- | --- |
| 0 = visual + embed + 8 层 | 5.27B | 58.9 GiB ❌ | **21.1 GiB** ✅ |
| 中间 = 8 层 | 3.90B | 43.6 GiB ❌ | **15.6 GiB** ✅ |

加上 activation、NCCL 缓冲、CUDA context 大约再 2 GiB，SGD 下 stage 0 约 23 GiB、
中间 stage 约 17 GiB，32 GiB 的卡上有余量。

**AdamW 在所有 PP 档位都装不下**：可用参数上限 `(32-3)/12 ≈ 2.4B`，
stage 0 光视觉塔+`embed_tokens` 就 1.37B，只能再塞 2 层；PP=16 时中间 stage 各 4 层，
合计 `2 + 15×4 = 62 < 64` 层，无论怎么切都差 2 层。PP=32/DP=1 能装下，
但那样跨节点就没有梯度 all-reduce 了，实验本身失去意义。

代价：32B 这组和 8B 那组的优化器不一致，**收敛行为不可直接对比**；
通信/计算的耗时对比仍然有效（两边 baseline 和 POLAR 用的都是 SGD）。

## 跑起来

```bash
# 创新路径 POLAR + bitscom（node 0 在 u62 上直接跑）
bash experiments/qwenvl32b/0_train_qwenvl32b_polar_dp_pp.sh
ssh u210 "bash /data1/tangruijing/udtca/experiments/qwenvl32b/1_train_qwenvl32b_polar_dp_pp.sh"

# 稠密 DP baseline
bash experiments/qwenvl32b/0_train_qwenvl32b_baseline_ddp_1f1b.sh
ssh u210 "bash /data1/tangruijing/udtca/experiments/qwenvl32b/1_train_qwenvl32b_baseline_ddp_1f1b.sh"
```

走编排器（推荐，会自动限速 + 收日志）：

```bash
python udtca-config/qwenvl32b/generate_and_run.py            # 主入口，POLAR + baseline 全部
python udtca-config/qwenvl32b/_run_baseline.py               # 内部脚本：只跑 baseline 的 case
```

## 单机自测（改代码时用这个，比双机快）

一台节点的 16 张卡跑 PP=8 + DP=2：

```bash
export HF_ENDPOINT=https://hf-mirror.com
torchrun --nproc_per_node=16 --nnodes=1 --node_rank=0 \
  --master_addr=127.0.0.1 --master_port=29528 \
  experiments/qwenvl32b/run_qwenvl32b_polar_dp_pp.py \
  --pp-size 8 --tp-size 1 --micro-batches 8 --per-device-batch-size 8 \
  --seq-len 256 --images-per-sample 1 --image-grid-h 4 --image-grid-w 4 \
  --comm-timing 4 --max-steps 3 --optimizer sgd \
  --using-polar false --baseline-mode manual --polar-hook none --method none \
  --num-workers 0 --disable-profiler true
```

跑之前先确认卡空着：

```bash
ixsmi --query-gpu=index,memory.used --format=csv,noheader
pgrep -af 'run_qwen.*py|torchrun'
```

## 已知限制

- `--tp-size` 只能给 1：polar wrapper 的 TP 计划写死了 `model.model.layers`，
  而 Qwen3-VL 的语言层在 `model.model.language_model.layers`。要开 TP 得先扩 wrapper。
- `--pp-size` 必须 >= 2：`PolarParallel` 的 1F1B 循环只在 `is_first` / `is_last`
  分支里喂 `target`，单 stage 算不出 loss。
- **DeepStack 语义目前是错的**（8B 脚本也一样，这里沿用未改）：视觉塔输出的 3 组
  deepstack 特征应该加回语言模型第 8/16/24 层，但当前实现是 stage 0 上的
  `global_idx` 从 0 开始数，实际加到了第 0/1/2 层；而第 8/16/24 层所在的 stage
  （PP=8 时是 stage 1/3/5）根本收不到这份输入，等于整个跳过。
  随机数据 + 只看耗时的实验不受影响；要用真数据训模型前必须修。
- `--use-flash-attn` 默认关（走 sdpa）。视觉塔用的是 varlen `cu_seqlens` 路径，
  开 flash attention 需要 Corex 版 flash-attn 支持 varlen，没验证过。
