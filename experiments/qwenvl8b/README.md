# QwenVL8B 2DP x 8PP Experiments

仿照 [`experiments/qwen14b`](../qwen14b/) 搭的 Qwen3-VL 多模态训练实验，模型是
[`Qwen/Qwen3-VL-8B-Instruct`](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)。

拓扑（两个节点 x 16 卡 = 32 卡）：

- DP = 4
- PP = 8
- TP = 1
- 1F1B pipeline schedule
- micro-batches = 8
- per-device batch size = 8
- sequence length = 256
- 每样本 1 张图，patch 网格 4x4（即 64x64 像素，patch_size=16）

> 和 Qwen14B 的唯一拓扑差别是 **TP=1 而不是 TP=2**：
> polar wrapper 的 TP 计划目前写死了 `model.model.layers`，而 Qwen3-VL 的语言层在
> `model.model.language_model.layers`。要把 TP 打开需要先扩 wrapper 的 TP 计划，
> 这一版先不做，用 PP x DP 把 32 张卡铺满。

## 和 Qwen14B 的三点关键区别

### 1. 模型是 VL 模型，视觉塔挂在 PP 的第一个 stage

Qwen3-VL 的结构是：

```
Qwen3VLForConditionalGeneration
├── model (Qwen3VLModel)
│   ├── visual          (Qwen3VLVisionModel, 27 层 ViT + merger + deepstack)
│   └── language_model  (Qwen3VLTextModel)
│       ├── embed_tokens
│       ├── layers      (36 层)
│       ├── norm
│       └── rotary_emb
└── lm_head
```

PP 切分规则（`partition_qwen_vl_model`，见
[`run_qwenvl8b_polar_dp_pp.py`](run_qwenvl8b_polar_dp_pp.py)）：

| stage | 保留的模块 |
| --- | --- |
| 0 | `visual` + `language_model.embed_tokens` + 自己那段 `layers` |
| 中间 | 只有自己那段 `layers` |
| 最后一个 | `language_model.norm` + `lm_head` + 自己那段 `layers` |

36 层切 8 段：前 4 段各 5 层，后 4 段各 4 层。

阶段 0 的 forward 多做两件事：

1. 跑视觉塔 `visual(pixel_values, grid_thw=image_grid_thw)`，把输出的图片特征
   用 `masked_scatter` 填到 `input_ids` 里 `image_token_id`（151655）那些位置上；
2. DeepStack：视觉塔第 8/16/24 层抽出来的特征，分别加回语言模型第 0/1/2 层。
   这三层都在 stage 0，所以只有 stage 0 需要这份额外输入。

### 2. 数据是随机搓的，只保留 get_dataloader 接口

`get_dataloader(cfg, tokenizer, pp_size, tokens_per_image)` 目前返回
`RandomVLDataset`：随机 token + 随机 `pixel_values` + 规整的 `image_grid_thw`。
换成真实数据集时，只要产出的 batch 字段一样就行，训练循环不用动：

| 字段 | 形状 | 说明 |
| --- | --- | --- |
| `input_ids` | `[B, seq_len]` | 开头若干位置是 `image_token_id` |
| `labels` | `[B, seq_len]` | 目前等于 `input_ids` |
| `attention_mask` | `[B, seq_len]` | 全 1 |
| `pixel_values` | `[B * patches_per_image, 1536]` | 1536 = 3 通道 x 2 帧 x 16 x 16 |
| `image_grid_thw` | `[B * images_per_sample, 3]` | 每行 `[1, grid_h, grid_w]` |

注意 `vl_collate`：`pixel_values` / `image_grid_thw` 是沿 dim 0 **拼接**而不是
stack 的，因为它们要按图片数被 pipeline 均分到各个 micro-batch 里。

### 3. 权重只从网上取 config，本地随机初始化

`build_qwen_vl_model()` 默认走 `AutoConfig.from_pretrained()` +
`AutoModelForImageTextToText.from_config()`（在 `torch.device("meta")` 上），
**不下载任何权重文件**。HF 缓存里 `models--Qwen--Qwen3-VL-8B-Instruct` 只会有一个
几百字节的 `config.json` 加 tokenizer。

真正的参数在 `PolarParallel` 里由 `to_empty()` + `_reset_module_parameters()` 落在
设备上随机初始化。`--init-from-pretrained` 是预留的真权重入口，打开后走
`from_pretrained()`。

## 跑起来

```bash
# 创新路径 POLAR + bitscom（node 0 在 u62 上直接跑）
bash experiments/qwenvl8b/0_train_qwenvl8b_polar_dp_pp.sh
ssh u210 "bash /data1/tangruijing/udtca/experiments/qwenvl8b/1_train_qwenvl8b_polar_dp_pp.sh"

# 稠密 DP baseline
bash experiments/qwenvl8b/0_train_qwenvl8b_baseline_ddp_1f1b.sh
ssh u210 "bash /data1/tangruijing/udtca/experiments/qwenvl8b/1_train_qwenvl8b_baseline_ddp_1f1b.sh"
```

改地址/端口：

```bash
MASTER_ADDR=<node0_ip> MASTER_PORT=11234 bash experiments/qwenvl8b/0_train_qwenvl8b_polar_dp_pp.sh
```

`--comm-timing` 是“第几个 micro-batch 触发 POLAR 通信”，必须落在
`[0, micro-batches)` 里，这里 8 个 micro-batch 取 4。

## 单机自测（改代码时用这个，比双机快）

一台节点的 16 张卡跑 PP=8 + DP=2，和双机的 PP/DP 形状一致：

```bash
export HF_ENDPOINT=https://hf-mirror.com
torchrun --nproc_per_node=16 --nnodes=1 --node_rank=0 \
  --master_addr=127.0.0.1 --master_port=29528 \
  experiments/qwenvl8b/run_qwenvl8b_polar_dp_pp.py \
  --pp-size 8 --tp-size 1 --micro-batches 8 --per-device-batch-size 8 \
  --seq-len 256 --images-per-sample 1 --image-grid-h 4 --image-grid-w 4 \
  --comm-timing 4 --max-steps 3 \
  --using-polar true --polar-hook ef_lowmem --polar-bucket-numel 64000000 \
  --polar-max-inflight-buckets 4 --method bitscom --bitwidth 4 \
  --num-workers 0 --disable-profiler true
```

跑之前先确认卡空着、限速关掉：

```bash
ixsmi | grep MiB                 # 有没有别的进程占卡
tc qdisc show dev ens1f0 | grep htb   # 出现 htb 就是在限速
ps aux | grep torchrun | grep -v grep
```

**注意这台机器是共用节点。** 别的任务随时可能占走卡；如果训练卡在
`dist.barrier()` 或者其他集合通信上不动，先确认卡还在不在自己手里。

## 已验证的配置

下面这些都在这台机器上实际跑通过（`max-steps 3`，loss ≈ 12.7，符合
`ln(151936) ≈ 11.9` 的随机基线预期）：

| 配置 | 卡数 | 结果 |
| --- | --- | --- |
| PP=8 DP=1 baseline，seq 64 | 8 | loss 12.7211 |
| PP=8 DP=1 POLAR+bitscom 4bit，seq 64 | 8 | loss 12.7342，ef_lowmem 分桶走 bitscom |
| PP=8 DP=2 POLAR+bitscom 4bit，seq 256（= 双机同形状） | 16 | loss 12.7280 |
| PP=8 DP=2 baseline，seq 256 | 16 | loss 12.7465 |

**双机（u210）还没实跑过**：跑之前要先确认 u210 上的 `polar-sgd` 也带上了这一版
wrapper 改动（见下），否则 TP / 参数初始化的分支对不上。

## 对 polar-sgd 的改动

Qwen3-VL 和 Qwen2 的结构差异需要 `polar-sgd/src/psgd/parallelism/polar/wrapper.py`
配合，改动都是**向后兼容**的（对 Qwen2/Llama 完全不生效）：

1. 训练循环把 batch 里 `input_ids` / `labels` / `attention_mask` 之外的张量
   （`pixel_values`、`image_grid_thw`）作为 kwargs 透传给各 stage 的 forward。
   纯文本 batch 没有这些 key，`extra_inputs` 是空 dict，调用方式和以前一模一样。
2. `_reset_module_parameters()` 增加 `nn.LayerNorm` / `ConvNd` 两个分支 ——
   视觉塔用的是标准 LayerNorm 和 Conv3d，漏掉的话 `to_empty()` 之后这些参数是
   未初始化的显存垃圾。Qwen2.5-14B 里没有任何 LayerNorm / Conv 模块，新分支不会触发。
3. 新增 `_stage_layer_list()`，从 `model.model.layers`（Qwen2/Llama）或
   `model.model.language_model.layers`（Qwen3-VL）取 decoder 层，替换原来写死
   `self.stage_model.model.layers` 的那句调试打印。
4. RoPE 的 `inv_freq` 是非持久 buffer，`to_empty()` 之后是垃圾值。脚本里
   `_ensure_rotary_embeddings()` 在第一次 forward 前重建视觉塔和语言模型的
   `rotary_emb`（Qwen14B 脚本里对 RoPE 也是同样的处理思路）。

## 显存

单 stage 的显存 ≈ 权重(bf16) + 梯度 + AdamW 的 2 份 fp32 状态：

| stage | 参数量 | 权重 | 梯度 | AdamW 状态 | 合计 |
| --- | --- | --- | --- | --- | --- |
| 0（视觉塔 + embed + 5 层） | ~2.1B | 4.3G | 4.3G | 17.0G | ~26G |
| 1-3（5 层） | ~1.0B | 1.9G | 1.9G | 7.7G | ~12G |
| 4-7（4 层） | ~0.8B | 1.5G | 1.5G | 6.2G | ~9G |

stage 0 是最紧的一个，32G 卡上站得住但没多少余量。要更保险可以调小
`--micro-batches` / `--seq-len`（只影响 activation，不影响上面这块）。

## 已知限制

- `--tp-size` 只能给 1，理由见上。
- `--pp-size` 必须 >= 2：`PolarParallel` 的 1F1B 循环只在 `is_first` / `is_last`
  两个分支里喂 `target`，单 stage 会算不出 loss。
- `--use-flash-attn` 默认关（走 sdpa）。视觉塔用的是 varlen `cu_seqlens` 路径，
  开 flash attention 需要 Corex 版 flash-attn 支持 varlen，没验证过。
