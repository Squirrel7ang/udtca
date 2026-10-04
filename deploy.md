# 部署与启动

两机 16 卡 ×2（共 32 卡）跑 Qwen14B / QwenVL8B 的 PP+DP 训练。
机器固定，环境相同，本文只讲**怎么配、怎么起、日志在哪**。

---

## 0. 机器与拓扑

| | node0 | node1 |
|---|---|---|
| 主机 | u62 (`10.31.10.62`) | u210 (`10.31.10.210`) |
| ssh 别名 | 本机 | `u210` |
| 网卡 | `ens1f0` | `ens1f0` |
| 仓库 | `/data1/tangruijing/udtca` | 同一路径 |

代码只改 node0，改完同步到 u210：

```bash
cd /data1/tangruijing/udtca
rsync -a --delete experiments/qwen14b experiments/qwenvl8b u210:/data1/tangruijing/udtca/experiments/
rsync -a --delete udtca-config u210:/data1/tangruijing/udtca/
```

---

## 1. 环境

conda 环境固定为 `trj-test`，依赖已装好（torch 2.7.1 corex 版本）。

如果需要自己重新启一个 conda 环境，需要自行安装 corex 依赖库。包括 torch 等，具体参考 trj-test 里面的内容。此外，需要安装 polar-sgd 库以及编译安装 bitscom 库。对于 bitscom 库，请查看 README\_cn.md 中关于天数 GPU 的编译安装说明。大体就是设置好环境变量。这个环境变量已经放在 .bashrc 中，应该是默认设置好的。

```bash
conda activate trj-test
```

Corex 4.4.0 相关的变量（非交互 shell 不会自动带上，**每条命令前都要有**）：

这里需要注意，直接使用 udtca-config 运行时，脚本会默认为每一个 ssh 命令加上，但是如果要自己跑的话，建议都加上。尤其是当提示 ld 或者某某 libxxx.so 找不到的时候。

```bash
export COREX_PATH=/usr/local/corex-4.4.0
export CUDA_HOME="$COREX_PATH"
export CPATH="$COREX_PATH/include:${CPATH:-}"
export LIBRARY_PATH="$COREX_PATH/lib64:${LIBRARY_PATH:-}"
export TRITON_CUDA_SYSROOT="$COREX_PATH"
export BITSCOM_CUDA_COMPILER="$COREX_PATH/bin/clang++"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PATH="$COREX_PATH/bin:/root/miniconda3/envs/trj-test/bin:$PATH"
```

目前 qwen14b 是走 HF 拉 config 走镜像（非登录 shell 不读 `~/.bashrc`，必须显式导出）：

```bash
unset http_proxy HTTP_PROXY https_proxy HTTPS_PROXY all_proxy ALL_PROXY NO_PROXY no_proxy
export HF_ENDPOINT=https://hf-mirror.com
```

> 上面这一整套在 `udtca-config/common/case_common.py` 的 `SETUP_CMDS` 里，
> 走 `generate_and_run.py` 时**自动注入两边的 shell**，不用手敲。
> 只有自己手跑 `.sh` 或调试时才需要手动 export。

看卡是否空闲 / 是否有残留进程：

```bash
ixsmi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
pgrep -af 'run_qwen.*py|pt_elastic|torchrun'
ssh u210 "source /tmp/vl_setup.sh; ixsmi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader"
```

---

## 2. 启动（推荐：编排器）

上面的所有流程逻辑都被编写到 udtca-config 里面了。** 也就是说你只需要配置好 conda 环境，克隆 udtca 并初始化所有子模块，正确设置代码路径，然后运行 udtca/.../generate_and_run.py 文件就可以了跑起来了 **。

编排器会自己：清 tc → 按速率配 tc → 生成两节点的启动脚本 → scp 到 u210 → 同时拉起 →
回收日志到 `runtime_log/<case>/` → 复原 tc。

```bash
cd /data1/tangruijing/udtca
python udtca-config/qwenvl8b/generate_and_run.py
```

跑哪些 case 由 `test.json` 决定：

```bash
cat udtca-config/qwenvl8b/test.json
# [ {"index":1,"rate":"2gbit","bit-width":2,"baseline":"False"},   ← POLAR + bitscom
#   {"index":2,"rate":"2gbit","baseline":"True"},                 ← 普通 DP baseline
#   {"index":3,"rate":"1gbit","bit-width":2,"baseline":"False"},
#   {"index":4,"rate":"1gbit","baseline":"True"} ]
```

常用参数：

```bash
python udtca-config/qwenvl8b/generate_and_run.py --only 1,3      # 只跑指定 index
python udtca-config/qwenvl8b/generate_and_run.py --runtime-log off   # 冒烟，不归档
```

超参改 `udtca-config/qwenvl8b/default.json`（`max-steps` / `micro-batches` / `pp-size` …）。

跑起来后**别关终端**，或者挂后台：

```bash
nohup python udtca-config/qwenvl8b/generate_and_run.py > /tmp/matrix.log 2>&1 &
tail -f /tmp/matrix.log
```

qwen14b 换 `qwen14b/generate_and_run.py`，用法完全一样。

---

## 3. 启动（手动：直接跑 .sh）

不想走编排器时，两节点各跑一条（**先起 node1，再起 node0**）：

```bash
# 在 u210 上
ssh u210
cd /data1/tangruijing/udtca/experiments/qwenvl8b
bash 1_train_qwenvl8b_polar_dp_pp.sh

# 在 u62 上
cd /data1/tangruijing/udtca/experiments/qwenvl8b
bash 0_train_qwenvl8b_polar_dp_pp.sh
```

脚本里可覆盖的变量：`MASTER_ADDR`（必须指向 node0，即 `10.31.10.62`）、`MASTER_PORT`、
`NNODES`、`NPROC_PER_NODE`、`NCCL_SOCKET_IFNAME`。

四份脚本对应关系：

| 文件 | 含义 |
|---|---|
| `0/1_train_qwenvl8b_polar_dp_pp.sh` | POLAR + bitscom，PP=8 / TP=1 / DP=4 |
| `0/1_train_qwenvl8b_baseline_ddp_1f1b.sh` | 普通 DP（`--using-polar false --method none`） |
| `0/1` = node0 / node1 | 只有 `--node_rank` 和文件名不同 |

qwen14b 同理，文件名是 `*_qwen14b_*`。

---

## 4. 限速（可选）

`traffic_control*.sh` 在 node0 上给 `ens1f0` 的出向加 HTB 限速。
`generate_and_run.py` 会按 `test.json` 的 `rate` 自动建/删，不用手动调。

```bash
sudo bash traffic_control_exp1.sh start     # RATE=2gbit
sudo bash traffic_control_exp1.sh stop
```

看实时带宽（只读，不干扰训练）：

```bash
python udtca-config/utils/watch_bandwidth.py -i 5
python udtca-config/utils/watch_bandwidth.py --once
```

---

## 5. 日志与结果在哪里

每个 case 一个目录：`runtime_log/<4 位编号>/`

```
runtime_log/0008/
├── launch.sh            # 本次真正执行的启动脚本（复现用）
├── case.json            # 本次的全部参数
├── traffic_control.sh   # 本 case 的限速脚本
├── train_node0.log      # node0 全部 16 个 rank 的输出
├── train_node1.log
├── log/                 # tensorboard scalars / profiler trace
└── step_csv/            # 每步耗时
```

case 编号由 `runtime_log/` 里已有的最大编号自增，**不会覆盖**。

拍平 + 算对比（`case` 传目录名，不是编号）：

```bash
python udtca-config/utils/collect_case_logs.py runtime_log/0008   # 深层 log/ → tb_scalars/ tb_trace/ step_csv/

python udtca-config/utils/trace_processor.py \
    --runtime-log runtime_log --case 0008 0009 --csv /tmp/cmp.csv  # 每步耗时/吞吐对比
```

---

## 6. QwenVL8B 的 DataLoader

### 6.1 现在是怎么调的

训练入口 `experiments/qwenvl8b/run_qwenvl8b_polar_dp_pp.py`，全部数据逻辑集中在这三处：

| 位置 | 作用 |
|---|---|
| `class RandomVLDataset` | 造随机样本，形状与真实 Qwen3-VL 输入一致 |
| `def vl_collate(samples)` | 组 batch |
| `def get_dataloader(cfg, tokenizer, pp_size, tokens_per_image)` | 返回一个 `DataLoader` |

调用点在 `main()` 里：

```python
dataloader = get_dataloader(cfg, tokenizer, pp_size, tokens_per_image)
...
trainer = PolarParallel(..., dataloader=dataloader, ...)
```

dataLoader 的迭代和消费发生在 `polar-sgd` 的 `PolarParallel.train()` 里，训练脚本不参与：

```python
for batch_idx, batch in enumerate(self.dataloader):
    input_ids      = batch["input_ids"].to(self.device)
    labels         = batch["labels"].to(self.device)     # 只有最后一个 stage 用
    attention_mask = batch["attention_mask"].to(self.device)
    extra_inputs   = {k: v.to(self.device) for k, v in batch.items()
                      if k not in ("input_ids", "labels", "attention_mask")
                      and isinstance(v, torch.Tensor)}
    ...
    self.schedule.step(..., **extra_inputs)
```

### 6.2 batch 契约

DataLoader 产出的是一个 **dict**，键的含义：

| 键 | 形状 | 谁用 |
|---|---|---|
| `input_ids` | `[B, seq_len]` long | 第一个 stage 做 embedding；后续 stage 收到的是 hidden states，这个键被忽略 |
| `labels` | `[B, seq_len]` long | 只有最后一个 stage 用；别的 stage 会被丢掉 |
| `attention_mask` | `[B, seq_len]` long | 所有 stage |
| 其它任意 **Tensor** 键 | 自定义 | 原样透传给每个 stage 的 `forward` 作为关键字参数 |

几条硬约束：

- `B`（batch 的第一维）必须等于 `--per-device-batch-size`，且能被 `--micro-batches` 整除 —— pipeline 会沿 batch 维切成 micro-batch。
- 迭代出的 batch 数要 ≥ `--max-steps`，训练循环到 `max_steps` 就 break。
- 每个 rank 各拿各的 DataLoader（没有 sampler 帮你切分，自己按 `dp_rank` 分片）。

### 6.3 换成自己的 DataLoader

只改 `get_dataloader()`，返回满足上面契约的任意可迭代对象即可，**其余代码一行都不用动**。

```python
def get_dataloader(cfg, tokenizer, pp_size, tokens_per_image):
    dataset = MyDataset(cfg, tokenizer)          # 你的数据集
    return DataLoader(
        dataset,
        batch_size=cfg.per_device_batch_size,    # 必须 == --per-device-batch-size
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=my_collate,
    )
```

分片到各个 rank（按 DP 切，同一个 PP stage 的 4 个 DP 副本拿不同数据）。
mesh 是 `(dp, pp)` **行优先**的，`rank = dp_rank * pp_size + pp_rank`，所以：

```python
dp_rank  = dist.get_rank() // pp_size     # ← 不是 % dp_size
dp_world = world_size // pp_size
```

要稳的话直接从 mesh 拿，别自己算：

```python
dp_mesh = device_mesh["dp"]               # main() 里已经建好了
dp_rank, dp_world = dp_mesh.get_local_rank(), dp_mesh.size()
```

注意当前实现是每张卡各自建 DataLoader，rank 之间不会自动错开；
同一个 PP stage 的 DP 副本拿到**完全相同**的随机数据（`RandomVLDataset` 的种子只跟 `idx` 有关）。
换真实数据集时这里要自己按 `dp_rank` 切 shard。

新增一个自定义输入字段（比如 `extra_feat`）要两步：

1. collate 时把它放进 batch dict：`batch["extra_feat"] = torch.stack(...)`
2. 在训练脚本的 `custom_forward(...)` 签名里加同名参数：

```python
def custom_forward(input_ids_or_hidden, attention_mask=None,
                   pixel_values=None, image_grid_thw=None,
                   extra_feat=None):        # ← 加这里
```

`PolarParallel` 会把 batch 里所有非保留键**按名字**透传，签名对不上会直接报 `unexpected keyword argument`。

### 6.4 复用现成的 `vl_collate` 时要注意

图片相关的两个字段**必须在 dim 0 上 cat，不能 stack**：

```python
"pixel_values":   torch.cat([s["pixel_values"]   for s in samples], dim=0),  # [总图数*patches, patch_dim]
"image_grid_thw": torch.cat([s["image_grid_thw"] for s in samples], dim=0),  # [总图数, 3]
```

用默认 collate 会把 `image_grid_thw` 叠成 `[B, 1, 3]`，pipeline 沿 batch 维切开后变成 `[1, 1, 3]`，
模型里 `t, h, w = grid_thw[...]` 会报 `not enough values to unpack (expected 3, got 1)`。

另外 `image_grid_h` / `image_grid_w` 必须能被 `vision_config.spatial_merge_size`（Qwen3-VL 是 2）整除，
脚本启动时会检查。

### 6.5 用真实权重

默认是 `AutoConfig.from_pretrained` 取 config + `torch.device("meta")` 上随机初始化，
不下载权重。要加载真权重：

```bash
--init-from-pretrained
```

其余接口不用改。
