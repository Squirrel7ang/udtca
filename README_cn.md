# udtca 中文说明

本仓库是分布式训练通信优化的实验工作区，由两个子模块和一批实验脚本组成：

| 子模块 | 作用 |
|---|---|
| [`bitscom/`](bitscom/) | 低比特通信库。提供自定义 `torch.distributed` 后端 `lowbit` 和 `LowBitGroup` API，把全精度张量换成低比特量化后的表示再通信 |
| [`polar-sgd/`](polar-sgd/) | Polar SGD。在 DP/PP 之上做梯度预测与通信/计算重叠 |
| [`udtca-config/`](udtca-config/) | **实验编排器**。按矩阵生成启动脚本、下发到两个节点、限速、起训、回收日志 |

英文版见 [README.md](README.md)，那份讲的是**仓库结构和每个目录里有什么**；
本文不重复那些，只讲**怎么把环境装起来、怎么把实验跑起来**，以及
`udtca-config/` 怎么用。全部按步骤来，照着做即可。

---

## 0. 机器与拓扑

| 项 | 值 |
|---|---|
| 调度节点（MASTER） | **u62**，`10.31.10.62` |
| 训练节点 | **u210**，`10.31.10.210` |
| 仓库路径 | `/data1/tangruijing/udtca`，**两个节点上必须一致** |
| conda 环境 | `trj-test`（两节点都要有） |
| GPU | 单机 16 卡，两机共 32 卡 |
| 互联网卡 | `ens1f0`，接 100 Gb 八口交换机，**每台机器理论带宽 12.5 Gb/s** |
| 模型 | `Qwen/Qwen2.5-14B-Instruct` |
| 数据集 | `HuggingFaceFW/fineweb`（走 `HF_ENDPOINT=https://hf-mirror.com`） |
| 并行拓扑 | 2 节点 × 16 卡 = **DP=2, PP=8, TP=2**，1F1B 流水 |
| 训练入口 | `experiments/qwen14b/run_qwen14b_polar_dp_pp_tp.py` |

rank 分配：`rank = dp * (pp * tp) + pp_index * tp_size + tp_index`，
DP 维在节点间切分，所以 PP 的前几级在 u62、后几级在 u210。

---

## 1. 在天数（Iluvatar Corex）平台上装环境

天数平台的核心差异：**没有 CUDA/cuDNN，用的是 Corex 工具链**，
`nvidia-smi` 也不存在，要看卡得用 `ixsmi`。

本文使用 Corex **4.4.0**，装在 `/usr/local/corex-4.4.0`。

### 1.1 先确认 Corex 在位

```bash
ls /usr/local/corex-4.4.0/bin/ixsmi      # 显卡工具
ls /usr/local/corex-4.4.0/bin/clang++    # Corex 的 CUDA 编译器（编译扩展要用）
ixsmi --query-gpu=index,memory.used --format=csv,noheader
```

> u62 上 `ixsmi` 在 PATH 里；**u210 上不在**，要用绝对路径
> `/usr/local/corex-4.4.0/bin/ixsmi`。两台上都没有 `nvidia-smi`。

### 1.2 拉代码

本仓库带 **3 个子模块**，缺一个都跑不起来：

| 子模块 | 路径 | 上游地址 | 分支 |
|---|---|---|---|
| `bitscom` | `bitscom/` | `git@github.com:Squirrel7ang/bitscom.git` | `main` |
| `polar-sgd` | `polar-sgd/` | `git@github.com:Squirrel7ang/polar-sgd.git` | `main` |
| `udtca-config` | `udtca-config/` | `git@github.com:Squirrel7ang/udtca-config.git` | `master` |

> `.gitmodules` 里登记的就是上表这些地址，和子模块自己的 `origin` 一致。
> 如果哪天改了地址，记得 `git submodule sync --recursive` 让本地跟着更新。

新机器上直接克隆（`--recursive` 会连子模块一起拉）：

```bash
git clone --recursive git@github.com:Squirrel7ang/udtca.git
cd udtca
```

**已经有仓库、但子模块目录是空的**，补拉：

```bash
cd /data1/tangruijing/udtca
git submodule update --init --recursive
```

如果子模块拉不下来、或者提示 URL 对不上（`.gitmodules` 改过 URL 的情况），
先同步一遍再试：

```bash
git submodule sync --recursive
git submodule update --init --recursive
```

**两个节点上都要拉全**，并且要停在**同一个 commit** 上，否则两边的 bitscom /
polar-sgd 版本不一致，通信后端的行为会对不上。核对方法（两边输出应完全相同）：

```bash
git submodule status
# 每行形如： <hash> bitscom (heads/main)
#           <hash> polar-sgd (heads/main)
#           <hash> udtca-config (heads/master)
```

> 行首的符号：空格 = 已就位且与主库记录一致；`+` = 检出的 commit 和主库记录的
> 不一样（多半是子模块里又提交了新东西）；`-` = 子模块**没初始化**（目录空的，
> 或者 `.git/modules` 里的注册信息缺失）。出现 `-` 就跑 `git submodule init`
> 补注册，再 `git submodule update --init --recursive` 补代码。

### 1.3 建 conda 环境

```bash
conda create -n trj-test python=3.10 -y
conda activate trj-test
```

Python 用 **3.10**（当前验证过的版本）。天数的 PyTorch 要用
**Corex 版**的构建，不是 PyPI 上那个，装完 `torch.__version__` 应该是
`2.7.1+corex.4.4.0` 这种带 `corex` 后缀的：

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda)"
# 2.7.1+corex.4.4.0 10.2
```

`torchvision` 同理。这两个包走天数官方渠道装，剩下的依赖
（`datasets` / `transformers` / `sentencepiece` / `tensorboard` / `torch_tb_profiler` 等）
按平常的方式 `pip install` 即可，`polar-sgd/pyproject.toml` 里有清单，不赘述。

### 1.4 装 bitscom

bitscom 带 C++/CUDA 扩展，**必须用 Corex 的 clang++ 编**，
并且要给全头文件和库路径，否则会去找系统 `nvcc` 然后失败。

```bash
cd /data1/tangruijing/udtca/bitscom

CUDA_HOME=/usr/local/corex-4.4.0 \
BITSCOM_CUDA_COMPILER=/usr/local/corex-4.4.0/bin/clang++ \
NCCL_LIB_DIR=/usr/local/corex-4.4.0/lib64 \
NCCL_INCLUDE_DIR=/usr/local/corex-4.4.0/include \
/root/miniconda3/envs/trj-test/bin/python -m pip install -e . --no-build-isolation
```

> - `--no-build-isolation` 是必须的，不然 pip 会另起一个干净的构建环境，
>   里面没有 Corex 的 torch，编译直接失败。
> - `BITSCOM_CUDA_COMPILER` 是最关键的一个：**不指定它就会去调 `nvcc`**。
> - `NCCL_LIB_DIR` / `NCCL_INCLUDE_DIR` 在 u210 上尤其要加，
>   不然链接阶段 conda 的 `compiler_compat/ld` 找不到 `-lnccl`。
> - 编译过程会在仓库根生成 `compile_commands.json`（给 clangd 用），
>   以及 `build/` 中间目录 —— 这些都已经在 `.gitignore` 里，不用管。

装完自检：

```bash
python -c "import bitscom; print(bitscom.__file__)"
ls /root/miniconda3/envs/trj-test/lib/python3.10/site-packages/__editable__.bitscom-0.1.0.pth
```

### 1.5 装 polar-sgd

纯 Python 包，不用编译：

```bash
cd /data1/tangruijing/udtca/polar-sgd
/root/miniconda3/envs/trj-test/bin/python -m pip install -e . --no-build-isolation
```

> polar-sgd 现在在 **`main` 分支**上（原来的 `refactor` 分支已经合并进
> `main`，两者指向同一个 commit，`refactor` 分支可以不用了）。
> 两个节点要停在**同一个 commit**，否则通信/重叠逻辑会对不上。
>
> 这条同样适用于 bitscom —— 它是需要编译的，两边版本不一致会直接连不上。

自检：

```bash
python -c "import psgd; print(psgd.__file__)"
ls /root/miniconda3/envs/trj-test/lib/python3.10/site-packages/__editable__.psgd-0.1.0.pth
```

### 1.6 装 flash-attn（Qwen14B 需要）

Qwen14B 的入口脚本里 `--use-flash-attn` 是 **默认为真且无法关闭**的
（`store_true` + `default=True`），**没装会在 import 阶段直接报
`ImportError: FlashAttention2 has been toggled on`**。

天数用的是预编译好的 Corex 轮子（cp310），六个一起装：

```bash
cd /root/apps/py3.10/flash_attn          # u62 上的位置
/root/miniconda3/envs/trj-test/bin/python -m pip install \
    flash_attn-2.6.3+corex.4.4.0-cp310-cp310-linux_x86_64.whl \
    fused_dense_lib-0.1+corex.4.4.0-cp310-cp310-linux_x86_64.whl \
    fused_softmax_lib-0.1+corex.4.4.0-cp310-cp310-linux_x86_64.whl \
    rotary_emb-0.1+corex.4.4.0-cp310-cp310-linux_x86_64.whl \
    xentropy_cuda_lib-0.1+corex.4.4.0-cp310-cp310-linux_x86_64.whl \
    dropout_layer_norm-0.1+corex.4.4.0-cp310-cp310-linux_x86_64.whl
```

> **u210 上没有 `/root/apps` 这个目录**。把整个 `flash_attn/` 目录
> `scp` 过去（比如放到 `/root/flash_wheels/`）再装即可，轮子是通用的。

自检：

```bash
python -c "import flash_attn; print(flash_attn.__version__)"   # 2.6.3
```

### 1.7 两节点一致性检查

装完在**两台**上分别跑一遍：

```bash
P=/root/miniconda3/envs/trj-test/bin/python
$P -c "import torch, bitscom, psgd, flash_attn, transformers, datasets; \
print(torch.__version__); print(bitscom.__file__); print(psgd.__file__)"
```

两边的 `torch.__version__` 要一样，`bitscom` / `psgd` 指向的 commit 也要一样。

### 1.8 环境变量：两台机器的 `~/.bashrc`

Corex 相关的环境变量写在 **`/root/.bashrc`** 里，**两台机器必须设成一样**，
否则编译和运行都会出问题（典型症状：编译时找不到 `nvcc`/头文件，或者运行时
`libcorex.so` 加载不到）。

**u62 的 `~/.bashrc` 末尾：**

```bash
# Corex 4.4.0 Environment for Megatron-LM
export COREX_PATH=/usr/local/corex-4.4.0
export CPATH=$COREX_PATH/include:$CPATH
export LIBRARY_PATH=$COREX_PATH/lib64:$LIBRARY_PATH
export LD_LIBRARY_PATH=$COREX_PATH/lib64:$LD_LIBRARY_PATH

# Triton Specific
export TRITON_CUDA_SYSROOT=$COREX_PATH

# Training Optimization
export CUDA_DEVICE_MAX_CONNECTIONS=1

export CUDA_HOME=/usr/local/corex-4.4.0
export BITSCOM_CUDA_COMPILER=/usr/local/corex-4.4.0/bin/clang++
export HF_ENDPOINT=https://hf-mirror.com
```

**u210 的 `~/.bashrc`** 里也有 Corex 相关设置，但**内容不一样**：

```bash
export PATH=/usr/local/corex/bin:/usr/local/corex/lib64/python3/dist-packages/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/corex/lib64
export PATH=/usr/local/corex/bin:$PATH
export PYTHONPATH=/usr/local/corex/lib64/python3/dist-package
export PATH=/usr/local/corex/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

export HF_ENDPOINT=https://hf-mirror.com
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY no_proxy NO_PROXY

export CUDA_HOME=/usr/local/corex-4.4.0
export BITSCOM_CUDA_COMPILER=/usr/local/corex-4.4.0/bin/clang++
```

**两台的关键差异：**

| 变量 | u62 | u210 | 作用 |
|---|---|---|---|
| `COREX_PATH` | ✅ | ❌ 缺 | Corex 安装根目录，`CPATH` 等都基于它 |
| `CPATH` | ✅ | ❌ 缺 | 编译扩展时找头文件 |
| `LIBRARY_PATH` | ✅ | ❌ 缺 | **链接期**找库，缺了会报 `cannot find -lnccl` |
| `TRITON_CUDA_SYSROOT` | ✅ | ❌ 缺 | Triton 编 kernel 用的 sysroot |
| `CUDA_DEVICE_MAX_CONNECTIONS=1` | ✅ | ❌ 缺 | 训练性能相关 |
| `PYTHONPATH` | ❌ 缺 | ✅ | 指向 corex 的 dist-package |
| `LD_LIBRARY_PATH` | ✅ | ✅ | 运行期加载 corex 动态库 |

> **`ssh u210` 是非交互 shell，默认不 source `~/.bashrc`**，所以上面这些在远程
> 直接 `ssh` 过去是**不会有**的。另外 u62 那份 `.bashrc` 开头有
> `[ -z "$PS1" ] && return` 之类的非交互保护，直接 `source` 也会提前返回。
>
> 走 `udtca-config` 不用管这些：`case_common.SETUP_CMDS` 会先绕过保护 source 一遍
> `~/.bashrc`，再把上表里 u62 有、u210 没有的那几个变量**在两边显式补齐**，
> 保证两个节点环境一致。
>
> 手工在 u210 上跑实验时，记得先 `. ~/.bashrc`，或者按上表把缺的变量 export 一遍。

---

## 2. 手工跑一个 Qwen14B 实验

想先不借助编排器、直接手动起一次，按这一节做。日常跑实验用第 3 节的
`udtca-config` 更省事。

### 2.1 关键：u62 必须是 MASTER

`torchrun` 的 `node_rank=0` 会 **bind** `MASTER_ADDR:MASTER_PORT` 作为
rendezvous 服务端，`node_rank=1` 作为客户端连过去。

**如果 `MASTER_ADDR` 写的是 u210 的地址，u62 会绑不上（`EADDRNOTAVAIL`），
然后两边永远干等，一个 worker 都起不来。** 所以：

| | u62 | u210 |
|---|---|---|
| 跑哪个脚本 | `0_train_*`（`--node_rank=0`） | `1_train_*`（`--node_rank=1`） |
| 角色 | **MASTER_NODE** | 客户端 |
| `MASTER_ADDR` | `10.31.10.62` | `10.31.10.62` |
| `MASTER_PORT` | polar 用 `29500`，baseline 用 `29501` | 同左 |

> ⚠️ **`experiments/qwen14b/` 下那 4 个脚本的 `MASTER_ADDR` 默认值是
> `10.31.10.210`（u210），不是 u62。** 手工跑的时候必须自己覆盖：
>
> ```bash
> MASTER_ADDR=10.31.10.62 bash experiments/qwen14b/0_train_qwen14b_polar_dp_pp_tp.sh
> ```
>
> 走 `udtca-config`（第 3 节）时不需要管 —— 编排器会在起训前
> `export MASTER_ADDR=10.31.10.62`，脚本里是 `${MASTER_ADDR:-...}` 的写法，
> 环境变量优先。

### 2.2 两个节点的启动脚本

| 脚本 | 节点 | 跑什么 |
|---|---|---|
| `0_train_qwen14b_polar_dp_pp_tp.sh` | u62 | POLAR + bitscom，node_rank=0 |
| `1_train_qwen14b_polar_dp_pp_tp.sh` | u210 | POLAR + bitscom，node_rank=1 |
| `0_train_qwen14b_baseline_ddp_1f1b_tp.sh` | u62 | 纯 DP baseline，node_rank=0 |
| `1_train_qwen14b_baseline_ddp_1f1b_tp.sh` | u210 | 纯 DP baseline，node_rank=1 |

两者都是 `torchrun --nnodes=2 --nproc_per_node=16`，区别只在
**走不走 bitscom 的 `lowbit` 后端**。

脚本里写死的超参是 `--max-steps 200`、`--bitwidth 4`、`--micro-batches 32`、
`--seq-len 256`、`--pp-size 8`、`--tp-size 2`、`--lr 2e-4`。
**通过 `udtca-config` 跑时这些会被 `default.json` / `test.json` 覆盖**
（当前配置是 `max-steps=30`、`bit-width=2`）。手工跑想改就直接编辑脚本。

### 2.3 起训

u62 上开一个终端：

```bash
cd /data1/tangruijing/udtca
conda activate trj-test
MASTER_ADDR=10.31.10.62 bash experiments/qwen14b/0_train_qwen14b_polar_dp_pp_tp.sh
```

u210 上开另一个终端：

```bash
ssh u210
cd /data1/tangruijing/udtca
conda activate trj-test
MASTER_ADDR=10.31.10.62 bash experiments/qwen14b/1_train_qwen14b_polar_dp_pp_tp.sh
```

> 这 4 个脚本**自己不 export Corex 的环境变量**（`COREX_PATH` / `CPATH` /
> `LIBRARY_PATH` / `TRITON_CUDA_SYSROOT` / `BITSCOM_CUDA_COMPILER` 等），
> 它们都依赖你当前的 shell 已经把这些设好。所以手工跑之前请确认：
>
> - 在 u62 上，你的交互 shell 已经 source 过 `~/.bashrc`（登录时自动）；
> - 在 u210 上，`ssh u210` 是**非交互 shell**，默认什么都不 source，
>   需要在跑之前 `. ~/.bashrc`，或者自己把那一组 Corex 变量 export 一遍。
>
> 上面这组变量两台机器的 `~/.bashrc` 里都有，但**内容不一样**，
> 明细见 [1.8 环境变量](#18-环境变量两台机器的-bashrc)。
>
> 走 `udtca-config` 时不用操心：编排器会把两台机器的环境对齐
> （`case_common.SETUP_CMDS` 里先强制 source 一遍 `~/.bashrc`，再把 u62 有、
> u210 缺的那几个 Corex 变量显式补齐）。

### 2.4 限速

实验要比较不同网络带宽下的表现，用 `tc` 把 `ens1f0` 的出向流量卡住。
脚本在 `udtca-config` 自动生成；手工做就是：

```bash
# 限到 2 Gbit/s
tc qdisc replace dev ens1f0 root handle 1: htb default 1
tc class replace dev ens1f0 parent 1: classid 1:1 htb rate 2gbit

# 恢复不限速
tc qdisc del dev ens1f0 root
```

> - **`tc` 只限出向（egress），入向不限。**
> - 限速规则**要在两个节点上都执行**。
> - 跑完记得 `tc qdisc del`，否则会给共享机器留一条限速规则。
> - 不限速时两机之间就是交换机给的理论带宽 **12.5 Gb/s**。

---

## 3. 用 `udtca-config` 编排（推荐）

`udtca-config/` 不是训练代码，是**调度器**：读配置矩阵 → 生成两节点的启动脚本 →
下发到 u210 → 两边限速 → 起训 → 回收日志 → 归档到 `runtime_log/<编号>/`。

**只需要在 u62 上跑一条命令**，它会同时操控 u62（node 0）和 u210（node 1）。

### 3.1 前置检查

```bash
# 1) 免密 ssh 通
ssh u210 hostname

# 2) 两节点的 GPU 都空闲（脚本自己也会查，占用会直接中止）
ixsmi --query-gpu=index,memory.used --format=csv,noheader
ssh u210 '/usr/local/corex-4.4.0/bin/ixsmi --query-gpu=index,memory.used --format=csv,noheader'

# 3) 两节点都装好了 bitscom / polar-sgd（见第 1 节）
```

### 3.2 目录里有什么

| 文件 | 作用 |
|---|---|
| `generate_and_run.py` | **主编排入口**。按 `test.json` 逐条跑，按每条的 `baseline` 标记分派 polar / baseline |
| `generate_and_run_baseline.py` | 只跑 baseline 的入口，同时对外提供 `run_baseline_case()` 供上面那个 import |
| `case_common.py` | 核心库：两节点命令封装、`SETUP_CMDS`、case 编号与 `case.json` 读写、GPU 检查、日志回收 |
| `collect_case_logs.py` | 把训练侧写的深层 `./log/...` 拍平成 `tb_scalars` / `tb_trace` / `step_csv` |
| `trace_processor.py` | 从 `runtime_log/` 提取每个 case 的平均步时和吞吐，做 POLAR vs baseline 对比 |
| `watch_bandwidth.py` | 实时监测两机之间的网卡带宽（只读，不干扰训练） |
| `download_fineweb.py` | 按需下载 fineweb parquet 数据集到指定目录 |
| `build_comm_opt_data.py` | 把实验数据整理成交付目录树并打包成 zip |
| `default.json` | 默认超参，没在 `test.json` 里写的字段都从这里继承 |
| `test.json` | 实验矩阵（要跑哪些 case） |

### 3.3 配置格式

`default.json` —— 默认值：

```json
{
    "baseline": "False",
    "index": 0,
    "pp-size": 8,
    "tp-size": 2,
    "rate": "5gbit",
    "bit-width": 2,
    "micro-batches": 32,
    "max-steps": 30,
    "seq-len": 256
}
```

`test.json` —— 一个数组，**按顺序跑**，每条覆盖若干字段：

```json
[
    { "index": 1, "rate": "2gbit", "bit-width": 2, "baseline": "False" },
    { "index": 2, "rate": "2gbit", "baseline": "True" },
    { "index": 3, "rate": "1gbit", "bit-width": 2, "baseline": "False" },
    { "index": 4, "rate": "1gbit", "baseline": "True" }
]
```

字段说明：

| 字段 | 含义 |
|---|---|
| `index` | 这一条的编号，决定生成脚本叫 `*_exp{index}.sh` 和限速脚本叫 `traffic_control_exp{index}.sh` |
| `rate` | `tc` 限速值，如 `10gbit` / `5gbit` / `2gbit` / `1gbit` |
| `baseline` | **字符串** `"True"` / `"False"`。为真走纯 DP baseline，为假走 POLAR + bitscom |
| `bit-width` | bitscom 量化位宽，支持 1/2/4/8 |
| `max-steps` | 训练步数 |

> `baseline` 是**字符串不是布尔**，代码里用 `as_bool()` 解析。
> 这是踩过的坑：`"False"` 在 Python 里是 truthy，直接用会导致所有 case 都被当成 baseline。

**建议同一档网速下让 POLAR 和 baseline 紧挨着跑**（如上面的排法），
减少机器状态漂移对对比的干扰。**先跑网速快的**，慢的后面再说。

### 3.4 跑起来

在 **u62** 上：

```bash
cd /data1/tangruijing/udtca
python udtca-config/generate_and_run.py
```

跑之前会先检查两个节点的 GPU 占用，**被占用就直接报错退出，不会启动任何训练**。

### 3.5 先冒烟测试

正式跑之前建议先验证环境能跑通。用 `--runtime-log off`：

```bash
python udtca-config/generate_and_run.py --runtime-log off
```

这个开关关掉时：不占 case 编号、不建 `runtime_log/000N/`、不写 `case.json`、
不归档任何脚本和日志，**训练输出直接打在终端**（方便看实时进度）。
训练本身照跑，所以验证环境是准的。

也可以用环境变量代替命令行开关（优先级：命令行 > 环境变量 > 默认 on）：

```bash
UDTCA_RUNTIME_LOG=0 python udtca-config/generate_and_run.py
```

### 3.6 只跑 baseline

```bash
python udtca-config/generate_and_run_baseline.py            # 只跑 test.json 里 baseline=True 的
python udtca-config/generate_and_run_baseline.py --runtime-log off
```

两个入口共用同一套 case 编号和同一个 `runtime_log/case.json`。

### 3.7 产物：`runtime_log/`

```
<仓库根>/runtime_log/
├── case.json              每个 case 的启动脚本与启动参数
├── environment.json       机器与网络环境说明（交数据时用）
├── 0001/
│   ├── launch.sh                       完整自包含的复现脚本
│   ├── 0_train_qwen14b_exp1.sh         node0(u62) 实际跑的
│   ├── 1_train_qwen14b_exp1.sh         node1(u210) 实际跑的（从远程拉回）
│   ├── traffic_control_exp1.sh         网络限速脚本
│   ├── train_node0.log                 node0 的 torchrun 全部输出
│   ├── train_node1.log                 node1 的 torchrun 全部输出（从远程拉回）
│   ├── tb_scalars/                     tensorboard 事件文件
│   ├── tb_trace/                       profiler trace
│   ├── step_csv/                       每步耗时 CSV
│   └── debug_logs/                     调试信息
├── 0002/ ...
```

`case` 编号按**实际运行顺序**递增（四位，跨多次运行连续），
**不是** `test.json` 里的 `index`。

**训练日志**：两个节点的 torchrun 输出**各自重定向**到 case 目录下的
`train_node0.log` / `train_node1.log`，不刷屏。u210 的那份先写在 u210 上，
跑完由 rsync 拉回同一个 case 目录。**终端上只看到 u62 的输出**
（PP 最后一级在 u62 上有 rank 14/15，loss 照样看得到）。

想看实时输出就另开一个终端：

```bash
tail -f runtime_log/0001/train_node0.log
```

### 3.8 `launch.sh` —— 不依赖编排器的复现

每个 case 目录里的 `launch.sh` 是**自包含**的：

```bash
# node 0 (u62)
bash runtime_log/0001/launch.sh 0
# node 1 (u210)
bash runtime_log/0001/launch.sh 1
```

里面包含环境准备（shell 初始化 → Corex/CUDA 变量 → conda 环境 → HF 镜像）、
限速命令说明、以及两个节点各自的完整 `torchrun` 命令。

### 3.9 中断（Ctrl-C）

收到 SIGINT/SIGTERM 时编排器会：

1. 杀掉本机训练进程，并 **ssh 过去把 u210 上的训练进程也杀掉**
   —— 中断时 ssh 链路断了，但 u210 上的 torchrun 不会跟着死，会一直占着卡；
2. **撤销两台机器上的 `tc` 限速** —— 不撤会给共享机器留一条限速规则。

清理用的是「解释器路径 + 训练入口脚本名」精确匹配，**不会误杀共享机器上别人的任务**。

### 3.10 辅助脚本

```bash
# 实时看两机带宽（每 1 秒一行，Ctrl-C 退出）
python udtca-config/watch_bandwidth.py

# 汇总每个 case 的平均步时 / 吞吐，并做 POLAR vs baseline 同网速对比
python udtca-config/trace_processor.py
python udtca-config/trace_processor.py --case 0001 0002 --csv out.csv

# 下载 fineweb 数据集（先 --dry-run 看计划）
python udtca-config/download_fineweb.py --dry-run
python udtca-config/download_fineweb.py --target-gb 20 --dest /data1/tangruijing/fineweb_data

# 把实验数据整理成交付目录树并打包 zip
python udtca-config/build_comm_opt_data.py --dry-run
python udtca-config/build_comm_opt_data.py --dest /data1/tangruijing/comm-opt-data
```

> `build_comm_opt_data.py` 遵守两条硬规则：**只做复制 / 新建，绝不移动、
> 删除或修改任何源文件**；目标目录下已存在且大小一致的文件会跳过（可增量重跑）。

---

## 4. 踩过的坑

| 现象 | 原因 | 解决 |
|---|---|---|
| 两边都不起 worker，一直干等 | `MASTER_ADDR` 指向了 u210，u62 的 `node_rank=0` 绑不上 | `MASTER_ADDR` 必须是 `10.31.10.62`（u62 是 MASTER） |
| 远程命令里的变量全是空串 | `ssh host '/bin/bash -c "...$VAR..."'` 两层引号让远程登录 shell 先展开了 | 整条脚本 base64 后送过去（`case_common.ssh_execute` 已这么做） |
| `bash: command not found` / 找不到 `bash` | `export PATH='/root/miniconda3/bin:$PATH'` 用了单引号，`$PATH` 没展开，PATH 变成字面量 | 必须用**双引号** |
| `launch.sh` 跑到 `. ~/.bashrc` 就静默退出 | `.bashrc` 里引用了未定义变量（`debian_chroot`），`set -u` 下非交互 shell 直接退出，`set +e` 拦不住 | source 前后把 `-e` `-u` 关掉再恢复（已写进 `SETUP_CMDS`） |
| `ImportError: FlashAttention2 has been toggled on` | `--use-flash-attn` 是 `store_true, default=True`，关不掉，而 flash_attn 没装 | 按 1.6 节装 Corex 版 flash-attn |
| bitscom 编译失败，找不到 `nvcc` | 没指定 `BITSCOM_CUDA_COMPILER` | 显式设成 `$COREX_PATH/bin/clang++` |
| u210 链接期找不到 `-lnccl` | conda 的 `compiler_compat/ld` 搜索路径里没有 Corex 的库 | 编译时加 `NCCL_LIB_DIR` / `NCCL_INCLUDE_DIR` |
| 所有 case 都被当成 baseline 跑 | `"False"` 在 Python 里是 truthy | 用 `as_bool()` 解析，不要直接 `if cfg["baseline"]` |
| `step_csv` 里每步有 2 行重复 | TP=2 时 PP 最后一级的 2 个 rank 都会写入 | 读的时候按 step 去重（`trace_processor.py` 已处理） |
| `log/` 目录几十 GB | 早期实验留下的旧结构（`log/True/...`） | 新实验产物一律走 `runtime_log/`，互不干扰 |

---

## 5. README 索引

- [顶层 README（英文）](README.md) / **本文件（中文）**
- [bitscom README](bitscom/README.md) / [中文](bitscom/README_cn.md)
- [polar-sgd README](polar-sgd/README.md) / [中文](polar-sgd/README_cn.md)
- [udtca-config README](udtca-config/README.md)
- [Qwen14B 实验 README](experiments/qwen14b/README.md)
- [量化实验 README](experiments/quantization/README.md) / [中文](experiments/quantization/README_cn.md)
