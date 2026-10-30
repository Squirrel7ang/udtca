# 云边 300 系列部署说明

两台机器：云侧 **powerone1** (10.205.200.14)、边侧 **powerone2** (10.205.200.16)，
各 16 张 Iluvatar Corex 卡，通过 1 Gbps 以太网互联。

## 1. 宿主机准备

- 操作系统：Ubuntu 22.04，驱动/Corex 运行时装在宿主机 `/usr/local/corex`。
- `ixsmi` 能看到 16 张卡：
  ```bash
  ixsmi
  ```
- 两台机器都要有 docker，并且互相能免密 ssh（云侧 → 边侧即可）：
  ```bash
  ssh powerone2 hostname      # 云侧执行，不提示密码
  ```

## 2. 导入镜像并起容器

镜像由厂商提供（tar 包），先 `docker load`：

```bash
docker load -i corex-4.5.0.20261008-ubuntu22.04-llm-py3.12-v2.tar
docker images | grep corex
```

起容器（两台机器一样，**只需改 `NCCL_SOCKET_IFNAME`**）：

```bash
docker run -d --name trj-udtca \
  --privileged --cap-add ALL \
  --network host --ipc host \
  --restart unless-stopped \
  -v /dev:/dev \
  -v /lib/modules:/lib/modules \
  -v /home:/home \
  -v /data:/data \
  -v /usr/src:/usr/src \
  -v /etc/localtime:/etc/localtime:ro \
  -v /usr/local/corex/lib64/libcuda.so:/usr/local/corex/lib64/libcuda.so \
  -v /usr/local/corex/lib64/libcuda.so.1:/usr/local/corex/lib64/libcuda.so.1 \
  -v /usr/local/corex/lib64/libixml.so:/usr/local/corex/lib64/libixml.so \
  -v /usr/local/corex/bin/ixsmi:/usr/local/corex/bin/ixsmi \
  -e NCCL_SOCKET_IFNAME=ens120f0 \
  -e UDTCA_NODE0_IP=10.205.200.14 \
  -e UDTCA_NODE1_HOST=10.205.200.16 \
  -w /usr/local/corex \
  corex:4.5.0.20261008-ubuntu22.04-llm-py3.12-v2 \
  bash -c "ln -sfn /data /data1; exec sleep infinity"
```

要点：

| 项 | 值 | 为什么 |
|----|----|--------|
| `--privileged` + `--cap-add ALL` | 是 | 需要直接访问 Corex 设备 |
| `-v /dev:/dev` | 是 | 容器内要看到 `/dev/ix*` 全部 16 张卡 |
| `--network host` | 是 | 跨机 NCCL 走 Socket，容器必须直接用宿主机网卡 |
| `--ipc host` | 是 | 共享内存通信 |
| `/usr/local/corex/...` 单独挂 | 是 | 容器和宿主机驱动必须同一份（`libcuda.so`、`ixsmi` 等） |
| `-v /data:/data` | 是 | 代码、数据集缓存、日志都在这里 |
| `ln -sfn /data /data1` | 是 | 代码里两种路径写法都有（`/data1/tangruijing/...`） |
| `NCCL_SOCKET_IFNAME` | 云 `ens120f0` / 边 `enx00e04c680003` | 跨机建连必须钉到对的那块网卡 |

`--ipc host` 时不需要 `--shm-size`。

验证：

```bash
docker exec trj-udtca bash -c 'ixsmi | head; python3 -c "import torch; print(torch.__version__)"'
# 期望：16 张卡；torch 2.13.0+corex
```

## 3. 代码

仓库在 `/data1/tangruijing/udtca`（容器内是 `/data/tangruijing/udtca`，同一份），
**两台机器保持同一个提交**，子模块 `udtca-config`、`polar-sgd`、`bitscom` 版本一致：

```bash
cd /data1/tangruijing/udtca
git status && git submodule status
```

训练入口和脚本都在 `udtca-config/qwen14b/` 里：

| 用途 | 位置 |
|------|------|
| 双机 32 卡训练（PP=8 TP=2） | `cloud/run_xnode.sh`（云侧）、`edge/run_xnode.sh`（边侧） |
| 双机用法说明 | `XNODE-USAGE.md` |
| 跨机连通性测试 | `cloud/run_nccl_xnode.sh`、`edge/run_nccl_xnode.sh` + `nccl_xnode_test.py` |
| 单机 8 卡 | `cloud/run_single_node_8gpu.sh`、`edge/run_single_node_8gpu.sh` |
| 宿主机一键跑两台容器 | `run_docker.py`（在 powerone1 宿主机执行） |
| 训练入口 | `/data1/tangruijing/udtca/experiments/qwen14b/run_qwen14b_polar_dp_pp_tp.py` |

快速开始（两边容器内，边侧先起）：

```bash
MODE=mixed bash /data1/tangruijing/udtca/udtca-config/qwen14b/edge/run_xnode.sh    # 边侧
MODE=mixed bash /data1/tangruijing/udtca/udtca-config/qwen14b/cloud/run_xnode.sh   # 云侧
```

## 4. 训练数据

- 数据集：`HuggingFaceFW/fineweb`（`--dataset-config sample-10BT`），用
  `load_dataset(..., streaming=True)` 流式读取，不落原始 parquet。
- 外网通过镜像站：脚本里设了 `HF_ENDPOINT=https://hf-mirror.com`，并 unset 了所有 proxy。
- 每个节点第一次跑时会把 tokenize 好的样本缓存成本地 `.pt`：

  ```
  experiments/qwen14b/cache/node<rank>_<hostname>_HuggingFaceFW_fineweb_sample-10BT_seq256_samples<N>_labels_unshifted_v2.pt
  ```

  `N = max_steps × per_device_batch_size`；换 `max_steps` 会自动复用样本数更大的旧缓存，
  所以**每台机器都要有一份自己的缓存**（文件名带节点号，不会互相串用）。
- 之后跑只要缓存命中就不再联网。想强制重建：

  ```bash
  # 删掉对应 .pt，或给训练加 --refresh-dataset-cache
  ```

- 模型权重：默认 `--init-from-pretrained` 关闭，模型按 `Qwen2.5-14B-Instruct` 的
  config 随机初始化，只需下载 config/tokenizer（几十 MB）。要真权重就加
  `--init-from-pretrained`，同样走 `HF_ENDPOINT` 镜像。

> 断网/镜像站不通时，训练会卡在 `load_dataset` 上——先确认缓存在不在。
