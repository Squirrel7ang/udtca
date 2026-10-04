#!/usr/bin/env python3
import os
import argparse
import sys
import socket
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _path in (
    _REPO_ROOT / "polar-sgd" / "src",
    _REPO_ROOT / "bitscom" / "python",
):
    if _path.exists():
        sys.path.insert(0, str(_path))

"""
Polar-SGD pretraining for Qwen3-VL models with DP+PP parallelism.
Experiment entrypoint for udtca/experiments/qwenvl8b.

和 qwen14b 的差别：
  * 模型是 Qwen3-VL（视觉塔 + 语言模型），PP 切分时视觉塔留在 stage 0；
  * 数据先用随机张量搓，只保留 get_dataloader(cfg, tokenizer, pp_size) 接口；
  * 权重只从网上取 config，本地随机初始化；--init-from-pretrained 预留真权重入口。
"""

from psgd.parallelism.polar.wrapper import PolarParallel

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.pipelining import Schedule1F1B
from torch.distributed.device_mesh import init_device_mesh
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoTokenizer, AutoModelForImageTextToText
from typing import List, Optional
from dataclasses import dataclass

# -----------------------------
# Training Configuration
# -----------------------------

DEFAULT_MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"


def import_bitscom():
    try:
        import bitscom

        return bitscom
    except ImportError:
        bitscom_python = _REPO_ROOT / "bitscom" / "python"
        if bitscom_python.exists():
            sys.path.insert(0, str(bitscom_python))
        import bitscom

        return bitscom


def create_lowbit_dp_group(dp_group):
    """Create lowbit backend DP groups in a globally consistent order."""
    my_ranks = tuple(int(r) for r in dist.get_process_group_ranks(dp_group))
    gathered = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, my_ranks)
    dp_rank_groups = sorted({tuple(int(r) for r in ranks) for ranks in gathered})

    selected_group = None
    for ranks in dp_rank_groups:
        group = dist.new_group(ranks=list(ranks), backend="lowbit")
        if ranks == my_ranks:
            selected_group = group

    if selected_group is None:
        raise RuntimeError(f"failed to create lowbit DP group for ranks={my_ranks}")
    return selected_group


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value!r}")


@dataclass
class TrainConfig:
    model_name: str = DEFAULT_MODEL_NAME
    tokenizer_name: str = None
    # 随机数据：一个样本 = 一段 token 序列 + 一张（或多张）图
    seq_len: int = 256
    per_device_batch_size: int = 32
    images_per_sample: int = 1
    # 图像在 patch 网格里的高/宽（不是像素）。64x64 的图 -> 4x4 个 16x16 patch。
    image_grid_h: int = 4
    image_grid_w: int = 4
    seed: int = 1234
    # 训练超参
    grad_accum_steps: int = 4
    lr: float = 2.0e-4
    warmup_ratio: float = 0.02
    max_tokens: int = 0
    max_steps: int = 10
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    clip_norm: float = 1.0
    log_interval: int = 10
    save_interval: int = 1000
    save_dir: str = "checkpoints/qwen3_vl_8b_instruct"
    num_workers: int = 2
    use_flash_attn: bool = False
    bf16: bool = True
    fp16: bool = False
    activation_checkpointing: bool = False
    init_from_pretrained: bool = False
    # 由 config 推导出来的常量
    vocab_size: int = 0
    image_token_id: int = 0
    patch_dim: int = 0


# -----------------------------
# 随机 VL 数据集
# -----------------------------
class RandomVLDataset(Dataset):
    """先搓随机数据，形状和真实 Qwen3-VL 输入一致。

    每个样本：
      input_ids        [seq_len]              前 n_image_tokens 个位置是 image_token_id
      labels           [seq_len]
      attention_mask   [seq_len]
      pixel_values     [images_per_sample * patches_per_image, patch_dim]
      image_grid_thw   [images_per_sample, 3]

    patches_per_image = t * grid_h * grid_w；t 恒为 1。
    图片在语言侧占用的 token 数 = (grid_h / merge) * (grid_w / merge)（merge 在 build 时填进 cfg）。
    """

    def __init__(self, cfg: TrainConfig, num_samples: int, tokens_per_image: int):
        self.cfg = cfg
        self.num_samples = int(num_samples)
        self.tokens_per_image = int(tokens_per_image)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        cfg = self.cfg
        # 每个样本一个独立且可复现的 generator，避免依赖 DataLoader 的 worker 顺序
        g = torch.Generator().manual_seed(int(cfg.seed) + int(idx))

        # 先按 [0, vocab_size-1) 抽，再把 >= image_token_id 的抬一位：
        # 这样随机 id 里**不会**出现 image_token_id 本身。
        # 否则非图片位置也会撞上这个 id，语言侧数出来的 image token 数会比视觉塔
        # 产出的 feature 多，forward 里的 masked_scatter 一致性检查会直接报错。
        input_ids = torch.randint(
            0, cfg.vocab_size - 1, (cfg.seq_len,), generator=g, dtype=torch.long
        )
        input_ids[input_ids >= cfg.image_token_id] += 1

        n_image_tokens = self.tokens_per_image * cfg.images_per_sample
        if n_image_tokens > cfg.seq_len:
            raise ValueError(
                f"images_per_sample={cfg.images_per_sample} 需要 "
                f"{n_image_tokens} 个 image token，超过 seq_len={cfg.seq_len}"
            )
        # 把开头的若干位置换成 image placeholder，让语言侧的 masked_scatter 有处可填
        input_ids[:n_image_tokens] = cfg.image_token_id

        patches_per_image = cfg.image_grid_h * cfg.image_grid_w
        pixel_values = torch.randn(
            cfg.images_per_sample * patches_per_image, cfg.patch_dim, generator=g
        )
        image_grid_thw = torch.tensor(
            [[1, cfg.image_grid_h, cfg.image_grid_w]] * cfg.images_per_sample,
            dtype=torch.long,
        )

        return {
            "input_ids": input_ids,
            "labels": input_ids.clone(),
            "attention_mask": torch.ones(cfg.seq_len, dtype=torch.long),
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
        }


def vl_collate(samples: List[dict]) -> dict:
    """默认 collate 会把 image_grid_thw 叠成 [B, images, 3]，

    而模型和 pipeline 都要求它是 [总图片数, 3] 并且在 dim 0 上按 micro-batch 切开，
    所以图片相关的两个字段沿 dim 0 拼接，文本字段照常 stack。
    """
    return {
        "input_ids": torch.stack([s["input_ids"] for s in samples], dim=0),
        "labels": torch.stack([s["labels"] for s in samples], dim=0),
        "attention_mask": torch.stack(
            [s["attention_mask"] for s in samples], dim=0
        ),
        "pixel_values": torch.cat([s["pixel_values"] for s in samples], dim=0),
        "image_grid_thw": torch.cat([s["image_grid_thw"] for s in samples], dim=0),
    }


def get_dataloader(
    cfg: TrainConfig,
    tokenizer,
    pp_size: int,
    tokens_per_image: int,
):
    """构建训练 dataloader。

    目前是随机数据；换成真实数据集时，只要保持产出同样的 batch 字段
    （input_ids / labels / attention_mask / pixel_values / image_grid_thw）即可，
    训练循环不需要改。
    """
    dataset = RandomVLDataset(
        cfg,
        num_samples=max(cfg.max_steps, 1) * cfg.per_device_batch_size,
        tokens_per_image=tokens_per_image,
    )
    return DataLoader(
        dataset,
        batch_size=cfg.per_device_batch_size,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=vl_collate,
    )


# -----------------------------
# 从网上取 config（每节点只让 LOCAL_RANK=0 联网）
# -----------------------------
def _prepare_hf_cache(args) -> None:
    """LOCAL_RANK=0 负责把 config / tokenizer 拉下来，其余 rank 等 barrier。"""
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if local_rank == 0:
        tokenizer_name = args.tokenizer_name or args.model_name
        print(f"[hf-cache] rank {dist.get_rank()} fetching {tokenizer_name} ...", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            trust_remote_code=True,
            local_files_only=args.hf_local_files_only,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        AutoConfig.from_pretrained(
            args.model_name,
            trust_remote_code=True,
            local_files_only=args.hf_local_files_only,
        )
        print(f"[hf-cache] rank {dist.get_rank()} fetch done", flush=True)

    dist.barrier()
    print(f"[hf-cache] rank {dist.get_rank()} barrier passed", flush=True)


# -----------------------------
# 建模 / PP 切分
# -----------------------------
def build_qwen_vl_model(cfg: TrainConfig, *, local_files_only: bool = False):
    """按 HF 上的 config 建模型；默认在 meta 上随机初始化，不加载任何权重。"""
    attn_impl = "flash_attention_2" if cfg.use_flash_attn else "sdpa"
    kwargs = {
        "dtype": torch.bfloat16 if cfg.bf16 else (torch.float16 if cfg.fp16 else None),
        "attn_implementation": attn_impl,
        "trust_remote_code": True,
    }
    if cfg.init_from_pretrained:
        model = AutoModelForImageTextToText.from_pretrained(
            cfg.model_name,
            local_files_only=local_files_only,
            **kwargs,
        )
    else:
        config = AutoConfig.from_pretrained(
            cfg.model_name,
            trust_remote_code=True,
            local_files_only=local_files_only,
        )
        if attn_impl is not None:
            config._attn_implementation = attn_impl
        config.text_config._attn_implementation = attn_impl
        config.vision_config._attn_implementation = attn_impl
        with torch.device("meta"):
            model = AutoModelForImageTextToText.from_config(
                config,
                dtype=kwargs["dtype"],
                trust_remote_code=True,
            )

    # 两条分支都要填，随机数据那边靠这几个值定形状。
    config = model.config
    cfg.vocab_size = int(config.text_config.vocab_size)
    cfg.image_token_id = int(config.image_token_id)
    cfg.patch_dim = int(
        config.vision_config.in_channels
        * config.vision_config.temporal_patch_size
        * config.vision_config.patch_size
        * config.vision_config.patch_size
    )

    if cfg.activation_checkpointing:
        # Qwen3-VL 的 use_cache 挂在 text_config 上
        config.use_cache = False
        config.text_config.use_cache = False
        model.gradient_checkpointing_enable()

    return model


def build_tokenizer(cfg: TrainConfig, *, local_files_only: bool = False):
    tokenizer_name = cfg.tokenizer_name or cfg.model_name
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        trust_remote_code=True,
        local_files_only=local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _ensure_rotary_embeddings(model, device) -> None:
    """to_empty() 会把 meta 上的 non-persistent buffer 变成未初始化的显存垃圾。

    视觉塔和语言模型的 RoPE inv_freq 都是这种 buffer，必须在第一次 forward 前重建。
    """
    from transformers.models.qwen3_vl.modeling_qwen3_vl import (
        Qwen3VLTextRotaryEmbedding,
        Qwen3VLVisionRotaryEmbedding,
    )

    visual = getattr(model.model, "visual", None)
    if visual is not None:
        head_dim = visual.config.hidden_size // visual.config.num_heads
        visual.rotary_pos_emb = Qwen3VLVisionRotaryEmbedding(head_dim // 2).to(device)

    language_model = model.model.language_model
    language_model.rotary_emb = Qwen3VLTextRotaryEmbedding(
        config=language_model.config
    ).to(device)


def partition_qwen_vl_model(
    model,
    stage_idx: int,
    num_stages: int,
    debug_nan_steps: int = 0,
):
    """把 Qwen3-VL 按 PP 切开。

      stage 0        : 视觉塔 + language_model.embed_tokens
      中间 stage     : 只有自己那一段 language_model.layers
      最后一个 stage : language_model.norm + lm_head

    和 qwen14b 一样，每个 stage 都保留自己的 layers，多余的部分置 None；
    模型自己的 forward 换成一个能处理分片状态的 custom_forward。
    """
    config = model.config
    text_config = config.text_config
    num_layers = text_config.num_hidden_layers

    layers_per_stage = num_layers // num_stages
    remainder = num_layers % num_stages
    start_layer = stage_idx * layers_per_stage + min(stage_idx, remainder)
    end_layer = start_layer + layers_per_stage + (1 if stage_idx < remainder else 0)

    layers_to_keep = list(range(start_layer, end_layer))
    model.model.language_model.layers = torch.nn.ModuleList(
        [model.model.language_model.layers[i] for i in layers_to_keep]
    )
    if len(model.model.language_model.layers) == 0:
        model.model.language_model.layers = torch.nn.ModuleList([torch.nn.Identity()])

    is_first = stage_idx == 0
    is_last = stage_idx == num_stages - 1

    # 视觉塔和 embedding 只有 stage 0 需要
    if not is_first:
        model.model.visual = None
        model.model.language_model.embed_tokens = None
    # norm / lm_head 只有最后一个 stage 需要
    if not is_last:
        model.lm_head = None
        model.model.language_model.norm = None

    image_token_id = int(config.image_token_id)
    language_model = model.model.language_model
    visual = getattr(model.model, "visual", None)
    num_deepstack = len(visual.deepstack_merger_list) if visual is not None else 0

    def layer_attention_mask(attention_mask):
        if attention_mask is None:
            return None
        if attention_mask.dim() != 2:
            return attention_mask
        # 全 1（正常 batch）或全 0（pipeline 形状推断时的哑输入）都不需要显式 mask
        if bool(attention_mask.all().item()) or not bool(attention_mask.any().item()):
            return None
        return attention_mask

    def custom_forward(
        input_ids_or_hidden,
        attention_mask=None,
        pixel_values=None,
        image_grid_thw=None,
    ):
        debug_forward = int(debug_nan_steps or 0) > 0
        debug_count = int(getattr(model, "_polar_forward_debug_count", 0))

        # to_empty() 之后 RoPE 的 inv_freq buffer 是垃圾值，第一次 forward 前重建一次。
        # 每个 stage 都要，因为每个 stage 都要自己算 position_embeddings。
        if getattr(model, "_polar_rotary_ready", False) is False:
            _ensure_rotary_embeddings(model, input_ids_or_hidden.device)
            model._polar_rotary_ready = True

        visual_pos_masks = None
        deepstack_visual_embeds = None

        if language_model.embed_tokens is not None:
            # stage 0：输入是 token ids，先做 embedding，再把图片特征 scatter 进去
            hidden_states = language_model.embed_tokens(input_ids_or_hidden)

            image_mask_2d = input_ids_or_hidden == image_token_id
            n_image_tokens = int(image_mask_2d.sum().item())
            if (
                pixel_values is not None
                and model.model.visual is not None
                and n_image_tokens > 0
            ):
                vision_out = model.model.visual(
                    pixel_values, grid_thw=image_grid_thw
                )
                image_embeds = vision_out.pooler_output
                if isinstance(image_embeds, (list, tuple)):
                    image_embeds = torch.cat(image_embeds, dim=0)
                image_embeds = image_embeds.to(
                    hidden_states.device, hidden_states.dtype
                )
                if image_embeds.shape[0] != n_image_tokens:
                    raise RuntimeError(
                        f"[stage {stage_idx}] image token / feature 数量对不上："
                        f"tokens={n_image_tokens} features={image_embeds.shape[0]}"
                    )
                image_mask = image_mask_2d.unsqueeze(-1).expand_as(hidden_states)
                hidden_states = hidden_states.masked_scatter(image_mask, image_embeds)
                visual_pos_masks = image_mask_2d
                deepstack_visual_embeds = vision_out.deepstack_features
        else:
            # 后续 stage：输入已经是上一段传过来的 hidden states
            hidden_states = input_ids_or_hidden
            if torch.is_floating_point(hidden_states) and not hidden_states.requires_grad:
                hidden_states.requires_grad_(True)

        seq_length = hidden_states.shape[1]
        batch_size = hidden_states.shape[0]
        position_ids = (
            torch.arange(seq_length, device=hidden_states.device)
            .unsqueeze(0)
            .repeat(batch_size, 1)
        )
        position_embeddings = language_model.rotary_emb(hidden_states, position_ids)

        decoder_attention_mask = layer_attention_mask(attention_mask)

        for local_idx, layer in enumerate(language_model.layers):
            if isinstance(layer, torch.nn.Identity):
                hidden_states = layer(hidden_states)
            else:
                layer_outputs = layer(
                    hidden_states,
                    attention_mask=decoder_attention_mask,
                    position_embeddings=position_embeddings,
                )
                hidden_states = (
                    layer_outputs[0] if isinstance(layer_outputs, tuple) else layer_outputs
                )

            # DeepStack：视觉塔抽出来的浅层特征加回语言模型前几层
            global_idx = start_layer + local_idx
            if (
                deepstack_visual_embeds is not None
                and global_idx < len(deepstack_visual_embeds)
            ):
                hidden_states = language_model._deepstack_process(
                    hidden_states,
                    visual_pos_masks,
                    deepstack_visual_embeds[global_idx],
                )

            if debug_forward and debug_count < int(debug_nan_steps):
                hs_float = hidden_states.detach().float()
                if not bool(torch.isfinite(hs_float).all().item()):
                    print(
                        f"[debug_forward][stage {stage_idx}] nonfinite after "
                        f"local_layer={local_idx}",
                        flush=True,
                    )
                    break

        if language_model.norm is not None:
            hidden_states = language_model.norm(hidden_states)

        if model.lm_head is not None:
            output = model.lm_head(hidden_states)
        else:
            output = hidden_states

        if debug_forward and debug_count < int(debug_nan_steps):
            out_float = output.detach().float()
            print(
                f"[debug_forward][stage {stage_idx}] exit "
                f"shape={tuple(output.shape)} dtype={output.dtype} "
                f"finite={bool(torch.isfinite(out_float).all().item())} "
                f"min={float(out_float.min().item()) if output.numel() else 0.0:.6g} "
                f"max={float(out_float.max().item()) if output.numel() else 0.0:.6g}",
                flush=True,
            )
            model._polar_forward_debug_count = debug_count + 1
        return output

    model.forward = custom_forward

    assigned_layers = list(range(start_layer, end_layer))
    print(
        f"[partition] Stage {stage_idx}: assigned layers {assigned_layers}, "
        f"visual={model.model.visual is not None}, "
        f"embed_tokens={language_model.embed_tokens is not None}, "
        f"norm={language_model.norm is not None}, "
        f"lm_head={model.lm_head is not None}, "
        f"deepstack={num_deepstack}"
    )

    return model


def vocab_parallel_lm_loss(
    output,
    target,
    ignore_index: int,
    tp_mesh,
    debug: bool = False,
):
    """Causal LM loss for either full logits or TP vocab-sharded DTensor logits."""
    shift_labels = target[..., 1:].contiguous()

    if not hasattr(output, "to_local"):
        shift_logits = output[..., :-1, :].contiguous()
        return F.cross_entropy(
            shift_logits.float().view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=ignore_index,
        )

    if tp_mesh is None:
        raise RuntimeError("DTensor logits require a TP mesh for vocab-parallel loss.")

    local_logits = output.to_local()[..., :-1, :].float().contiguous()
    flat_logits = local_logits.view(-1, local_logits.size(-1))
    flat_labels = shift_labels.view(-1)

    tp_rank = int(tp_mesh.get_local_rank())
    tp_size = int(tp_mesh.size())
    global_vocab_size = int(output.size(-1))
    base = global_vocab_size // tp_size
    remainder = global_vocab_size % tp_size
    vocab_start = tp_rank * base + min(tp_rank, remainder)
    vocab_end = vocab_start + base + (1 if tp_rank < remainder else 0)

    valid = flat_labels.ne(ignore_index)
    in_local_vocab = valid & flat_labels.ge(vocab_start) & flat_labels.lt(vocab_end)
    local_target = (flat_labels - vocab_start).clamp(
        min=0,
        max=max(vocab_end - vocab_start - 1, 0),
    )

    local_max = flat_logits.max(dim=-1).values
    global_max = local_max.clone()
    dist.all_reduce(global_max, op=dist.ReduceOp.MAX, group=tp_mesh.get_group())

    exp_sum = torch.exp(flat_logits - global_max.unsqueeze(-1)).sum(dim=-1)
    dist.all_reduce(exp_sum, op=dist.ReduceOp.SUM, group=tp_mesh.get_group())

    target_logits = torch.zeros_like(global_max)
    if in_local_vocab.any():
        target_logits[in_local_vocab] = flat_logits[
            in_local_vocab,
            local_target[in_local_vocab],
        ]
    dist.all_reduce(target_logits, op=dist.ReduceOp.SUM, group=tp_mesh.get_group())

    losses = torch.log(exp_sum.clamp_min(1e-20)) + global_max - target_logits
    if debug:
        rank = dist.get_rank()
        print(
            f"[debug_loss][rank {rank}] dtensor=True "
            f"logits_finite={bool(torch.isfinite(flat_logits).all().item())} "
            f"losses_finite={bool(torch.isfinite(losses).all().item())} "
            f"valid={int(valid.sum().item())} "
            f"local_targets={int(in_local_vocab.sum().item())} "
            f"vocab_range=[{vocab_start},{vocab_end})",
            flush=True,
        )
    if valid.any():
        return losses[valid].mean()
    return losses.sum() * 0.0


# -----------------------------
# Main Training Loop
# -----------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Polar-SGD pretraining for Qwen3-VL models"
    )

    # Model and tokenizer
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--tokenizer-name", type=str, default=None)
    parser.add_argument(
        "--hf-local-files-only",
        action="store_true",
        help="Require model/tokenizer metadata to already be cached.",
    )

    # 数据集：目前数据是随机搓的，这三个参数只是给 polar wrapper 拼 trace 目录名用，
    # 以及为以后接真实数据集留位。
    parser.add_argument("--dataset-name-or-path", type=str, default="random-vl")
    parser.add_argument("--dataset-config", type=str, default=None)
    parser.add_argument("--text-field", type=str, default="text")

    # 随机数据
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--images-per-sample", type=int, default=1)
    parser.add_argument("--image-grid-h", type=int, default=4)
    parser.add_argument("--image-grid-w", type=int, default=4)
    parser.add_argument("--data-seed", type=int, default=1234)

    # Training hyperparameters
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2.0e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.02)
    parser.add_argument("--max-tokens", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--clip-norm", type=float, default=1.0)

    # Logging and saving
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--save-interval", type=int, default=1000)
    parser.add_argument("--save-dir", type=str, default="checkpoints/qwen3_vl_8b_instruct")
    parser.add_argument("--run-label", type=str, default="")
    parser.add_argument(
        "--step-log-dir",
        type=str,
        default="experiments/qwenvl8b/outputs/step_csv",
    )
    parser.add_argument(
        "--debug-nan-steps",
        type=int,
        default=0,
        help="Print parameter and batch finite/range checks for the first N steps.",
    )
    parser.add_argument("--disable-profiler", type=str_to_bool, default=False)
    parser.add_argument("--profiler-wait-steps", type=int, default=1)
    parser.add_argument("--profiler-warmup-steps", type=int, default=1)
    parser.add_argument("--profiler-active-steps", type=int, default=1)
    parser.add_argument("--profiler-repeat", type=int, default=1)
    parser.add_argument("--profiler-memory", type=str_to_bool, default=False)
    parser.add_argument("--profiler-shapes", type=str_to_bool, default=False)
    parser.add_argument("--profiler-stack", type=str_to_bool, default=False)
    parser.add_argument("--profiler-flops", type=str_to_bool, default=False)
    parser.add_argument("--profiler-acc-events", type=str_to_bool, default=False)

    # Data loader
    parser.add_argument("--num-workers", type=int, default=2)

    # Mixed precision and optimization
    parser.add_argument(
        "--use-flash-attn",
        action="store_true",
        default=False,
        help=(
            "Use flash_attention_2. 默认关掉：视觉塔走的是 varlen cu_seqlens 路径，"
            "先用 sdpa 保证能跑通。"
        ),
    )
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--fp16", action="store_true", default=False)
    parser.add_argument(
        "--activation-checkpointing",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--init-from-pretrained",
        action="store_true",
        default=False,
        help=(
            "Load pretrained weights before partitioning. By default this "
            "script builds from config because PolarParallel initializes the "
            "partitioned stage on device."
        ),
    )

    # Parallelism
    parser.add_argument("--pp-size", type=int, default=1)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--micro-batches", type=int, default=1)
    parser.add_argument("--comm-timing", type=int, default=-1)
    parser.add_argument("--using-polar", type=str_to_bool, default=True)

    # Polar hooks
    parser.add_argument(
        "--polar-hook",
        type=str,
        default="momentum",
        choices=[
            "io",
            "momentum",
            "gpipe",
            "ef_only",
            "ef_lowmem",
            "ef_full_async_launch",
            "scaling_only",
            "none",
        ],
    )
    parser.add_argument("--polar-beta", type=float, default=0.9)
    parser.add_argument("--polar-bucket-numel", type=int, default=4_000_000)
    parser.add_argument("--polar-max-inflight-buckets", type=int, default=1)

    # Baseline mode
    parser.add_argument(
        "--baseline-mode",
        type=str,
        default="manual",
        choices=["manual", "ddp"],
    )

    # Local-SGD arguments
    parser.add_argument("--use-local-sgd", action="store_true")
    parser.add_argument("--local-sgd-steps", type=int, default=10)

    # Communication backend for POLAR DP all-reduce.
    parser.add_argument(
        "--method",
        type=str,
        default="bitscom",
        choices=["none", "bitscom"],
    )
    parser.add_argument("--bitwidth", type=int, default=4)
    parser.add_argument("--simulate-quantization", action="store_true")
    parser.add_argument("--stochastic-rounding", action="store_true")

    args = parser.parse_args()

    if args.method == "bitscom" and not args.using_polar:
        raise ValueError("--method bitscom requires --using-polar true")
    if args.using_polar and args.method != "bitscom":
        print(
            "[warn] POLAR is enabled without bitscom; DP communication will "
            "use dense torch.distributed all-reduce.",
            flush=True,
        )

    if args.pp_size <= 0 or args.tp_size <= 0:
        raise ValueError("--pp-size and --tp-size must be positive")
    if args.pp_size < 2:
        raise ValueError(
            "--pp-size must be >= 2: PolarParallel 的 1F1B 循环只在 "
            "is_first / is_last 分支里喂 target，单 stage 会算不出 loss。"
        )
    if args.tp_size > 1:
        # polar wrapper 的 TP 计划目前只认 model.model.layers，Qwen3-VL 的
        # 语言层在 model.model.language_model.layers，走 TP 需要先扩 wrapper。
        raise ValueError(
            "QwenVL8B 暂不支持 --tp-size > 1（polar wrapper 的 TP 计划还没适配 "
            "Qwen3-VL 的层命名），请用 --tp-size 1，靠 PP x DP 铺满卡数。"
        )
    if args.micro_batches < args.pp_size:
        raise ValueError(
            f"--micro-batches ({args.micro_batches}) must be >= "
            f"--pp-size ({args.pp_size})"
        )
    if args.per_device_batch_size < args.micro_batches:
        raise ValueError(
            f"--per-device-batch-size ({args.per_device_batch_size}) must be >= "
            f"--micro-batches ({args.micro_batches})"
        )
    if args.per_device_batch_size % args.micro_batches != 0:
        raise ValueError(
            f"--per-device-batch-size ({args.per_device_batch_size}) must be "
            f"divisible by --micro-batches ({args.micro_batches})"
        )
    if args.comm_timing != -1 and not (0 <= args.comm_timing < args.micro_batches):
        raise ValueError(
            f"--comm-timing must be -1 or in [0, {args.micro_batches - 1}], "
            f"got {args.comm_timing}."
        )
    if args.polar_bucket_numel <= 0:
        raise ValueError("--polar-bucket-numel must be positive")
    if args.polar_max_inflight_buckets <= 0:
        raise ValueError("--polar-max-inflight-buckets must be positive")
    if args.images_per_sample <= 0:
        raise ValueError("--images-per-sample must be positive")

    bitscom_module = None
    if args.method == "bitscom":
        bitscom_module = import_bitscom()
        bitscom_module.init(bitwidth=args.bitwidth)

    # Initialize distributed
    dist.init_process_group(backend="nccl", init_method="env://")
    world_size = dist.get_world_size()
    if dist.get_rank() == 0:
        print(
            "[qwenvl8b-config] "
            f"using_polar={args.using_polar} polar_hook={args.polar_hook} "
            f"method={args.method} bitwidth={args.bitwidth} "
            f"comm_timing={args.comm_timing} micro_batches={args.micro_batches} "
            f"pp={args.pp_size} tp={args.tp_size} world_size={world_size}",
            flush=True,
        )

    pp_size = args.pp_size
    tp_size = args.tp_size
    model_parallel_size = pp_size * tp_size
    assert world_size % model_parallel_size == 0, (
        f"world_size {world_size} must be divisible by "
        f"PP_SIZE * TP_SIZE ({pp_size} * {tp_size})"
    )
    dp_size = world_size // model_parallel_size
    if tp_size > 1:
        device_mesh = init_device_mesh(
            "cuda",
            (dp_size, pp_size, tp_size),
            mesh_dim_names=("dp", "pp", "tp"),
        )
    else:
        device_mesh = init_device_mesh(
            "cuda",
            (dp_size, pp_size),
            mesh_dim_names=("dp", "pp"),
        )
    dp_mesh = device_mesh["dp"]
    pp_mesh = device_mesh["pp"]

    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    print(
        "[qwenvl8b-process] "
        f"rank={dist.get_rank()} local_rank={local_rank} "
        f"pid={os.getpid()} host={socket.gethostname()}",
        flush=True,
    )

    lowbit_dp_group = None
    if args.method == "bitscom":
        lowbit_dp_group = create_lowbit_dp_group(dp_mesh.get_group())

    # Enable TF32 for better performance
    torch.backends.cuda.matmul.allow_tf32 = True

    _prepare_hf_cache(args)

    cfg = TrainConfig(
        model_name=args.model_name,
        tokenizer_name=args.tokenizer_name,
        seq_len=args.seq_len,
        per_device_batch_size=args.per_device_batch_size,
        images_per_sample=args.images_per_sample,
        image_grid_h=args.image_grid_h,
        image_grid_w=args.image_grid_w,
        seed=args.data_seed,
        grad_accum_steps=args.grad_accum_steps,
        lr=args.lr,
        warmup_ratio=args.warmup_ratio,
        max_tokens=args.max_tokens,
        max_steps=args.max_steps,
        weight_decay=args.weight_decay,
        beta1=args.beta1,
        beta2=args.beta2,
        clip_norm=args.clip_norm,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        save_dir=args.save_dir,
        num_workers=args.num_workers,
        use_flash_attn=args.use_flash_attn,
        bf16=args.bf16,
        fp16=args.fp16,
        activation_checkpointing=args.activation_checkpointing,
        init_from_pretrained=args.init_from_pretrained,
    )

    tokenizer = build_tokenizer(cfg, local_files_only=True)
    model = build_qwen_vl_model(cfg, local_files_only=True)
    print(
        f"[build] rank {dist.get_rank()} config built, "
        f"params={sum(p.numel() for p in model.parameters())/1e9:.3f}B "
        f"vocab={cfg.vocab_size} image_token_id={cfg.image_token_id}",
        flush=True,
    )

    # 一张图在语言侧占几个 token：(grid_h / merge) * (grid_w / merge)
    vision_config = model.config.vision_config
    merge = int(vision_config.spatial_merge_size)
    if cfg.image_grid_h % merge or cfg.image_grid_w % merge:
        raise ValueError(
            f"--image-grid-h/-w 必须能被 spatial_merge_size({merge}) 整除"
        )
    tokens_per_image = (cfg.image_grid_h // merge) * (cfg.image_grid_w // merge)

    stage_idx = pp_mesh.get_local_rank()
    tp_rank = device_mesh["tp"].get_local_rank() if tp_size > 1 else 0
    print(f"Stage index: {stage_idx} / {pp_size}; TP rank: {tp_rank} / {tp_size}")

    stage_model = partition_qwen_vl_model(
        model,
        stage_idx,
        pp_size,
        debug_nan_steps=args.debug_nan_steps,
    )

    dp_rank = dp_mesh.get_local_rank()
    print(f"DP rank: {dp_rank} / {dp_size}")

    lowbit_group = None
    if args.method == "bitscom":
        lowbit_group = bitscom_module.LowBitGroup(
            bitwidth=args.bitwidth,
            process_group=lowbit_dp_group,
            simulate_quantization=args.simulate_quantization,
            stochastic_rounding=args.stochastic_rounding,
            backend_allreduce=True,
        )
        if dist.get_rank() == 0:
            print(
                "[bitscom] enabled for POLAR DP communication: "
                f"bitwidth={args.bitwidth} "
                f"simulate_quantization={args.simulate_quantization} "
                f"stochastic_rounding={args.stochastic_rounding} "
                "backend_allreduce=True",
                flush=True,
            )

    dataloader = get_dataloader(cfg, tokenizer, pp_size, tokens_per_image)

    def loss_fn(output, target):
        return vocab_parallel_lm_loss(
            output,
            target,
            ignore_index=tokenizer.pad_token_id,
            tp_mesh=device_mesh["tp"] if tp_size > 1 else None,
            debug=int(args.debug_nan_steps or 0) > 0,
        )

    trainer = PolarParallel(
        args=args,
        device_mesh=device_mesh,
        micro_batches=args.micro_batches,
        loss_fn=loss_fn,
        stage_model=stage_model,
        dataloader=dataloader,
        comm_timing=args.comm_timing,
        use_local_sgd=args.use_local_sgd,
        local_sgd_steps=args.local_sgd_steps,
        baseline_mode=args.baseline_mode,
    )
    trainer.lowbit_group = lowbit_group

    trainer.train()


if __name__ == "__main__":
    main()
