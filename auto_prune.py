# classification/auto_prune_qp.py
# ------------------------------------------------------------
# One-click QP-SNN aligned pruning for QSD/Spike-Driven Transformer V2 style models
#
# Pipeline:
#   1) auto-discover spike activation modules (Multispike / name contains "spike")
#   2) auto-discover prunable quant layers (Conv2dReScaW/Conv1dReScaW/LinearReScaW)
#   3) extract rank_conv (SVS matrix_rank) from activations (QP aligned)
#   4) prune quant layers by rank + compress_rate (QP aligned)
#
# By default, it pairs activation_i -> layer_i by traversal order.
# ------------------------------------------------------------

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Union
import re
import torch
import torch.nn as nn

from qp_prune import (
    extract_rank_conv,
    prune_by_rank_conv,
    RankExtractConfig,
    PruneConfig,
    parse_compress_rate,
)
from qp_quant import Conv2dReScaW, Conv1dReScaW, LinearReScaW


@dataclass
class AutoPruneConfig:
    # rank extraction
    limit_batches: int = 6
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    save_rank_dir: Optional[str] = "./rank_conv_auto"  # None -> do not save
    save_rank_prefix: str = "rank_conv"

    # pruning
    compress_rate: Union[str, List[float]] = "[0.35]*999"  # default: prune 35% each layer
    min_keep: int = 1
    skip_layer_ids_1based: Optional[List[int]] = None

    # auto-discovery behavior
    activation_name_keywords: Tuple[str, ...] = ("multispike", "spike")
    # if True: use only modules whose class name contains "spike"
    strict_spike_classname: bool = False

    # pairing strategy
    # "order": activation_i -> prunable_layer_i (QP-like)
    pairing: str = "order"

    # optional: keep only first N pairs (debug)
    max_pairs: Optional[int] = None


def _is_spike_activation(name: str, module: nn.Module, cfg: AutoPruneConfig) -> bool:
    cls = module.__class__.__name__.lower()
    nm = name.lower()

    if cfg.strict_spike_classname:
        return "spike" in cls

    # relaxed: class name OR module name contains keywords
    for kw in cfg.activation_name_keywords:
        if kw in cls or kw in nm:
            return True
    return False


def discover_spike_activations(model: nn.Module, cfg: AutoPruneConfig) -> List[str]:
    acts: List[str] = []
    for name, m in model.named_modules():
        if _is_spike_activation(name, m, cfg):
            acts.append(name)
    return acts


def discover_prunable_quant_layers(model: nn.Module) -> List[str]:
    layers: List[str] = []
    for name, m in model.named_modules():
        if isinstance(m, (Conv2dReScaW, Conv1dReScaW, LinearReScaW)):
            layers.append(name)
    return layers


def _truncate_pairs(acts: List[str], layers: List[str], max_pairs: Optional[int]) -> Tuple[List[str], List[str]]:
    n = min(len(acts), len(layers))
    if max_pairs is not None:
        n = min(n, int(max_pairs))
    return acts[:n], layers[:n]

@torch.no_grad()
def _get_activation_channels(model: nn.Module, dataloader, act_names: List[str], device: torch.device, max_batches: int = 1) -> Dict[str, int]:
    """
    跑 1 个 batch，通过 hook 拿到每个激活层输出的通道数 C（用于匹配 out_channels）
    只需要通道维，不做 rank。
    """
    name_to_module = dict(model.named_modules())
    ch: Dict[str, int] = {}

    handles = []
    def make_hook(nm):
        def hook(mod, inp, out):
            # out: could be [B,T,C,H,W] or [B,C,H,W] or others
            if isinstance(out, (list, tuple)):
                out = out[0]
            if out is None:
                return
            if out.dim() == 5:
                # assume [B,T,C,H,W] or [T,B,C,H,W]
                # choose the dim that looks like C (usually the 3rd index)
                # common: [B,T,C,H,W]
                C = out.shape[2]
            elif out.dim() == 4:
                # [B,C,H,W]
                C = out.shape[1]
            elif out.dim() == 3:
                # [B,T,C] or [B,C,L]
                # if middle dim small it's T else it's C; fallback to last dim as C for [B,T,C]
                C = out.shape[-1]
            else:
                return
            ch[nm] = int(C)
        return hook

    for a in act_names:
        if a not in name_to_module:
            continue
        handles.append(name_to_module[a].register_forward_hook(make_hook(a)))

    model.eval()
    model.to(device)
    for i, batch in enumerate(dataloader):
        if i >= max_batches:
            break
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        x = x.to(device, non_blocking=True)
        _ = model(x)

    for h in handles:
        h.remove()

    return ch


def _prefix(name: str) -> str:
    """
    用模块名的前缀做粗粒度分组：比如去掉最后一个字段
    module.ConvBlock1_1.0.Conv.lif2 -> module.ConvBlock1_1.0.Conv
    """
    parts = name.split(".")
    if len(parts) <= 1:
        return name
    return ".".join(parts[:-1])


def pair_acts_to_layers_by_shape_and_prefix(
    model: nn.Module,
    dataloader,
    act_names: List[str],
    layer_names: List[str],
    device: torch.device,
    max_probe_batches: int = 1,
) -> Tuple[List[str], List[str]]:
    """
    核心：按 (同前缀优先 + 通道数一致) 配对
    - 激活层给出 C_act
    - 权重层给出 C_out = weight.shape[0]
    - 在“同前缀或近前缀”的候选里找 C_out == C_act 的最近层
    """
    # 1) 先拿激活通道数
    act_ch = _get_activation_channels(model, dataloader, act_names, device, max_batches=max_probe_batches)

    # 2) 准备权重层输出通道数
    name_to_module = dict(model.named_modules())
    layer_out = {}
    for ln in layer_names:
        m = name_to_module[ln]
        layer_out[ln] = int(m.weight.shape[0])

    # 3) 建索引：按 prefix 分组
    layers_by_prefix: Dict[str, List[str]] = {}
    for ln in layer_names:
        px = _prefix(ln)
        layers_by_prefix.setdefault(px, []).append(ln)

    # 4) 逐个激活找匹配层（贪心：每层只用一次）
    used_layers = set()
    paired_a, paired_l = [], []

    for a in act_names:
        if a not in act_ch:
            continue
        C = act_ch[a]
        apx = _prefix(a)

        # 候选 prefix：同 prefix -> 上一级 prefix -> 再上一级
        cand_prefixes = [apx]
        if "." in apx:
            cand_prefixes.append(".".join(apx.split(".")[:-1]))
        if cand_prefixes[-1].count(".") >= 1:
            cand_prefixes.append(".".join(cand_prefixes[-1].split(".")[:-1]))

        candidates = []
        for px in cand_prefixes:
            for ln in layers_by_prefix.get(px, []):
                if ln in used_layers:
                    continue
                if layer_out[ln] == C:
                    candidates.append(ln)

        # 如果同前缀找不到，就全局找 shape 匹配
        if len(candidates) == 0:
            for ln in layer_names:
                if ln in used_layers:
                    continue
                if layer_out[ln] == C:
                    candidates.append(ln)

        if len(candidates) == 0:
            # 没有 shape 一致的就跳过
            continue

        # 选“名字距离最近”的：简单用共同前缀长度最大
        def score(ln: str) -> int:
            common = 0
            for x, y in zip(a.split("."), ln.split(".")):
                if x == y:
                    common += 1
                else:
                    break
            return common

        best = max(candidates, key=score)
        used_layers.add(best)
        paired_a.append(a)
        paired_l.append(best)

    return paired_a, paired_l

def auto_prune_qp_aligned(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    cfg: AutoPruneConfig,
) -> Dict[str, torch.Tensor]:
    """
    Run the whole QP-aligned pruning automatically.

    Returns:
      masks dict: {layer_name: mask_1d}
    """
    # 1) discover
    act_names = discover_spike_activations(model, cfg)
    layer_names = discover_prunable_quant_layers(model)

    paired_act, paired_layer = pair_acts_to_layers_by_shape_and_prefix(
        model=model,
        dataloader=dataloader,
        act_names=act_names,
        layer_names=layer_names,
        device=torch.device(cfg.device),
        max_probe_batches=1
    )

    # 可选：限制 max_pairs
    if cfg.max_pairs is not None:
        paired_act = paired_act[:cfg.max_pairs]
        paired_layer = paired_layer[:cfg.max_pairs]

    act_names, layer_names = paired_act, paired_layer


    if len(act_names) == 0 or len(layer_names) == 0:
        raise RuntimeError("After pairing/truncation, no layers left to prune.")

    if len(act_names) != len(layer_names):
        # We always make them equal by truncation
        raise RuntimeError("Internal pairing bug: act_names != layer_names")

    print("\n[AutoPrune] Found spike activations =", len(act_names))
    print("[AutoPrune] Found prunable quant layers =", len(layer_names))
    print("[AutoPrune] Will prune pairs =", len(layer_names))
    print("[AutoPrune] Example pairing (first 10):")
    for i in range(min(10, len(layer_names))):
        print(f"  #{i+1:02d}  act: {act_names[i]}   ->   layer: {layer_names[i]}")

    # 3) extract ranks (QP aligned)
    rank_cfg = RankExtractConfig(limit_batches=cfg.limit_batches, device=cfg.device, use_no_grad=True)

    save_dir = cfg.save_rank_dir
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    _ = extract_rank_conv(
        model=model,
        dataloader=dataloader,
        activation_module_names=act_names,
        cfg=rank_cfg,
        save_dir=save_dir,
        save_prefix=cfg.save_rank_prefix,
    )

    # 4) prune by ranks + compress_rate (QP aligned)
    prune_cfg = PruneConfig(min_keep=cfg.min_keep, skip_layer_ids_1based=cfg.skip_layer_ids_1based)

    masks = prune_by_rank_conv(
        model=model,
        compress_rate=cfg.compress_rate,
        rank_dir=save_dir,                 # ranks saved there
        rank_prefix=cfg.save_rank_prefix,  # rank_conv1.npy ...
        cfg=prune_cfg,
        prunable_layer_names=layer_names,  # IMPORTANT: prune exactly those layers, aligned with activations
    )

    print("\n[AutoPrune] Applied masks to layers =", len(masks))
    # show keep ratio summary
    kept = []
    for ln, mk in list(masks.items())[:10]:
        kept.append(f"{ln}: keep {int(mk.sum().item())}/{mk.numel()}")
    print("[AutoPrune] Example mask stats (first 10):")
    for s in kept:
        print("  ", s)

    return masks
