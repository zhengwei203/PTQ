# tg_srs_prune.py
# -----------------------------------------------------------------------------
# TG-SRS: Task-guided Spatiotemporal Rank-preserving Structured Pruning
# A drop-in replacement for the current QP-SNN/SVS pruning scorer.
#
# Put this file under QSDT_classfication/, then call auto_prune_tgsrs(...)
# in main_finetune.py before finetuning.
# -----------------------------------------------------------------------------

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from qp_prune import parse_compress_rate


# ----------------------------- helpers ---------------------------------------

def _module_type(name: str) -> str:
    """Type prior used by structure-protected pruning-rate scheduling."""
    n = name.lower()
    if any(k in n for k in ["q_conv", "k_conv", "v_conv", "qkv", "query", "key", "value", "proj", "attn"]):
        return "attn"
    if any(k in n for k in ["head", "classifier", "cls_head"]):
        return "head"
    if any(k in n for k in ["encode_conv", "patch", "embed", "stem", "downsample"]):
        return "embed"
    if any(k in n for k in ["fc1", "fc2", "mlp"]):
        return "mlp"
    if "dwconv" in n:
        return "dwconv"
    if "conv" in n:
        return "conv"
    return "other"


def _zscore(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x = x.float()
    return (x - x.mean()) / (x.std(unbiased=False) + eps)


def _as_tensor_output(out):
    if isinstance(out, (list, tuple)):
        out = out[0]
    return out if isinstance(out, torch.Tensor) else None


def _to_btcx(x: torch.Tensor) -> torch.Tensor:
    """
    Convert common spike activation layouts to [B, T, C, S].

    Supported examples in this project:
      [T,B,C,H,W] -> [B,T,C,H*W]
      [B,T,C,H,W] -> [B,T,C,H*W]
      [T,B,C,N]   -> [B,T,C,N]
      [B,C,H,W]   -> [B,1,C,H*W]
      [B,C,N]     -> [B,1,C,N]
      [T,B,C]     -> [B,T,C,1]
      [B,C]       -> [B,1,C,1]
    """
    if x.dim() == 5:
        # In your model most spike tensors are [T,B,C,H,W].
        if x.shape[0] <= 32 and x.shape[1] >= x.shape[0]:
            x = x.permute(1, 0, 2, 3, 4).contiguous()  # [B,T,C,H,W]
        # else assume already [B,T,C,H,W]
        return x.flatten(3)  # [B,T,C,S]

    if x.dim() == 4:
        # [T,B,C,N] from 1D/MLP path, or [B,C,H,W] from conv path.
        if x.shape[0] <= 32 and x.shape[1] >= x.shape[0]:
            x = x.permute(1, 0, 2, 3).contiguous()  # [B,T,C,N]
            return x
        return x.unsqueeze(1).flatten(3)  # [B,1,C,H*W]

    if x.dim() == 3:
        # [T,B,C] or [B,T,C] or [B,C,N]
        if x.shape[0] <= 32 and x.shape[1] >= x.shape[0]:
            return x.permute(1, 0, 2).contiguous().unsqueeze(-1)  # [B,T,C,1]
        if x.shape[1] <= 32 and x.shape[0] > x.shape[1]:
            return x.unsqueeze(-1)  # [B,T,C,1]
        return x.unsqueeze(1)  # [B,1,C,N]

    if x.dim() == 2:
        return x.unsqueeze(1).unsqueeze(-1)  # [B,1,C,1]

    raise ValueError(f"Unsupported activation shape: {tuple(x.shape)}")


def _safe_svdvals(mat: torch.Tensor) -> torch.Tensor:
    """mat: [N, T, S]. Return singular values [N, min(T,S)]."""
    # Run SVD in fp32 for numerical stability. For very old PyTorch, fall back to CPU.
    mat = mat.float()
    try:
        return torch.linalg.svdvals(mat)
    except RuntimeError:
        return torch.linalg.svdvals(mat.cpu()).to(mat.device)

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

@dataclass
class TGSRSConfig:
    # score extraction
    limit_batches: int = 6
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    hook_group_size: int = 8       # controls memory; increase if GPU memory is enough
    sv_threshold: float = 1e-6
    save_score_dir: Optional[str] = "./rank_conv_tgsrs"
    save_score_prefix: str = "tgsrs_score"

    # base pruning setting, still supports QP-style expression: "[0.2]+[0.3]*10"
    compress_rate: Union[str, List[float]] = "[0.30]*999"
    min_keep: int = 1
    skip_layer_ids_1based: Optional[List[int]] = None

    # TG-SRS weights from the thesis design
    w_svs: float = 0.40
    w_erank: float = 0.35
    w_td: float = 0.25
    w_tg: float = 1.00
    w_ac: float = 0.15
    alpha_struct: float = 1.00
    beta_task: float = 1.00

    # structure-protected pruning-rate scheduling: rate_l = clip(base * eta(type) * xi(bit), 0, max_rate(type))
    eta_by_type: Dict[str, float] = field(default_factory=lambda: {
        "attn": 0.65,   # protect q/k/v/proj/attention-related layers
        "head": 0.00,   # normally do not prune classifier output channels
        "embed": 0.75,  # protect stem/downsample/patch embedding
        "mlp": 1.00,
        "conv": 1.00,
        "dwconv": 1.00,
        "other": 1.00,
    })
    max_prune_by_type: Dict[str, float] = field(default_factory=lambda: {
        "attn": 0.30,
        "head": 0.00,
        "embed": 0.35,
        "mlp": 0.50,
        "conv": 0.50,
        "dwconv": 0.50,
        "other": 0.50,
    })
    bit_protect_omega: float = 0.30  # xi = 1 - omega * normalized_expected_bit

    # auto discovery/pairing
    activation_name_keywords: Tuple[str, ...] = ("multispike", "spike", "lif")
    strict_spike_classname: bool = False
    max_pairs: Optional[int] = None


class _TGSRSProbe:
    """Store forward outputs for one hooked activation module."""
    def __init__(self):
        self.outputs: List[torch.Tensor] = []

    def __call__(self, module, inp, out):
        out = _as_tensor_output(out)
        if out is None or not torch.is_tensor(out):
            return
        # Need output gradients for the task-guided term.
        if out.requires_grad:
            out.retain_grad()
            self.outputs.append(out)

    def clear(self):
        self.outputs.clear()


def _score_one_output(out: torch.Tensor, grad: torch.Tensor, cfg: TGSRSConfig) -> torch.Tensor:
    """
    Compute TG-SRS score for a single activation output.
    Returns per-channel score [C]. Larger = more important.
    """
    A = _to_btcx(out.detach())
    G = _to_btcx(grad.detach())

    # Align just in case a special module produces slightly different layouts.
    C = min(A.shape[2], G.shape[2])
    A = A[:, :, :C, :].float()
    G = G[:, :, :C, :].float()

    B, T, C, S = A.shape
    eps = 1e-8

    # 1) spatiotemporal rank terms: SVD on each [T, S] channel matrix.
    #    Shape transform: [B,T,C,S] -> [B*C,T,S]
    mats = A.permute(0, 2, 1, 3).reshape(B * C, T, S)
    sv = _safe_svdvals(mats)  # [B*C, min(T,S)]
    sig_count = (sv > cfg.sv_threshold).float().sum(dim=1).view(B, C).mean(dim=0)

    p = sv / (sv.sum(dim=1, keepdim=True) + eps)
    erank = torch.exp(-(p * torch.log(p + eps)).sum(dim=1)).view(B, C).mean(dim=0)

    # 2) temporal dynamic term: mean |s_{t+1}-s_t|.
    if T > 1:
        td = (A[:, 1:] - A[:, :-1]).abs().mean(dim=(0, 1, 3))
    else:
        td = torch.zeros(C, device=A.device, dtype=A.dtype)

    # 3) task-guided Taylor term: mean |activation * gradient|.
    tg = (A * G).abs().mean(dim=(0, 1, 3))

    # 4) activity-cost penalty: mean spike/activity rate.
    ac = A.abs().mean(dim=(0, 1, 3))

    # Layer-wise standardization, as described in the thesis.
    struct_score = cfg.w_svs * _zscore(sig_count) + cfg.w_erank * _zscore(erank) + cfg.w_td * _zscore(td)
    task_score = cfg.w_tg * _zscore(tg) - cfg.w_ac * _zscore(ac)
    score = cfg.alpha_struct * struct_score + cfg.beta_task * task_score

    return score.detach()


def extract_tgsrs_scores(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    activation_module_names: List[str],
    criterion: nn.Module,
    cfg: TGSRSConfig,
) -> Dict[str, np.ndarray]:
    """
    Extract per-channel TG-SRS scores for selected activation modules.
    This function uses gradients, so it performs forward + backward on calibration batches.
    """
    device = torch.device(cfg.device)
    model.to(device)
    model.eval()  # keep BN statistics stable; gradients are still tracked

    name_to_module = dict(model.named_modules())
    results: Dict[str, np.ndarray] = {}

    if cfg.save_score_dir is not None:
        os.makedirs(cfg.save_score_dir, exist_ok=True)

    group_size = max(1, int(cfg.hook_group_size))
    for start in range(0, len(activation_module_names), group_size):
        group_names = activation_module_names[start:start + group_size]
        probes: Dict[str, _TGSRSProbe] = {}
        handles = []

        for name in group_names:
            if name not in name_to_module:
                raise KeyError(f"Activation module not found: {name}")
            probe = _TGSRSProbe()
            probes[name] = probe
            handles.append(name_to_module[name].register_forward_hook(probe))

        score_sum: Dict[str, Optional[torch.Tensor]] = {name: None for name in group_names}
        score_count: Dict[str, int] = {name: 0 for name in group_names}

        used = 0
        for batch in dataloader:
            if used >= cfg.limit_batches:
                break
            if isinstance(batch, (list, tuple)):
                x = batch[0]
                y = batch[1] if len(batch) > 1 else None
            else:
                x, y = batch, None
            if y is None:
                raise RuntimeError("TG-SRS needs labels to compute task-guided gradients; dataloader must return (image, target).")

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            model.zero_grad(set_to_none=True)
            out = model(x)
            loss = criterion(out, y)
            loss.backward()

            for name, probe in probes.items():
                # A module might be called multiple times; average all calls in this batch.
                for act_out in probe.outputs:
                    if act_out.grad is None:
                        continue
                    score = _score_one_output(act_out, act_out.grad, cfg)
                    if score_sum[name] is None:
                        score_sum[name] = score
                    else:
                        # Align if a special module changes C across calls; normally not needed.
                        C = min(score_sum[name].numel(), score.numel())
                        score_sum[name] = score_sum[name][:C] + score[:C]
                    score_count[name] += 1
                probe.clear()

            used += 1

        for h in handles:
            h.remove()

        for name in group_names:
            if score_sum[name] is None or score_count[name] == 0:
                raise RuntimeError(f"No TG-SRS score extracted for activation: {name}")
            score_np = (score_sum[name] / float(score_count[name])).detach().cpu().numpy().astype(np.float32)
            results[name] = score_np

            if cfg.save_score_dir is not None:
                idx = start + group_names.index(name) + 1
                np.save(os.path.join(cfg.save_score_dir, f"{cfg.save_score_prefix}{idx}.npy"), score_np)

        torch.cuda.empty_cache() if device.type == "cuda" else None

    return results


def _layer_expected_bit(module: nn.Module) -> Optional[Tuple[float, float, float]]:
    """Return (expected_bit, min_candidate_bit, max_candidate_bit) if available."""
    if not hasattr(module, "wq"):
        return None
    try:
        bits = tuple(int(b) for b in module.wq.candidate_bits)
        if len(bits) == 0:
            return None
        b = float(module.get_nbits()) if hasattr(module, "get_nbits") else float(max(bits))
        return b, float(min(bits)), float(max(bits))
    except Exception:
        return None


def _effective_prune_rate(layer_name: str, layer_module: nn.Module, base_rate: float, cfg: TGSRSConfig) -> float:
    t = _module_type(layer_name)
    eta = float(cfg.eta_by_type.get(t, cfg.eta_by_type.get("other", 1.0)))
    max_rate = float(cfg.max_prune_by_type.get(t, cfg.max_prune_by_type.get("other", 1.0)))

    xi = 1.0
    bit_info = _layer_expected_bit(layer_module)
    if bit_info is not None:
        b, b_min, b_max = bit_info
        if b_max > b_min:
            xi = 1.0 - float(cfg.bit_protect_omega) * ((b - b_min) / (b_max - b_min))
            xi = max(1.0 - float(cfg.bit_protect_omega), min(1.0, xi))

    rate = float(base_rate) * eta * xi
    return max(0.0, min(rate, max_rate))


def _make_mask_from_score(score: np.ndarray, keep: int, device: torch.device) -> torch.Tensor:
    C = int(score.shape[0])
    keep = max(1, min(C, int(keep)))
    # Larger score = more important -> keep top-k.
    select_index = np.argsort(score)[C - keep:]
    select_index.sort()
    mask = torch.zeros(C, dtype=torch.float32, device=device)
    mask[torch.from_numpy(select_index).long().to(device)] = 1.0
    return mask


def apply_tgsrs_masks(
    model: nn.Module,
    paired_layer_names: List[str],
    paired_act_names: List[str],
    scores: Dict[str, np.ndarray],
    cfg: TGSRSConfig,
) -> Dict[str, torch.Tensor]:
    """Apply TG-SRS channel masks to quantized Conv/Linear layers."""
    cpr = parse_compress_rate(cfg.compress_rate)
    name_to_module = dict(model.named_modules())
    skip = set(cfg.skip_layer_ids_1based or [])

    def get_base_rate(i0: int) -> float:
        if len(cpr) == 0:
            return 0.0
        return float(cpr[i0]) if i0 < len(cpr) else float(cpr[-1])

    masks: Dict[str, torch.Tensor] = {}
    for layer_id_1based, (act_name, layer_name) in enumerate(zip(paired_act_names, paired_layer_names), start=1):
        if layer_id_1based in skip:
            continue
        if act_name not in scores:
            raise KeyError(f"Missing score for activation {act_name}")
        if layer_name not in name_to_module:
            raise KeyError(f"Prunable layer not found: {layer_name}")

        layer = name_to_module[layer_name]
        if not _is_prunable_layer(layer):
            raise TypeError(f"{layer_name} is not a prunable quant-wrapper layer")

        score = scores[act_name]
        C_out = int(layer.weight.shape[0])
        C = min(C_out, int(score.shape[0]))
        score = score[:C]

        base_rate = get_base_rate(layer_id_1based - 1)
        rate = _effective_prune_rate(layer_name, layer, base_rate, cfg)
        keep = max(cfg.min_keep, int(C * (1.0 - rate)))

        mask = _make_mask_from_score(score, keep=keep, device=layer.weight.device)
        if C < C_out:
            # Rare fallback: pad remaining channels as kept to avoid accidental shape mismatch.
            padded = torch.ones(C_out, dtype=mask.dtype, device=mask.device)
            padded[:C] = mask
            mask = padded

        layer.set_out_mask(mask)
        masks[layer_name] = mask

    return masks

def _get_wrapped_op(m: nn.Module):
    """
    兼容不同量化包装层：
      Conv2dLSQ / Conv1dLSQ: m.conv
      LinearLSQ: m.fc
      普通层: m 自己
    """
    if hasattr(m, "conv") and hasattr(m.conv, "weight"):
        return m.conv

    if hasattr(m, "fc") and hasattr(m.fc, "weight"):
        return m.fc

    if hasattr(m, "linear") and hasattr(m.linear, "weight"):
        return m.linear

    if hasattr(m, "weight"):
        return m

    return None


def _get_layer_weight(m: nn.Module):
    op = _get_wrapped_op(m)
    if op is None:
        return None

    w = getattr(op, "weight", None)
    if w is None:
        return None

    return w


def _get_out_channels(m: nn.Module) -> int:
    w = _get_layer_weight(m)
    if w is None:
        return 0
    return int(w.shape[0])


def _is_prunable_layer(m: nn.Module) -> bool:
    """
    判断是否是可剪枝的量化包装层。

    不再要求 m.weight 直接存在；
    只要包装层里有 conv/fc.weight，并且有 wq，就认为是候选层。
    """
    if not hasattr(m, "wq"):
        return False

    w = _get_layer_weight(m)
    if w is None:
        return False

    if not isinstance(w, torch.Tensor):
        return False

    if w.dim() < 2:
        return False

    if int(w.shape[0]) <= 1:
        return False

    return True


def discover_prunable_layers_robust(model: nn.Module) -> List[str]:
    layers = []

    for name, m in model.named_modules():
        if _is_prunable_layer(m):
            layers.append(name)

    return layers

def _infer_activation_channels(out: torch.Tensor, batch_size: int) -> Optional[int]:
    """
    更稳地从 SNN 激活输出中判断通道维 C。
    支持：
      [T,B,C,H,W]
      [B,T,C,H,W]
      [T,B,C,N]
      [B,T,C,N]
      [B,C,H,W]
      [B,C,N]
      [T,B,C]
      [B,T,C]
      [B,C]
    """
    if not isinstance(out, torch.Tensor):
        return None

    shape = tuple(out.shape)

    if out.dim() == 5:
        # [T,B,C,H,W]
        if shape[0] <= 32 and shape[1] == batch_size:
            return int(shape[2])
        # [B,T,C,H,W]
        if shape[0] == batch_size and shape[1] <= 32:
            return int(shape[2])
        # fallback
        return int(shape[2])

    if out.dim() == 4:
        # [T,B,C,N]
        if shape[0] <= 32 and shape[1] == batch_size:
            return int(shape[2])
        # [B,T,C,N]
        if shape[0] == batch_size and shape[1] <= 32:
            return int(shape[2])
        # [B,C,H,W]
        if shape[0] == batch_size:
            return int(shape[1])
        # fallback
        return int(shape[1])

    if out.dim() == 3:
        # [T,B,C]
        if shape[0] <= 32 and shape[1] == batch_size:
            return int(shape[2])
        # [B,T,C]
        if shape[0] == batch_size and shape[1] <= 32:
            return int(shape[2])
        # [B,C,N]
        if shape[0] == batch_size:
            return int(shape[1])
        return int(shape[-1])

    if out.dim() == 2:
        # [B,C]
        if shape[0] == batch_size:
            return int(shape[1])
        return int(shape[-1])

    return None


@torch.no_grad()
def _get_activation_channels_robust(
    model: nn.Module,
    dataloader,
    act_names: List[str],
    device: torch.device,
    max_batches: int = 1,
) -> Dict[str, int]:
    name_to_module = dict(model.named_modules())
    act_ch: Dict[str, int] = {}

    handles = []

    batch_size_holder = {"bs": None}

    def make_hook(nm):
        def hook(mod, inp, out):
            if isinstance(out, (tuple, list)):
                out = out[0]
            if not isinstance(out, torch.Tensor):
                return

            bs = batch_size_holder["bs"]
            if bs is None:
                return

            c = _infer_activation_channels(out, bs)
            if c is not None:
                act_ch[nm] = int(c)
        return hook

    for name in act_names:
        if name in name_to_module:
            handles.append(name_to_module[name].register_forward_hook(make_hook(name)))

    model.eval()
    model.to(device)

    for i, batch in enumerate(dataloader):
        if i >= max_batches:
            break

        x = batch[0] if isinstance(batch, (tuple, list)) else batch
        batch_size_holder["bs"] = int(x.shape[0])
        x = x.to(device, non_blocking=True)

        _ = model(x)

        try:
            from spikingjelly.clock_driven import functional
            functional.reset_net(model)
        except Exception:
            try:
                from spikingjelly.activation_based import functional
                functional.reset_net(model)
            except Exception:
                pass

    for h in handles:
        h.remove()

    return act_ch


def _common_prefix_len(a: str, b: str) -> int:
    aa = a.split(".")
    bb = b.split(".")
    n = 0
    for x, y in zip(aa, bb):
        if x == y:
            n += 1
        else:
            break
    return n


def pair_layers_to_following_acts_robust(
    model: nn.Module,
    dataloader,
    act_names: List[str],
    layer_names: List[str],
    device: torch.device,
    max_probe_batches: int = 1,
) -> Tuple[List[str], List[str]]:
    """
    更稳的配对方式：
      对每个可剪枝层，找“其后面第一个通道数相同的 spike activation”。
    这样比原来的 act -> layer 全局 shape 匹配更适合：
      conv -> bn -> lif
      q/k/v conv -> bn -> lif
      fc -> bn -> lif
    """
    name_to_module = dict(model.named_modules())
    module_order = {name: i for i, (name, _) in enumerate(model.named_modules())}

    act_ch = _get_activation_channels_robust(
        model=model,
        dataloader=dataloader,
        act_names=act_names,
        device=device,
        max_batches=max_probe_batches,
    )

    layer_out = {}
    for ln in layer_names:
        m = name_to_module[ln]
        try:
            layer_out[ln] = int(m.weight.shape[0])
        except Exception:
            continue

    used_acts = set()
    paired_act = []
    paired_layer = []

    for ln in layer_names:
        if ln not in layer_out:
            continue

        C = layer_out[ln]
        layer_idx = module_order.get(ln, -1)

        candidates = []

        for an in act_names:
            if an in used_acts:
                continue
            if an not in act_ch:
                continue
            if act_ch[an] != C:
                continue

            act_idx = module_order.get(an, -1)

            # 优先找 layer 后面的 activation
            if act_idx <= layer_idx:
                continue

            prefix_score = _common_prefix_len(ln, an)
            distance = act_idx - layer_idx

            # prefix 越大越好，distance 越小越好
            candidates.append((prefix_score, -distance, an))

        if len(candidates) == 0:
            continue

        candidates.sort(reverse=True)
        best_act = candidates[0][2]

        used_acts.add(best_act)
        paired_layer.append(ln)
        paired_act.append(best_act)

    return paired_act, paired_layer

def auto_prune_tgsrs(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    cfg: TGSRSConfig,
) -> Dict[str, torch.Tensor]:
    """
    One-call TG-SRS pruning pipeline:
      1) discover spike activations and quantized prunable layers;
      2) pair them by shape and prefix;
      3) extract TG-SRS scores with spatiotemporal rank + temporal dynamics + task gradient + activity cost;
      4) apply structure-protected pruning masks.
    """
    # Reuse the existing auto_prune discovery logic by mapping config fields.
    auto_cfg = AutoPruneConfig(
        limit_batches=cfg.limit_batches,
        device=cfg.device,
        save_rank_dir=cfg.save_score_dir,
        compress_rate=cfg.compress_rate,
        min_keep=cfg.min_keep,
        skip_layer_ids_1based=cfg.skip_layer_ids_1based,
        activation_name_keywords=cfg.activation_name_keywords,
        strict_spike_classname=cfg.strict_spike_classname,
        max_pairs=cfg.max_pairs,
    )

    act_names = discover_spike_activations(model, auto_cfg)
    layer_names = discover_prunable_layers_robust(model)

    print("[TG-SRS][Debug] discovered spike activations:", len(act_names))
    print("[TG-SRS][Debug] discovered prunable layers:", len(layer_names))

    print("[TG-SRS][Debug] first 20 activations:")
    for x in act_names[:20]:
        print("  act:", x)

    print("[TG-SRS][Debug] first 20 layers:")
    for x in layer_names[:20]:
        print("  layer:", x)

    paired_act, paired_layer = pair_layers_to_following_acts_robust(
        model=model,
        dataloader=dataloader,
        act_names=act_names,
        layer_names=layer_names,
        device=torch.device(cfg.device),
        max_probe_batches=1,
    )

    if cfg.max_pairs is not None:
        paired_act = paired_act[:cfg.max_pairs]
        paired_layer = paired_layer[:cfg.max_pairs]

    if len(paired_act) == 0 or len(paired_layer) == 0:
        raise RuntimeError("TG-SRS pairing failed: no activation-layer pairs found.")

    print("\n[TG-SRS] Found spike activations =", len(act_names))
    print("[TG-SRS] Found prunable quant layers =", len(layer_names))
    print("[TG-SRS] Will prune pairs =", len(paired_layer))
    print("[TG-SRS] Example pairing (first 10):")
    for i in range(min(10, len(paired_layer))):
        print(f"  #{i+1:02d}  act: {paired_act[i]}   ->   layer: {paired_layer[i]}")

    scores = extract_tgsrs_scores(
        model=model,
        dataloader=dataloader,
        activation_module_names=paired_act,
        criterion=criterion,
        cfg=cfg,
    )

    masks = apply_tgsrs_masks(
        model=model,
        paired_layer_names=paired_layer,
        paired_act_names=paired_act,
        scores=scores,
        cfg=cfg,
    )

    print("\n[TG-SRS] Applied masks to layers =", len(masks))
    print("[TG-SRS] Example mask stats (first 10):")
    for ln, mk in list(masks.items())[:10]:
        print(f"  {ln}: keep {int(mk.sum().item())}/{mk.numel()} | type={_module_type(ln)}")

    return masks
