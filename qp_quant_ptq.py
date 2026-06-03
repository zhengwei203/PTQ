# qp_quant_ptq_rescaw.py
# ------------------------------------------------------------
# PTQ + ReScaW implementation for QSDT / Spike-driven Transformer.
#
# Key points:
#   1) PTQ: bit-width and ReScaW gamma are fixed after calibration.
#   2) gamma is NOT the quantization step. It is a weight-range rescaling
#      coefficient: W_rescaled = W / gamma, then quantize W_rescaled, then
#      W_q = gamma * Q(W_rescaled).
#   3) Supports candidate bits such as (2, 3, 4, 6, 8).
#   4) Supports Fisher-weighted greedy mixed-precision bit allocation.
#   5) Keeps old wrapper names: Conv2dLSQ / Conv1dLSQ / LinearLSQ,
#      plus Conv2dReScaW / Conv1dReScaW / LinearReScaW aliases.
# ------------------------------------------------------------

from __future__ import annotations

import math
import copy
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from spikingjelly.clock_driven import functional as sj_functional
except Exception:  # pragma: no cover
    try:
        from spikingjelly.activation_based import functional as sj_functional
    except Exception:  # pragma: no cover
        sj_functional = None


class Qmodes:
    layer_wise = "layer_wise"
    kernel_wise = "kernel_wise"  # per-output-channel wrapper style


# ------------------------------------------------------------
# Global configuration
# ------------------------------------------------------------
_GLOBAL_PTQ_CFG: Dict[str, Any] = {
    "candidate_bits": (2, 3, 4, 6, 8),
    # ReScaW gamma mode. Recommended first checks:
    #   channel_maxabs     : robust W8 sanity baseline, close to common per-channel maxabs PTQ.
    #   layer_l1_mean      : QP-SNN-style layer-wise L1 mean gamma.
    #   channel_l1_mean    : QP-SNN-style mean gamma with channel adaptivity.
    "rescaw_gamma_mode": "channel_maxabs",
    "rescaw_clip_value": 1.0,
    # For L1-mean ReScaW, W/gamma has mean abs near 1 and max often > 1.
    # If clip=1.0 with no grid, most weights are saturated. Use clip candidates.
    # For maxabs modes, (1.0,) keeps the common max-abs quantizer.
    "grid_multipliers": (1.0,),
    "min_bits": {
        "head": 8,
        "stem": 8,
        "embed": 4,
        "attn": 3,
        "proj": 4,
        "mlp": 3,
        "conv": 3,
        "dwconv": 3,
        "other": 3,
    },
    "force_bits_by_keyword": (
        ("downsample1_1.encode_conv", 8),
        ("head", 8),
    ),
    "type_scale": {
        "attn": 1.40,
        "head": 1.50,
        "stem": 1.35,
        "embed": 1.25,
        "mlp": 1.05,
        "dwconv": 0.80,
        "conv": 1.00,
        "other": 1.00,
    },
}


def set_global_mpq_options(
    candidate_bits: Tuple[int, ...] = (2, 3, 4, 6, 8),
    gamma_percentile: float = 1.0,  # kept for old calls; unused by ReScaW modes
    hard_gumbel: bool = False,      # kept for compatibility; unused in PTQ
    min_bits: Optional[Dict[str, int]] = None,
    grid_multipliers: Optional[Tuple[float, ...]] = None,
    rescaw_gamma_mode: Optional[str] = None,
    rescaw_clip_value: Optional[float] = None,
    force_bits_by_keyword: Optional[Tuple[Tuple[str, int], ...]] = None,
):
    """Call this before building the model when changing candidate bits or gamma mode."""
    bits = tuple(sorted(set(int(b) for b in candidate_bits)))
    if len(bits) == 0:
        raise ValueError("candidate_bits must not be empty")
    _GLOBAL_PTQ_CFG["candidate_bits"] = bits

    if min_bits is not None:
        merged = dict(_GLOBAL_PTQ_CFG["min_bits"])
        merged.update({str(k): int(v) for k, v in min_bits.items()})
        _GLOBAL_PTQ_CFG["min_bits"] = merged

    if grid_multipliers is not None:
        _GLOBAL_PTQ_CFG["grid_multipliers"] = tuple(float(x) for x in grid_multipliers)

    if rescaw_gamma_mode is not None:
        _GLOBAL_PTQ_CFG["rescaw_gamma_mode"] = str(rescaw_gamma_mode)

    if rescaw_clip_value is not None:
        _GLOBAL_PTQ_CFG["rescaw_clip_value"] = float(rescaw_clip_value)

    if force_bits_by_keyword is not None:
        _GLOBAL_PTQ_CFG["force_bits_by_keyword"] = tuple((str(k), int(v)) for k, v in force_bits_by_keyword)


# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------
def _module_type(name: str) -> str:
    n = name.lower()
    if "head" in n or "classifier" in n:
        return "head"
    if "downsample1_1.encode_conv" in n:
        return "stem"
    if "downsample" in n or "embed" in n:
        return "embed"
    if any(k in n for k in ["q_conv", "k_conv", "v_conv", "attn"]):
        return "attn"
    # 残差输出 / 投影层，量化误差会直接影响残差主干
    if any(k in n for k in ["proj_conv", "pwconv2", ".conv2", "fc2_conv"]):
        return "proj"
    if any(k in n for k in ["fc1_conv", "mlp"]):
        return "mlp"
    if "dwconv" in n:
        return "dwconv"
    if "conv" in n:
        return "conv"

    return "other"


def _take_batch(batch: Any):
    if isinstance(batch, (list, tuple)):
        x = batch[0]
        y = batch[1] if len(batch) > 1 else None
        return x, y
    return batch, None


def _reset_snn_state(model: nn.Module):
    if sj_functional is not None:
        try:
            sj_functional.reset_net(model)
        except Exception:
            pass


def _reshape_gamma_for_weight(gamma: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Broadcast gamma over output channels."""
    if gamma.numel() == 1:
        return gamma.view(1)
    if w.dim() == 4:
        return gamma.view(-1, 1, 1, 1)
    if w.dim() == 3:
        return gamma.view(-1, 1, 1)
    if w.dim() == 2:
        return gamma.view(-1, 1)
    raise ValueError(f"Unsupported weight shape: {tuple(w.shape)}")


@torch.no_grad()
def _rescaw_gamma(
    w: torch.Tensor,
    mode: str = "layer_l1_mean",
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    ReScaW gamma. Gamma is a weight rescaling coefficient, not a quantization step.

    layer_l1_mean:
        gamma_l = ||W_l||_1 / |W_l| = mean(abs(W_l))
    channel_l1_mean:
        gamma_{l,c} = mean(abs(W_{l,c}))
    layer_maxabs:
        gamma_l = max(abs(W_l))
    channel_maxabs:
        gamma_{l,c} = max(abs(W_{l,c}))
    """
    mode = str(mode)
    a = w.detach().float().abs()

    if mode == "layer_l1_mean":
        return a.mean().view(1).clamp(min=eps)

    if mode == "channel_l1_mean":
        return a.flatten(1).mean(dim=1).clamp(min=eps)

    if mode == "layer_maxabs":
        return a.max().view(1).clamp(min=eps)

    if mode == "channel_maxabs":
        return a.flatten(1).max(dim=1).values.clamp(min=eps)

    raise ValueError(f"Unknown ReScaW gamma mode: {mode}")


@torch.no_grad()
# def _uniform_quantize_rescaled_weight(
#     w_rescaled: torch.Tensor,
#     bit: int,
#     clip_value: float = 1.0,
# ) -> torch.Tensor:
#     """
#     Signed uniform quantization of rescaled weights.

#     ReScaW pipeline:
#         W_rescaled = W / gamma
#         W_rescaled is clipped to [-clip_value, clip_value]
#         quantization step = clip_value / qmax
#         W_q = gamma * Q(W_rescaled)

#     This keeps zero exactly representable.
#     """
#     bit = int(bit)
#     if bit <= 0:
#         return w_rescaled

#     qmin = -(2 ** (bit - 1))
#     qmax = 2 ** (bit - 1) - 1
#     qmax = max(qmax, 1)

#     clip = float(clip_value)
#     if clip <= 0:
#         raise ValueError("clip_value must be positive")

#     step = clip / float(qmax)
#     q = torch.round(w_rescaled / step).clamp(float(qmin), float(qmax))
#     return q * step
def _uniform_quantize_rescaled_weight(
    w_rescaled: torch.Tensor,
    bit: int,
    clip_value: float = 1.0,
) -> torch.Tensor:
    """
    Symmetric signed uniform quantization for rescaled weights.

    ReScaW:
        W_rescaled = W / gamma

    Quant:
        step = clip_value / qmax
        q = round(W_rescaled / step)
        W_rescaled_q = q * step

    这样 0 是精确量化点，低比特下比 affine [-C,C] 量化稳定得多。
    """
    bit = int(bit)
    qmax = 2 ** (bit - 1) - 1

    if qmax <= 0:
        raise ValueError(f"Invalid bit width: {bit}")

    clip_value = float(clip_value)
    step = clip_value / float(qmax)

    q = torch.round(w_rescaled.float() / step)
    q = q.clamp(-qmax, qmax)

    w_q = q * step
    return w_q


@torch.no_grad()
def _quantize_weight_with_rescaw(
    w: torch.Tensor,
    bit: int,
    gamma: torch.Tensor,
    clip_value: float = 1.0,
) -> torch.Tensor:
    """
    QP-SNN-style ReScaW quantization:
        W_rescaled   = W / gamma
        W_rescaled_q = UniformQuant(W_rescaled)
        W_q          = gamma * W_rescaled_q

    gamma is not the quantization step.
    The effective step in original weight domain is gamma * clip_value / qmax.
    """
    gamma_view = _reshape_gamma_for_weight(gamma.to(device=w.device, dtype=torch.float32), w)
    w_rescaled = w.detach().float() / gamma_view
    w_rescaled_q = _uniform_quantize_rescaled_weight(w_rescaled, int(bit), float(clip_value))
    w_q = gamma_view * w_rescaled_q
    return w_q.to(dtype=w.dtype)


def _default_clip_candidates_for_mode(gamma_mode: str, clip_value: float, grid_multipliers: Tuple[float, ...]) -> Tuple[float, ...]:
    """
    Return actual clipping bounds used on W_rescaled = W / gamma.

    Important: for QP-SNN L1-mean gamma, gamma = mean(abs(W)).
    Therefore W/gamma has mean absolute value about 1 and many elements are larger than 1.
    A fixed clip_value=1.0 clips too many weights and can destroy W8 accuracy.

    For L1-mean modes, when the user leaves the old default clip=1 and grid=(1,),
    use a safe PTQ grid of normalized clipping bounds.
    For maxabs modes, W/gamma is already in [-1, 1], so clip=1 is correct.
    """
    gamma_mode = str(gamma_mode)
    grid = tuple(float(x) for x in grid_multipliers)
    clip = float(clip_value)

    if gamma_mode in ("layer_l1_mean", "channel_l1_mean"):
        if abs(clip - 1.0) < 1e-12 and len(grid) == 1 and abs(grid[0] - 1.0) < 1e-12:
            return (2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0)
        # Here grid_multipliers are treated as actual normalized clip candidates
        # when multiple values are given, e.g. --ptq_grid_multipliers 2 3 4 6 8 12.
        if len(grid) > 1:
            return grid
        return (clip * grid[0],)

    # maxabs modes: grid is a multiplier around clip_value, normally 1.0.
    return tuple(clip * g for g in grid)


@torch.no_grad()
def _quant_error_for_bit(
    w: torch.Tensor,
    bit: int,
    gamma_mode: str,
    clip_value: float,
    grid_multipliers: Tuple[float, ...] = (1.0,),
) -> Tuple[float, torch.Tensor, float]:
    """Return normalized L2 reconstruction error, gamma, and best actual clip bound."""
    denom = torch.sum(w.detach().float() ** 2).item() + 1e-12
    base_gamma = _rescaw_gamma(w, mode=str(gamma_mode))
    clip_candidates = _default_clip_candidates_for_mode(str(gamma_mode), float(clip_value), tuple(grid_multipliers))

    best_err: Optional[float] = None
    best_gamma: Optional[torch.Tensor] = None
    best_clip = float(clip_candidates[0])

    for cur_clip in clip_candidates:
        cur_clip = float(cur_clip)
        wq = _quantize_weight_with_rescaw(w, int(bit), base_gamma, cur_clip)
        err = torch.sum((w.detach().float() - wq.detach().float()) ** 2).item() / denom
        if best_err is None or err < best_err:
            best_err = float(err)
            best_gamma = base_gamma.detach().clone()
            best_clip = float(cur_clip)

    assert best_gamma is not None
    return float(best_err), best_gamma, best_clip


# ------------------------------------------------------------
# PTQ weight quantizer
# ------------------------------------------------------------
class PTQWeightQuantizer(nn.Module):
    """
    Fixed post-training ReScaW quantizer.

    It contains no learnable LSQ alpha or Gumbel logits. Calibration fixes:
      - fixed_bit
      - rescaw_gamma
      - clip multiplier
    """

    def __init__(
        self,
        weight_shape: torch.Size,
        candidate_bits: Optional[Tuple[int, ...]] = None,
        mode: str = Qmodes.kernel_wise,
        layer_name: str = "",
        gamma_mode: Optional[str] = None,
        clip_value: Optional[float] = None,
    ):
        super().__init__()
        self.weight_shape = tuple(weight_shape)
        self.mode = mode
        self.layer_name = layer_name
        self.candidate_bits = tuple(candidate_bits) if candidate_bits is not None else tuple(_GLOBAL_PTQ_CFG["candidate_bits"])
        if len(self.candidate_bits) == 0:
            self.candidate_bits = (4,)

        self.gamma_mode = str(gamma_mode if gamma_mode is not None else _GLOBAL_PTQ_CFG["rescaw_gamma_mode"])
        self.clip_value = float(clip_value if clip_value is not None else _GLOBAL_PTQ_CFG["rescaw_clip_value"])
        # Store chosen normalized clip bound as a persistent buffer.
        self.register_buffer("rescaw_clip", torch.tensor(float(self.clip_value), dtype=torch.float32))

        # Always store per-output-channel gamma. Layer-wise gamma is expanded to all channels.
        channels = int(weight_shape[0]) if len(weight_shape) >= 1 else 1

        self.register_buffer("enabled", torch.zeros(1, dtype=torch.uint8))
        self.register_buffer("fixed_bit", torch.tensor(int(self.candidate_bits[0]), dtype=torch.long))
        self.register_buffer("rescaw_gamma", torch.ones(channels, dtype=torch.float32))
        self.register_buffer("last_bit_probs", torch.zeros(len(self.candidate_bits), dtype=torch.float32))
        self.last_bit_probs[0] = 1.0

    def enable(self, flag: bool = True):
        self.enabled.fill_(1 if flag else 0)

    def disable(self):
        self.enable(False)

    def _copy_gamma(self, gamma: torch.Tensor):
        gamma = gamma.detach().to(device=self.rescaw_gamma.device, dtype=self.rescaw_gamma.dtype).flatten()
        if gamma.numel() == 1:
            gamma = gamma.expand_as(self.rescaw_gamma)
        if gamma.numel() != self.rescaw_gamma.numel():
            raise RuntimeError(
                f"[{self.layer_name}] gamma shape mismatch: "
                f"gamma={tuple(gamma.shape)}, buffer={tuple(self.rescaw_gamma.shape)}"
            )
        self.rescaw_gamma.copy_(gamma)

    def calibrate(
        self,
        w: torch.Tensor,
        bit: int,
        gamma_mode: Optional[str] = None,
        clip_value: Optional[float] = None,
        grid_multipliers: Optional[Tuple[float, ...]] = None,
    ) -> float:
        if gamma_mode is not None:
            self.gamma_mode = str(gamma_mode)
        if clip_value is not None:
            self.clip_value = float(clip_value)

        grid = tuple(_GLOBAL_PTQ_CFG["grid_multipliers"] if grid_multipliers is None else grid_multipliers)
        err, gamma, best_clip = _quant_error_for_bit(
            w=w.detach(),
            bit=int(bit),
            gamma_mode=str(self.gamma_mode),
            clip_value=float(self.clip_value),
            grid_multipliers=grid,
        )

        self.fixed_bit.fill_(int(bit))
        self.rescaw_clip.fill_(float(best_clip))
        self._copy_gamma(gamma)
        self.enabled.fill_(1)

        self.last_bit_probs.zero_()
        if int(bit) in self.candidate_bits:
            self.last_bit_probs[self.candidate_bits.index(int(bit))] = 1.0
        return float(err)

    def expected_bits(self) -> torch.Tensor:
        return self.fixed_bit.float()

    def bit_probs(self, training: bool = False) -> torch.Tensor:
        return self.last_bit_probs.detach()

    def set_temperature(self, temperature: float):
        return None

    def freeze_to_most_likely_bit(self):
        self.enable(True)

    def unfreeze_bit(self):
        self.disable()

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        if self.enabled.item() == 0:
            return w
        return _quantize_weight_with_rescaw(
            w=w,
            bit=int(self.fixed_bit.item()),
            gamma=self.rescaw_gamma.to(w.device),
            clip_value=float(self.rescaw_clip.item()),
        )


# For old code that imports MixedPrecisionLSQWeight directly.
MixedPrecisionLSQWeight = PTQWeightQuantizer


class _PTQBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("out_mask", None, persistent=True)

    def set_out_mask(self, mask_1d: Optional[torch.Tensor]):
        if mask_1d is None:
            self.out_mask = None
            return
        self.out_mask = mask_1d.detach().float()

    def _apply_out_mask_to_weight(self, w_q: torch.Tensor) -> torch.Tensor:
        if self.out_mask is None:
            return w_q
        m = self.out_mask.to(device=w_q.device, dtype=w_q.dtype)
        if w_q.dim() == 4:
            return w_q * m.view(-1, 1, 1, 1)
        if w_q.dim() == 3:
            return w_q * m.view(-1, 1, 1)
        if w_q.dim() == 2:
            return w_q * m.view(-1, 1)
        raise ValueError(f"Unsupported mask dim for weight dim={w_q.dim()}")

    def set_temperature(self, temperature: float):
        self.wq.set_temperature(temperature)

    def freeze_bit(self):
        self.wq.freeze_to_most_likely_bit()

    def unfreeze_bit(self):
        self.wq.unfreeze_bit()

    def get_nbits(self) -> float:
        return float(self.wq.expected_bits().item())

    def get_bit_probs(self) -> torch.Tensor:
        return self.wq.bit_probs(training=False).detach().cpu()

    def bit_penalty_scale(self) -> float:
        return 1.0


class Conv2dPTQ(_PTQBase):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        nbits_w=4,
        mode=Qmodes.kernel_wise,
        candidate_bits: Optional[Tuple[int, ...]] = None,
        layer_name: str = "",
        **kwargs,
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        bits = tuple(candidate_bits) if candidate_bits is not None else tuple(_GLOBAL_PTQ_CFG["candidate_bits"])
        if len(bits) == 0:
            bits = (int(nbits_w),)
        self.wq = PTQWeightQuantizer(self.conv.weight.shape, bits, mode=mode, layer_name=layer_name or "conv2d")
        self.act = nn.Identity()

    @property
    def weight(self):
        return self.conv.weight

    @property
    def bias(self):
        return self.conv.bias

    def forward(self, x):
        x = self.act(x)
        w_q = self._apply_out_mask_to_weight(self.wq(self.conv.weight))
        return F.conv2d(x, w_q, self.conv.bias, self.conv.stride, self.conv.padding, self.conv.dilation, self.conv.groups)


class Conv1dPTQ(_PTQBase):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        nbits_w=4,
        mode=Qmodes.kernel_wise,
        candidate_bits: Optional[Tuple[int, ...]] = None,
        layer_name: str = "",
        **kwargs,
    ):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        bits = tuple(candidate_bits) if candidate_bits is not None else tuple(_GLOBAL_PTQ_CFG["candidate_bits"])
        if len(bits) == 0:
            bits = (int(nbits_w),)
        self.wq = PTQWeightQuantizer(self.conv.weight.shape, bits, mode=mode, layer_name=layer_name or "conv1d")
        self.act = nn.Identity()

    @property
    def weight(self):
        return self.conv.weight

    @property
    def bias(self):
        return self.conv.bias

    def forward(self, x):
        x = self.act(x)
        w_q = self._apply_out_mask_to_weight(self.wq(self.conv.weight))
        return F.conv1d(x, w_q, self.conv.bias, self.conv.stride, self.conv.padding, self.conv.dilation, self.conv.groups)


class LinearPTQ(_PTQBase):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        nbits_w=4,
        mode=Qmodes.kernel_wise,
        candidate_bits: Optional[Tuple[int, ...]] = None,
        layer_name: str = "",
        **kwargs,
    ):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features, bias=bias)
        bits = tuple(candidate_bits) if candidate_bits is not None else tuple(_GLOBAL_PTQ_CFG["candidate_bits"])
        if len(bits) == 0:
            bits = (int(nbits_w),)
        self.wq = PTQWeightQuantizer(self.fc.weight.shape, bits, mode=mode, layer_name=layer_name or "linear")
        self.act = nn.Identity()

    @property
    def weight(self):
        return self.fc.weight

    @property
    def bias(self):
        return self.fc.bias

    def forward(self, x):
        x = self.act(x)
        w_q = self._apply_out_mask_to_weight(self.wq(self.fc.weight))
        return F.linear(x, w_q, self.fc.bias)


# Backward-compatible aliases for pruning / old model code.
Conv2dReScaW = Conv2dPTQ
Conv1dReScaW = Conv1dPTQ
LinearReScaW = LinearPTQ
QuantLayerTypes = (Conv2dPTQ, Conv1dPTQ, LinearPTQ)


def iter_quant_layers(model: nn.Module):
    for name, m in model.named_modules():
        if isinstance(m, QuantLayerTypes):
            yield name, m


def attach_layer_names(model: nn.Module):
    for name, m in iter_quant_layers(model):
        m.wq.layer_name = name


# ------------------------------------------------------------
# Compatibility functions used by old training code
# ------------------------------------------------------------
@dataclass
class MPQScheduleConfig:
    start_epoch: int = 0
    end_epoch: int = 0
    start_temperature: float = 1.0
    end_temperature: float = 1.0
    target_avg_bit: float = 4.0
    lambda_bit: float = 0.0
    lambda_entropy: float = 0.0


def update_mpq_temperature(model: nn.Module, epoch: int, cfg: Optional[MPQScheduleConfig] = None):
    return None


def freeze_all_bits(model: nn.Module):
    for _, m in iter_quant_layers(model):
        m.freeze_bit()


def unfreeze_all_bits(model: nn.Module):
    for _, m in iter_quant_layers(model):
        m.unfreeze_bit()


def mpq_regularizer(model: nn.Module, target_avg_bit: float = 4.0, lambda_bit: float = 0.0, lambda_entropy: float = 0.0):
    # PTQ has no differentiable bit logits. Return zero tensor on the model device.
    try:
        p = next(model.parameters())
        return p.new_tensor(0.0)
    except StopIteration:
        return torch.tensor(0.0)


def expected_model_bits(model: nn.Module) -> float:
    total_params = 0
    weighted = 0.0
    for _, m in iter_quant_layers(model):
        n = int(m.weight.numel())
        total_params += n
        weighted += n * float(m.get_nbits())
    return weighted / float(total_params + 1e-12)


def set_ptq_enabled(model: nn.Module, enabled: bool = True) -> int:
    n = 0
    for _, m in iter_quant_layers(model):
        m.wq.enable(bool(enabled))
        n += 1
    return n


def disable_ptq(model: nn.Module) -> int:
    return set_ptq_enabled(model, False)


def enable_ptq(model: nn.Module) -> int:
    return set_ptq_enabled(model, True)

def _disable_all_quant(model: nn.Module):
    disable_all_ptq(model, clear_masks=False)


def _enable_all_quant(model: nn.Module):
    enable_all_ptq(model)

def disable_all_ptq(model: nn.Module, clear_masks: bool = False):
    """
    Force all PTQ quantizers into full-precision mode.

    This is important when:
      1. training a full-precision baseline;
      2. resuming from a checkpoint that may contain PTQ buffers;
      3. loading a PTQ/pruned checkpoint but wanting to retrain FP32.
    """
    for _, m in iter_quant_layers(model):
        m.wq.disable()

        if clear_masks and hasattr(m, "set_out_mask"):
            m.set_out_mask(None)


def enable_all_ptq(model: nn.Module):
    """
    Enable PTQ quantization according to already calibrated fixed_bit and ptq_gamma.
    """
    for _, m in iter_quant_layers(model):
        m.wq.enable(True)


def set_full_precision_mode(
    model: nn.Module,
    clear_masks: bool = False,
    verbose: bool = True,
):
    """
    Public API used by main_finetune.py.

    When PTQ is not enabled, call this after every checkpoint loading operation.
    """
    disable_all_ptq(model, clear_masks=clear_masks)

    if verbose:
        n_layers = sum(1 for _ in iter_quant_layers(model))
        print(
            f"[FP32] PTQ disabled for {n_layers} quant-wrapper layers. "
            f"clear_masks={clear_masks}"
        )


def set_ptq_mode(model: nn.Module, verbose: bool = True):
    """
    Public API used when loading a calibrated PTQ checkpoint for evaluation/finetuning.
    """
    enable_all_ptq(model)

    if verbose:
        n_layers = sum(1 for _ in iter_quant_layers(model))
        print(f"[PTQ] PTQ enabled for {n_layers} quant-wrapper layers.")

def print_quant_state(model: nn.Module, max_lines: int = 20):
    n_total = 0
    n_enabled = 0

    print("\n[QuantState] PTQ layer states:")
    for name, m in iter_quant_layers(model):
        n_total += 1
        enabled = int(m.wq.enabled.item())
        n_enabled += enabled

        if n_total <= max_lines:
            print(
                f"  {name}: enabled={enabled}, "
                f"bit={int(m.wq.fixed_bit.item())}, "
                f"type={_module_type(name)}"
            )

    print(f"[QuantState] enabled layers: {n_enabled}/{n_total}\n")


# ------------------------------------------------------------
# Checkpoint loading helpers
# ------------------------------------------------------------
def strip_quantizer_state_dict(
    checkpoint_model: Dict[str, torch.Tensor],
    drop_all_wq_state: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Recommended when loading FP32 checkpoints trained with a different candidate_bits list.

    Actual model weights are stored in .conv.weight / .fc.weight, not in .wq.*.
    Dropping .wq.* avoids shape mismatch such as last_bit_probs [4] vs [5].
    """
    if not drop_all_wq_state:
        return dict(checkpoint_model)
    return {k: v for k, v in checkpoint_model.items() if ".wq." not in k}


def load_state_dict_ignore_mismatch(model: nn.Module, checkpoint_model: Dict[str, torch.Tensor]):
    model_state = model.state_dict()
    filtered = {}
    skipped = []
    for k, v in checkpoint_model.items():
        if k not in model_state:
            skipped.append((k, "missing_in_current_model"))
            continue
        if hasattr(v, "shape") and hasattr(model_state[k], "shape") and tuple(v.shape) != tuple(model_state[k].shape):
            skipped.append((k, f"shape_mismatch ckpt={tuple(v.shape)} model={tuple(model_state[k].shape)}"))
            continue
        filtered[k] = v
    msg = model.load_state_dict(filtered, strict=False)
    print(f"[PTQ-load] loaded keys={len(filtered)} | skipped keys={len(skipped)}")
    # for item in skipped[:30]:
    #     print("[PTQ-load] skip:", item)
    return msg


# ------------------------------------------------------------
# Fisher + mixed precision allocation
# ------------------------------------------------------------
def _normalize_dict(d: Dict[str, float], eps: float = 1e-12) -> Dict[str, float]:
    if not d:
        return {}
    vals = [float(v) for v in d.values()]
    vmin, vmax = min(vals), max(vals)
    if abs(vmax - vmin) < eps:
        return {k: 0.5 for k in d}
    return {k: (float(v) - vmin) / (vmax - vmin + eps) for k, v in d.items()}


def _total_weighted_bits(mapping: Dict[str, int], params: Dict[str, int]) -> float:
    return sum(float(params[n]) * float(b) for n, b in mapping.items())


def _avg_bit(mapping: Dict[str, int], params: Dict[str, int]) -> float:
    return _total_weighted_bits(mapping, params) / (sum(params.values()) + 1e-12)


def _nearest_allowed_ge(bits: Tuple[int, ...], value: int) -> int:
    bits = tuple(sorted(set(int(b) for b in bits)))
    for b in bits:
        if b >= int(value):
            return int(b)
    return int(bits[-1])


def _forced_bit_for_layer(name: str, force_bits_by_keyword: Tuple[Tuple[str, int], ...]) -> Optional[int]:
    for key, bit in force_bits_by_keyword:
        if str(key) in name:
            return int(bit)
    return None


def _start_bit_for_layer(name: str, bits: Tuple[int, ...], min_bits: Dict[str, int], force_bits_by_keyword: Tuple[Tuple[str, int], ...]) -> int:
    forced = _forced_bit_for_layer(name, force_bits_by_keyword)
    if forced is not None:
        return _nearest_allowed_ge(bits, forced)
    t = _module_type(name)
    min_b = int(min_bits.get(t, min_bits.get("other", min(bits))))
    return _nearest_allowed_ge(bits, min_b)


def _is_forced_layer(name: str, force_bits_by_keyword: Tuple[Tuple[str, int], ...]) -> bool:
    return _forced_bit_for_layer(name, force_bits_by_keyword) is not None


def _set_bn_eval(model: nn.Module):
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)):
            m.eval()


def empirical_fisher_scores(
    model: nn.Module,
    dataloader: Iterable,
    criterion: nn.Module,
    device: torch.device,
    num_batches: int = 32,
    use_train_mode: bool = False,
) -> Dict[str, float]:
    """
    Empirical Fisher trace proxy:
        F_l = E[ || dL/dW_l ||_2^2 ] * structure_prior(type_l)

    Quantization is disabled. By default uses eval mode to avoid BN-stat pollution.
    Gradients are still computed in eval mode.
    """
    layers = list(iter_quant_layers(model))
    if not layers:
        raise RuntimeError("No PTQ quant-wrapper layers found.")

    type_scale = dict(_GLOBAL_PTQ_CFG["type_scale"])
    raw = {name: 0.0 for name, _ in layers}

    was_training = model.training
    model.train(mode=bool(use_train_mode))
    if not use_train_mode:
        model.eval()
    _set_bn_eval(model)
    model.to(device)
    disable_ptq(model)

    used = 0
    model.zero_grad(set_to_none=True)

    for batch in dataloader:
        if used >= int(num_batches):
            break
        x, y = _take_batch(batch)
        if y is None:
            break

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        out = model(x)
        loss = criterion(out, y)
        loss.backward()

        for name, m in layers:
            g = m.weight.grad
            if g is None:
                continue
            g2 = float((g.detach().float() ** 2).sum().item()) / float(max(1, m.weight.numel()))
            t = _module_type(name)
            raw[name] += g2 * float(type_scale.get(t, type_scale.get("other", 1.0)))

        model.zero_grad(set_to_none=True)
        _reset_snn_state(model)
        used += 1

    if used == 0:
        raise RuntimeError("No calibration batches were used for Fisher estimation.")

    model.train(was_training)
    return {k: v / float(used) for k, v in raw.items()}


def allocate_bits_by_safe_downgrade(
    layers,
    fisher_scores,
    cfg,
):
    """
    Safe downgrade mixed-precision allocation.

    思路：
    1. 从所有层最高位宽开始；
    2. 每次尝试把某一层降低到下一个候选位宽；
    3. 只有当该低位宽的重构误差不超过阈值时才允许降级；
    4. 选择 damage_per_saved_bit 最小的降级操作；
    5. 直到达到 target_avg_bit，或者没有安全降级可选。
    """
    bits = tuple(sorted(set(int(b) for b in cfg.candidate_bits)))
    max_bit = int(bits[-1])
    min_bit = int(bits[0])

    params = {name: int(m.weight.numel()) for name, m in layers}
    total_params = float(sum(params.values()) + 1e-12)

    # 1. 预计算所有层、所有 bit 的重构误差
    qerr = {}
    for name, m in layers:
        qerr[name] = {}
        for b in bits:
            err, _, _ = _quant_error_for_bit(
                w=m.weight.detach(),
                bit=int(b),
                gamma_mode=str(cfg.rescaw_gamma_mode),
                clip_value=float(getattr(cfg, "rescaw_clip_value", 1.0)),
                grid_multipliers=tuple(getattr(cfg, "grid_multipliers", (1.0,))),
            )
            qerr[name][int(b)] = float(err)

    # 2. 从全 8bit 开始
    mapping = {name: max_bit for name, _ in layers}

    # 3. 强制保护层
    force_bits = getattr(cfg, "force_bits_by_keyword", ())
    for name, _ in layers:
        for key, forced_bit in force_bits:
            if key in name:
                mapping[name] = int(forced_bit)

    def avg_bit():
        return sum(params[n] * mapping[n] for n in mapping) / total_params

    def lower_bit(cur_b):
        idx = bits.index(int(cur_b))
        if idx <= 0:
            return None
        return int(bits[idx - 1])

    def min_allowed_bit(name):
        t = _module_type(name)
        min_bits_by_type = getattr(cfg, "min_bits", getattr(cfg, "min_bits_by_type", {}))
        v = int(min_bits_by_type.get(t, min_bits_by_type.get("other", min_bit)))

        # 找到 >= v 的最小候选 bit
        for b in bits:
            if b >= v:
                return int(b)
        return max_bit

    def is_forced(name):
        for key, _ in force_bits:
            if key in name:
                return True
        return False

    # 4. 低 bit 误差阈值，防止 3bit/4bit 高误差层进入模型
    max_err_by_bit = getattr(
        cfg,
        "max_err_by_bit",
        {
            2: 0.015,
            3: 0.030,
            4: 0.040,
            6: 0.010,
            8: 1.000,
        },
    )

    # 不同结构的风险权重
    type_risk = {
        "stem": 10.0,
        "head": 10.0,
        "embed": 4.0,
        "attn": 4.0,
        "proj": 5.0,
        "mlp": 2.0,
        "conv": 2.0,
        "dwconv": 2.0,
        "other": 2.0,
    }

    target_total = float(cfg.target_avg_bit) * total_params
    cur_total = sum(params[n] * mapping[n] for n in mapping)

    step_logs = []

    while cur_total > target_total + 1e-9:
        best = None

        for name, _ in layers:
            if is_forced(name):
                continue

            cur_b = int(mapping[name])
            nxt_b = lower_bit(cur_b)

            if nxt_b is None:
                continue

            if nxt_b < min_allowed_bit(name):
                continue

            err = float(qerr[name][nxt_b])
            threshold = float(max_err_by_bit.get(nxt_b, 0.02))

            # hard error guard
            if err > threshold:
                continue

            saved_bits = float(params[name]) * float(cur_b - nxt_b)
            if saved_bits <= 0:
                continue

            t = _module_type(name)
            fisher = float(fisher_scores.get(name, 0.0))
            risk_w = float(type_risk.get(t, 2.0))

            # 低 bit 的损伤估计：同时考虑 Fisher、重构误差、结构类型
            damage = risk_w * (0.20 + fisher) * err

            # 每节省一个 bit-param 所付出的损伤
            score = damage / (saved_bits + 1e-12)

            if best is None or score < best[0]:
                best = (score, name, cur_b, nxt_b, saved_bits, err, t, fisher)

        if best is None:
            print(
                f"[PTQ][safe-downgrade] No safe downgrade candidate. "
                f"Stop at avg_bit={avg_bit():.4f}, target={cfg.target_avg_bit:.4f}"
            )
            break

        _, name, old_b, new_b, saved_bits, err, t, fisher = best
        mapping[name] = int(new_b)
        cur_total -= float(saved_bits)

        if len(step_logs) < 30:
            step_logs.append((name, old_b, new_b, err, t, fisher))

    print(f"[PTQ][safe-downgrade] final avg_bit={avg_bit():.4f}, target={cfg.target_avg_bit:.4f}")
    print("[PTQ][safe-downgrade] first downgrade steps:")
    for item in step_logs[:20]:
        name, old_b, new_b, err, t, fisher = item
        print(
            f"  {name}: {old_b}->{new_b}, "
            f"err={err:.6f}, type={t}, fisher={fisher:.4f}"
        )

    return mapping, qerr

def _promote_large_error_layers(
    mapping,
    qerr,
    candidate_bits,
    max_err_by_type=None,
):
    """
    如果某层当前 bit 下的重构误差太大，自动提升到更高 bit。
    这一步可能会让实际 avg bit 略高于 target，但可以避免 Avg4 精度崩溃。
    """
    bits = tuple(sorted(set(int(b) for b in candidate_bits)))

    if max_err_by_type is None:
        max_err_by_type = {
            "stem": 0.003,
            "embed": 0.010,
            "attn": 0.010,
            "proj": 0.010,
            "mlp": 0.020,
            "conv": 0.020,
            "dwconv": 0.030,
            "head": 0.003,
            "other": 0.020,
        }

    changed = 0

    for name in list(mapping.keys()):
        t = _module_type(name)
        threshold = float(max_err_by_type.get(t, max_err_by_type["other"]))

        cur_b = int(mapping[name])
        cur_err = float(qerr[name][cur_b])

        if cur_err <= threshold:
            continue

        # 向上找第一个满足误差阈值的 bit
        new_b = cur_b
        for b in bits:
            if b <= cur_b:
                continue
            if float(qerr[name][b]) <= threshold:
                new_b = int(b)
                break

        # 如果所有 bit 都不满足，就升到最高 bit
        if new_b == cur_b:
            new_b = int(bits[-1])

        if new_b != cur_b:
            mapping[name] = new_b
            changed += 1
            print(
                f"[PTQ][err-guard] promote {name}: "
                f"{cur_b} -> {new_b}, err={cur_err:.6f}, type={t}"
            )

    print(f"[PTQ][err-guard] promoted layers = {changed}")
    return mapping

@torch.no_grad()
def apply_ptq_mapping(model: nn.Module, mapping: Dict[str, int], cfg: "PTQCalibConfig") -> Dict[str, float]:
    recon_error: Dict[str, float] = {}
    for name, m in iter_quant_layers(model):
        if name not in mapping:
            continue
        err = m.wq.calibrate(
            w=m.weight.detach(),
            bit=int(mapping[name]),
            gamma_mode=str(cfg.rescaw_gamma_mode),
            clip_value=float(cfg.rescaw_clip_value),
            grid_multipliers=tuple(cfg.grid_multipliers),
        )
        recon_error[name] = float(err)
    return recon_error


@dataclass
class PTQCalibConfig:
    candidate_bits: Tuple[int, ...] = (2, 3, 4, 6, 8)
    target_avg_bit: float = 4.0
    fisher_batches: int = 32
    fisher_use_train_mode: bool = False

    # ReScaW gamma, not quantization step.
    rescaw_gamma_mode: str = "channel_maxabs"
    rescaw_clip_value: float = 1.0
    grid_multipliers: Tuple[float, ...] = (1.0,)

    min_bits: Dict[str, int] = field(default_factory=lambda: dict(_GLOBAL_PTQ_CFG["min_bits"]))
    force_bits_by_keyword: Tuple[Tuple[str, int], ...] = field(default_factory=lambda: tuple(_GLOBAL_PTQ_CFG["force_bits_by_keyword"]))
    save_path: Optional[str] = None

    use_logit_sensitivity: bool = False
    sensitivity_batches: int = 8
    sensitivity_temperature: float = 2.0

    max_logit_damage_by_bit: Dict[int, float] = field(default_factory=lambda: {
        2: 0.010,
        3: 0.020,
        4: 0.035,
        6: 0.010,
        8: 1.000,
    })

    
# ------------------------------------------------------------
# Full PTQ calibration
# ------------------------------------------------------------
@torch.no_grad()
def compute_qerr_table(
    layers: List[Tuple[str, nn.Module]],
    cfg: "PTQCalibConfig",
) -> Dict[str, Dict[int, float]]:
    qerr: Dict[str, Dict[int, float]] = {}

    for name, m in layers:
        qerr[name] = {}

        for b in cfg.candidate_bits:
            err, _, _ = _quant_error_for_bit(
                w=m.weight.detach(),
                bit=int(b),
                gamma_mode=str(cfg.rescaw_gamma_mode),
                clip_value=float(cfg.rescaw_clip_value),
                grid_multipliers=tuple(cfg.grid_multipliers),
            )
            qerr[name][int(b)] = float(err)

    return qerr

def ptq_calibrate_model(
    model: nn.Module,
    dataloader: Iterable,
    criterion: nn.Module,
    device: torch.device,
    cfg: PTQCalibConfig,
) -> Dict[str, Any]:
    attach_layer_names(model)
    layers = list(iter_quant_layers(model))
    if not layers:
        raise RuntimeError("No PTQ quant-wrapper layers found.")

    set_global_mpq_options(
        candidate_bits=tuple(cfg.candidate_bits),
        grid_multipliers=tuple(cfg.grid_multipliers),
        rescaw_gamma_mode=str(cfg.rescaw_gamma_mode),
        rescaw_clip_value=float(cfg.rescaw_clip_value),
        min_bits=dict(cfg.min_bits),
        force_bits_by_keyword=tuple(cfg.force_bits_by_keyword),
    )
    print(f"[PTQ] active min_bits={dict(cfg.min_bits)}")
    print(f"[PTQ] active force_bits_by_keyword={tuple(cfg.force_bits_by_keyword)}")

    fisher_raw = empirical_fisher_scores(
        model=model,
        dataloader=dataloader,
        criterion=criterion,
        device=device,
        num_batches=int(cfg.fisher_batches),
        use_train_mode=bool(cfg.fisher_use_train_mode),
    )
    fisher_norm = _normalize_dict(fisher_raw)

    # 先预计算 weight reconstruction error，后面打印和保存报告都要用
    qerr = compute_qerr_table(
        layers=layers,
        cfg=cfg,
    )

    if bool(getattr(cfg, "use_logit_sensitivity", False)):
        print("[PTQ] Use logit-sensitivity allocation.")

        logit_damage = measure_layer_bit_logit_damage(
            model=model,
            dataloader=dataloader,
            layers=layers,
            cfg=cfg,
            device=device,
            max_batches=int(getattr(cfg, "sensitivity_batches", 8)),
            temperature=float(getattr(cfg, "sensitivity_temperature", 2.0)),
        )

        mapping = allocate_bits_by_logit_sensitivity(
            layers=layers,
            fisher_scores=fisher_norm,
            logit_damage=logit_damage,
            cfg=cfg,
        )

    else:
        print("[PTQ] Use fisher + weight-error allocation.")

        # 这里你可以继续用原来的函数，也可以换成 safe-downgrade
        mapping, qerr = allocate_bits_by_safe_downgrade(
            layers=layers,
            fisher_scores=fisher_raw,
            cfg=cfg,
        )

    # 最后统一把 mapping 真正写入每层 wq
    recon_error = apply_ptq_mapping(
        model=model,
        mapping=mapping,
        cfg=cfg,
    )

    params = {name: int(m.weight.numel()) for name, m in layers}
    avg = _avg_bit(mapping, params)
    bit_hist: Dict[int, int] = {}
    param_bit_hist: Dict[int, int] = {}
    for name, bit in mapping.items():
        bit_hist[int(bit)] = bit_hist.get(int(bit), 0) + 1
        param_bit_hist[int(bit)] = param_bit_hist.get(int(bit), 0) + int(params[name])

    report: Dict[str, Any] = {
        "mapping": mapping,
        "avg_bit": avg,
        "target_avg_bit": float(cfg.target_avg_bit),
        "fisher_raw": fisher_raw,
        "fisher_norm": fisher_norm,
        "qerr": qerr,
        "recon_error": recon_error,
        "selected_clip": {name: float(m.wq.rescaw_clip.item()) for name, m in layers},
        "candidate_bits": tuple(cfg.candidate_bits),
        "rescaw_gamma_mode": str(cfg.rescaw_gamma_mode),
        "rescaw_clip_value": float(cfg.rescaw_clip_value),
        "grid_multipliers": tuple(cfg.grid_multipliers),
        "min_bits": dict(cfg.min_bits),
        "force_bits_by_keyword": tuple(cfg.force_bits_by_keyword),
        "bit_hist": bit_hist,
        "param_bit_hist": param_bit_hist,
        "use_logit_sensitivity": bool(getattr(cfg, "use_logit_sensitivity", False)),
    }

    if cfg.save_path:
        torch.save(report, cfg.save_path)

    print("\n[PTQ] Calibration finished")
    print(
        f"[PTQ] layers={len(layers)} | target_avg_bit={cfg.target_avg_bit:.3f} | "
        f"actual_avg_bit={avg:.3f} | gamma_mode={cfg.rescaw_gamma_mode} | clip={cfg.rescaw_clip_value} | grid={cfg.grid_multipliers}"
    )
    print(f"[PTQ] bit_hist={bit_hist}")
    print("[PTQ] first 12 layer bit allocations:")
    for i, (name, _) in enumerate(layers[:12], start=1):
        print(
            f"  #{i:02d} {name}: bit={mapping[name]}, fisher={fisher_norm[name]:.3f}, "
            f"err={recon_error[name]:.6f}, type={_module_type(name)}"
        )

    return report


# Compatibility: old main_finetune.py may call layerwise_fisher_init.
def layerwise_fisher_init(
    model: nn.Module,
    dataloader: Iterable,
    criterion: nn.Module,
    device: torch.device,
    num_batches: int = 32,
    candidate_bits: Tuple[int, ...] = (2, 3, 4, 6, 8),
    target_avg_bit: float = 4.0,
):
    cfg = PTQCalibConfig(
        candidate_bits=tuple(candidate_bits),
        target_avg_bit=float(target_avg_bit),
        fisher_batches=int(num_batches),
        rescaw_gamma_mode=str(_GLOBAL_PTQ_CFG["rescaw_gamma_mode"]),
        rescaw_clip_value=float(_GLOBAL_PTQ_CFG["rescaw_clip_value"]),
        grid_multipliers=tuple(_GLOBAL_PTQ_CFG["grid_multipliers"]),
        min_bits=dict(_GLOBAL_PTQ_CFG["min_bits"]),
        force_bits_by_keyword=tuple(_GLOBAL_PTQ_CFG["force_bits_by_keyword"]),
    )
    return ptq_calibrate_model(model, dataloader, criterion, device, cfg)

def _reset_snn_state(model):
    try:
        from spikingjelly.clock_driven import functional
        functional.reset_net(model)
    except Exception:
        try:
            from spikingjelly.activation_based import functional
            functional.reset_net(model)
        except Exception:
            pass


def _model_logits(out):
    if isinstance(out, (tuple, list)):
        out = out[0]
    return out


def set_all_ptq_enabled(model, enabled: bool):
    for m in model.modules():
        if hasattr(m, "wq") and hasattr(m.wq, "enabled"):
            m.wq.enabled.fill_(1 if enabled else 0)


def _save_wq_state(wq):
    return {k: v.detach().clone() for k, v in wq.state_dict().items()}


def _restore_wq_state(wq, state):
    wq.load_state_dict(state, strict=False)


@torch.no_grad()
def _collect_calib_batches(dataloader, device, max_batches=8):
    batches = []
    for i, batch in enumerate(dataloader):
        if i >= max_batches:
            break

        x, y = batch[0], batch[1]
        batches.append((x.cpu(), y.cpu()))

    return batches


@torch.no_grad()
def _collect_fp_logits(model, batches, device):
    model.eval()
    set_all_ptq_enabled(model, False)

    fp_logits = []

    for x, _ in batches:
        x = x.to(device, non_blocking=True)
        out = _model_logits(model(x))
        fp_logits.append(out.detach().cpu())
        _reset_snn_state(model)

    return fp_logits


@torch.no_grad()
def measure_layer_bit_logit_damage(
    model,
    dataloader,
    layers,
    cfg,
    device,
    max_batches=8,
    temperature=2.0,
):
    """
    计算每层、每个 bit 的 logits-level sensitivity。

    对每个 layer-bit：
      1. 只开启这一层 PTQ；
      2. 其他层保持 FP32；
      3. 比较最终 logits 和 FP32 logits 的 KL + MSE。

    返回：
      damage[name][bit] = scalar
    """
    model.eval()
    device = torch.device(device)

    bits = tuple(sorted(set(int(b) for b in cfg.candidate_bits)))
    batches = _collect_calib_batches(
        dataloader=dataloader,
        device=device,
        max_batches=max_batches,
    )

    print(f"[PTQ][logit-sens] collected calibration batches = {len(batches)}")

    fp_logits = _collect_fp_logits(
        model=model,
        batches=batches,
        device=device,
    )

    damage = {}

    set_all_ptq_enabled(model, False)

    for li, (name, m) in enumerate(layers):
        damage[name] = {}

        old_state = _save_wq_state(m.wq)

        for bit in bits:
            # 8bit 也测，但通常损伤很小
            m.wq.calibrate(
                w=m.weight.detach(),
                bit=int(bit),
                gamma_mode=str(cfg.rescaw_gamma_mode),
                clip_value=float(cfg.rescaw_clip_value),
            )

            set_all_ptq_enabled(model, False)
            m.wq.enabled.fill_(1)

            total_kl = 0.0
            total_mse = 0.0
            total_n = 0

            for bi, (x, _) in enumerate(batches):
                x = x.to(device, non_blocking=True)

                q_logits = _model_logits(model(x)).detach().float().cpu()
                f_logits = fp_logits[bi].float()

                # KL
                T = float(temperature)
                log_pq = F.log_softmax(q_logits / T, dim=-1)
                pf = F.softmax(f_logits / T, dim=-1)
                kl = F.kl_div(log_pq, pf, reduction="batchmean") * (T * T)

                # normalized logits MSE
                denom = f_logits.pow(2).mean().item() + 1e-12
                mse = (q_logits - f_logits).pow(2).mean().item() / denom

                total_kl += float(kl.item())
                total_mse += float(mse)
                total_n += 1

                _reset_snn_state(model)

            d = total_kl / max(1, total_n) + 0.1 * total_mse / max(1, total_n)
            damage[name][int(bit)] = float(d)

        _restore_wq_state(m.wq, old_state)
        set_all_ptq_enabled(model, False)

        if li < 20:
            print(f"[PTQ][logit-sens] {li+1:03d} {name}: {damage[name]}")

    set_all_ptq_enabled(model, False)
    return damage

def allocate_bits_by_logit_sensitivity(
    layers,
    fisher_scores,
    logit_damage,
    cfg,
):
    """
    Budget-constrained logit-sensitivity bit allocation.

    关键修改：
    1. 从全 8bit 开始逐步降级；
    2. 用 delta_damage = damage[new_bit] - damage[current_bit]，而不是绝对 damage；
    3. 不再因为阈值太小直接停止；
    4. 必须尽量达到 target_avg_bit；
    5. 如果低 bit 会造成明显损伤，就用 soft penalty，而不是 hard reject。
    """
    bits = tuple(sorted(set(int(b) for b in cfg.candidate_bits)))
    max_bit = int(bits[-1])
    min_bit = int(bits[0])

    params = {name: int(m.weight.numel()) for name, m in layers}
    total_params = float(sum(params.values()) + 1e-12)

    mapping = {name: max_bit for name, _ in layers}

    force_bits = getattr(cfg, "force_bits_by_keyword", ())
    min_bits_by_type = getattr(cfg, "min_bits", getattr(cfg, "min_bits_by_type", {}))

    def avg_bit():
        return sum(params[n] * mapping[n] for n in mapping) / total_params

    def is_forced(name):
        return any(key in name for key, _ in force_bits)

    def forced_bit(name):
        for key, b in force_bits:
            if key in name:
                return int(b)
        return None

    def min_allowed_bit(name):
        forced = forced_bit(name)
        if forced is not None:
            return forced

        t = _module_type(name)
        v = int(min_bits_by_type.get(t, min_bits_by_type.get("other", min_bit)))

        for b in bits:
            if b >= v:
                return int(b)
        return max_bit

    def lower_bit(cur_b):
        idx = bits.index(int(cur_b))
        if idx <= 0:
            return None
        return int(bits[idx - 1])

    # 增量 damage 的软阈值，不再作为 hard reject
    delta_ref_by_bit = getattr(
        cfg,
        "delta_logit_ref_by_bit",
        {
            2: 0.20,
            3: 0.10,
            4: 0.06,
            6: 0.02,
            8: 1.00,
        },
    )

    # 结构风险权重
    type_risk = {
        "stem": 20.0,
        "head": 20.0,
        "embed": 5.0,
        "attn": 4.0,
        "proj": 5.0,
        "mlp": 2.0,
        "conv": 2.0,
        "dwconv": 1.5,
        "other": 2.0,
    }

    target_total = float(cfg.target_avg_bit) * total_params
    cur_total = sum(params[n] * mapping[n] for n in mapping)

    logs = []

    while cur_total > target_total + 1e-9:
        best = None

        for name, _ in layers:
            if is_forced(name):
                continue

            cur_b = int(mapping[name])
            nxt_b = lower_bit(cur_b)

            if nxt_b is None:
                continue

            if nxt_b < min_allowed_bit(name):
                continue

            d_cur = float(logit_damage[name][cur_b])
            d_new = float(logit_damage[name][nxt_b])

            # 核心：看相对增量，不看绝对值
            delta = max(0.0, d_new - d_cur)

            saved = float(params[name]) * float(cur_b - nxt_b)
            if saved <= 0:
                continue

            fisher = float(fisher_scores.get(name, 0.0))
            t = _module_type(name)
            risk = float(type_risk.get(t, 2.0))

            ref = float(delta_ref_by_bit.get(nxt_b, 0.05))

            # soft penalty：delta 超过参考值时加重惩罚，但不直接禁止
            penalty = 1.0 + max(0.0, delta / (ref + 1e-12))

            # 降级代价：增量 logit damage 为主，Fisher 和结构风险为辅
            cost = risk * (1.0 + 2.0 * fisher) * delta * penalty

            # 每节省一个 bit-param 的损伤
            score = cost / (saved + 1e-12)

            if best is None or score < best[0]:
                best = (
                    score,
                    name,
                    cur_b,
                    nxt_b,
                    saved,
                    d_cur,
                    d_new,
                    delta,
                    fisher,
                    t,
                )

        if best is None:
            print(
                f"[PTQ][logit-alloc] No downgrade candidate. "
                f"Stop at avg_bit={avg_bit():.4f}, target={cfg.target_avg_bit:.4f}"
            )
            break

        _, name, old_b, new_b, saved, d_cur, d_new, delta, fisher, t = best
        mapping[name] = int(new_b)
        cur_total -= float(saved)

        if len(logs) < 60:
            logs.append((name, old_b, new_b, d_cur, d_new, delta, fisher, t))

    print(f"[PTQ][logit-alloc] final avg_bit={avg_bit():.4f}, target={cfg.target_avg_bit:.4f}")
    print("[PTQ][logit-alloc] first downgrade steps:")
    for name, old_b, new_b, d_cur, d_new, delta, fisher, t in logs[:40]:
        print(
            f"  {name}: {old_b}->{new_b}, "
            f"d_cur={d_cur:.6f}, d_new={d_new:.6f}, "
            f"delta={delta:.6f}, fisher={fisher:.4f}, type={t}"
        )

    return mapping