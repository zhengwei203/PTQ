import math
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class Qmodes:
    layer_wise = "layer_wise"
    kernel_wise = "kernel_wise"


# ------------------------------------------------------------
# Global configuration
# ------------------------------------------------------------
_GLOBAL_MPQ_CFG = {
    "candidate_bits": (4,),
    "gamma_percentile": 0.999,
    "hard_gumbel": True,
    "min_bits": {
        "attn": 6,
        "head": 8,
        "embed": 6,
        "mlp": 4,
        "dwconv": 2,
        "conv": 2,
        "other": 2,
    },
    "bit_penalty_scale": {
        "attn": 0.35,
        "head": 0.10,
        "embed": 0.25,
        "mlp": 0.80,
        "dwconv": 1.00,
        "conv": 1.00,
        "other": 1.00,
    },
    "prior_bias": {
        "attn": 0.90,
        "head": 1.20,
        "embed": 0.75,
        "mlp": 0.35,
        "dwconv": -0.10,
        "conv": 0.10,
        "other": 0.0,
    },
}


def set_global_mpq_options(
    candidate_bits: Tuple[int, ...] = (2, 4, 6, 8),
    gamma_percentile: float = 0.999,
    hard_gumbel: bool = True,
    min_bits: Optional[Dict[str, int]] = None,
):
    bits = tuple(sorted(set(int(b) for b in candidate_bits)))
    _GLOBAL_MPQ_CFG["candidate_bits"] = bits
    _GLOBAL_MPQ_CFG["gamma_percentile"] = float(gamma_percentile)
    _GLOBAL_MPQ_CFG["hard_gumbel"] = bool(hard_gumbel)
    if min_bits is not None:
        merged = dict(_GLOBAL_MPQ_CFG["min_bits"])
        merged.update(min_bits)
        _GLOBAL_MPQ_CFG["min_bits"] = merged


# ------------------------------------------------------------
# Small helpers
# ------------------------------------------------------------
def ste_round(x: torch.Tensor) -> torch.Tensor:
    return (x.round() - x).detach() + x


def ste_clip(x: torch.Tensor, x_min: float, x_max: float) -> torch.Tensor:
    y = x.clamp(x_min, x_max)
    return (y - x).detach() + x


def grad_scale(x: torch.Tensor, scale: float) -> torch.Tensor:
    y = x
    y_grad = x * scale
    return (y - y_grad).detach() + y_grad


def _reshape_like_scale(w: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    if w.dim() == 4:
        return s.view(-1, 1, 1, 1)
    if w.dim() == 3:
        return s.view(-1, 1, 1)
    if w.dim() == 2:
        return s.view(-1, 1)
    raise ValueError(f"Unsupported weight dim: {w.dim()}")


@torch.no_grad()
def _channel_stat(w: torch.Tensor, mode: str, percentile: float = 0.999) -> torch.Tensor:
    eps = 1e-8
    if mode == Qmodes.layer_wise:
        flat = w.detach().abs().reshape(-1)
        q = torch.quantile(flat, percentile).clamp(min=eps)
        return q.view(1)

    flat = w.detach().abs().reshape(w.shape[0], -1)
    q = torch.quantile(flat, percentile, dim=1).clamp(min=eps)
    return q


def _module_type(name: str) -> str:
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


# ------------------------------------------------------------
# Improved mixed-precision weight quantizer
# ------------------------------------------------------------
class MixedPrecisionLSQWeight(nn.Module):
    """
    QSDT-friendly mixed-precision weight fake quantizer.

    Key changes versus the previous version:
    1) true ReScaW-style percentile scaling on weights;
    2) hard straight-through Gumbel bit selection instead of soft averaging;
    3) type-aware minimum-bit mask for sensitive attention/head layers;
    4) no duplicate prior injection in attach_layer_names.
    """

    def __init__(
        self,
        weight_shape: torch.Size,
        candidate_bits: Optional[Tuple[int, ...]] = None,
        mode: str = Qmodes.kernel_wise,
        layer_name: str = "",
        init_temperature: float = 5.0,
        min_temperature: float = 0.3,
        percentile: Optional[float] = None,
    ):
        super().__init__()
        self.weight_shape = tuple(weight_shape)
        self.mode = mode
        self.layer_name = layer_name
        self.temperature = init_temperature
        self.min_temperature = min_temperature
        self.percentile = float(percentile if percentile is not None else _GLOBAL_MPQ_CFG["gamma_percentile"])
        self.candidate_bits = tuple(candidate_bits) if candidate_bits is not None else tuple(_GLOBAL_MPQ_CFG["candidate_bits"])

        channels = 1 if mode == Qmodes.layer_wise else int(weight_shape[0])
        n_candidates = len(self.candidate_bits)

        # alpha lives in the normalized (ReScaW-scaled) domain.
        self.alpha = nn.Parameter(torch.ones(n_candidates, channels))
        self.logits = nn.Parameter(torch.zeros(n_candidates))
        self.register_buffer("init_done", torch.zeros(1))
        self.register_buffer("fixed_bit", torch.tensor(-1, dtype=torch.long))
        self.register_buffer("last_bit_probs", torch.ones(n_candidates) / max(n_candidates, 1))
        self.register_buffer("prior_initialized", torch.zeros(1))

    def set_temperature(self, temperature: float):
        self.temperature = max(float(temperature), self.min_temperature)

    def freeze_to_most_likely_bit(self):
        with torch.no_grad():
            idx = int(torch.argmax(self.masked_logits()).item())
            self.fixed_bit.fill_(idx)

    def unfreeze_bit(self):
        self.fixed_bit.fill_(-1)

    def expected_bits(self) -> torch.Tensor:
        probs = self.bit_probs(training=False)
        bits = torch.tensor(self.candidate_bits, device=probs.device, dtype=probs.dtype)
        return (probs * bits).sum()

    def _initialize(self):
        if self.init_done.item() == 1:
            return
        with torch.no_grad():
            for i, b in enumerate(self.candidate_bits):
                qp = max(1, 2 ** (int(b) - 1) - 1)
                # Since weights are first normalized to approx [-1, 1], 1/qp is a stable initial LSQ step.
                self.alpha.data[i].fill_(1.0 / float(qp))
            self.init_done.fill_(1)

    def _min_allowed_bit(self) -> int:
        t = _module_type(self.layer_name)
        min_map = _GLOBAL_MPQ_CFG["min_bits"]
        target = int(min_map.get(t, min_map.get("other", min(self.candidate_bits))))
        valid = [b for b in self.candidate_bits if b >= target]
        return valid[0] if len(valid) > 0 else max(self.candidate_bits)

    def masked_logits(self) -> torch.Tensor:
        self._apply_type_prior_once()
        logits = self.logits
        min_bit = self._min_allowed_bit()
        allowed = torch.tensor([1.0 if b >= min_bit else 0.0 for b in self.candidate_bits], device=logits.device, dtype=logits.dtype)
        masked = logits.masked_fill(allowed == 0, torch.finfo(logits.dtype).min)
        return masked

    def _apply_type_prior_once(self):
        if self.prior_initialized.item() == 1:
            return
        t = _module_type(self.layer_name)
        bias = float(_GLOBAL_MPQ_CFG["prior_bias"].get(t, 0.0))
        with torch.no_grad():
            if len(self.candidate_bits) > 1 and abs(bias) > 0:
                bits = torch.tensor(self.candidate_bits, dtype=self.logits.dtype, device=self.logits.device)
                # Positive bias -> prefer high bits, negative bias -> prefer low bits.
                prior = bias * (bits - bits.mean()) / max(float(bits.std(unbiased=False).item()), 1e-6)
                self.logits.add_(prior)
            self.prior_initialized.fill_(1)

    def bit_probs(self, training: bool = True) -> torch.Tensor:
        masked_logits = self.masked_logits()
        if self.fixed_bit.item() >= 0:
            probs = torch.zeros_like(masked_logits)
            probs[self.fixed_bit.item()] = 1.0
            return probs

        if training and self.training and len(self.candidate_bits) > 1:
            use_hard = _GLOBAL_MPQ_CFG["hard_gumbel"] and (self.temperature <= 2.0)
            if use_hard:
                probs = F.gumbel_softmax(masked_logits, tau=self.temperature, hard=True, dim=0)
            else:
                probs = torch.softmax(masked_logits / self.temperature, dim=0)
        else:
            probs = torch.softmax(masked_logits, dim=0)

        self.last_bit_probs.copy_(probs.detach())
        return probs

    def _weight_scale(self, w: torch.Tensor) -> torch.Tensor:
        # True ReScaW-style scaling: normalize weights with percentile statistics before LSQ.
        return _channel_stat(w, self.mode, percentile=self.percentile)

    def _quant_one_bit(self, w: torch.Tensor, scale_vec: torch.Tensor, alpha_vec: torch.Tensor, bit: int) -> torch.Tensor:
        qn = -(2 ** (bit - 1))
        qp = 2 ** (bit - 1) - 1
        alpha_g = 1.0 / math.sqrt(w.numel() * max(qp, 1))

        scale = _reshape_like_scale(w, scale_vec.clamp(min=1e-8))
        alpha = _reshape_like_scale(w, grad_scale(alpha_vec.abs() + 1e-8, alpha_g))

        w_scaled = w / scale
        q = ste_round(ste_clip(w_scaled / alpha, float(qn), float(qp)))
        return q * alpha * scale

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        self._initialize()
        probs = self.bit_probs(training=True)
        scale = self._weight_scale(w)

        out = 0.0
        for i, bit in enumerate(self.candidate_bits):
            q = self._quant_one_bit(w, scale, self.alpha[i], int(bit))
            out = out + probs[i] * q
        return out


class _MPQBase(nn.Module):
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
        t = _module_type(self.wq.layer_name)
        return float(_GLOBAL_MPQ_CFG["bit_penalty_scale"].get(t, 1.0))


class Conv2dLSQ(_MPQBase):
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
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        bits = tuple(candidate_bits) if candidate_bits is not None else tuple(_GLOBAL_MPQ_CFG["candidate_bits"])
        if len(bits) == 0:
            bits = (nbits_w,) if isinstance(nbits_w, int) else tuple(nbits_w)
        self.wq = MixedPrecisionLSQWeight(self.conv.weight.shape, bits, mode=mode, layer_name=layer_name or "conv")
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


class Conv1dLSQ(_MPQBase):
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
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        bits = tuple(candidate_bits) if candidate_bits is not None else tuple(_GLOBAL_MPQ_CFG["candidate_bits"])
        if len(bits) == 0:
            bits = (nbits_w,) if isinstance(nbits_w, int) else tuple(nbits_w)
        self.wq = MixedPrecisionLSQWeight(self.conv.weight.shape, bits, mode=mode, layer_name=layer_name or "conv1d")
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


class LinearLSQ(_MPQBase):
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
        bits = tuple(candidate_bits) if candidate_bits is not None else tuple(_GLOBAL_MPQ_CFG["candidate_bits"])
        if len(bits) == 0:
            bits = (nbits_w,) if isinstance(nbits_w, int) else tuple(nbits_w)
        self.wq = MixedPrecisionLSQWeight(self.fc.weight.shape, bits, mode=mode, layer_name=layer_name or "linear")
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


# backward-compatible aliases for pruning / allocation scripts
Conv2dReScaW = Conv2dLSQ
Conv1dReScaW = Conv1dLSQ
LinearReScaW = LinearLSQ


QuantLayerTypes = (Conv2dLSQ, Conv1dLSQ, LinearLSQ)


def iter_quant_layers(model: nn.Module):
    for name, m in model.named_modules():
        if isinstance(m, QuantLayerTypes):
            yield name, m


def attach_layer_names(model: nn.Module):
    for name, m in iter_quant_layers(model):
        m.wq.layer_name = name
        # allow type prior to be re-initialized exactly once with the real module name
        m.wq.prior_initialized.zero_()
        _ = m.wq.masked_logits()


@dataclass
class MPQScheduleConfig:
    init_temp: float = 5.0
    final_temp: float = 0.5
    warmup_epochs: int = 20
    freeze_bit_epoch: int = 220
    total_epochs: int = 300


def update_mpq_temperature(model: nn.Module, epoch: int, cfg: MPQScheduleConfig):
    if epoch < cfg.warmup_epochs:
        t = cfg.init_temp
    else:
        progress = min(1.0, max(0.0, (epoch - cfg.warmup_epochs) / max(1, cfg.freeze_bit_epoch - cfg.warmup_epochs)))
        t = cfg.init_temp * ((cfg.final_temp / cfg.init_temp) ** progress)
    for _, m in iter_quant_layers(model):
        m.set_temperature(t)
    if epoch >= cfg.freeze_bit_epoch:
        for _, m in iter_quant_layers(model):
            m.freeze_bit()


def mpq_regularizer(model: nn.Module, target_avg_bit: float = 4.0, lam: float = 1e-4) -> torch.Tensor:
    bit_sum, p_sum = 0.0, 0.0
    device = None
    for _, m in iter_quant_layers(model):
        n = float(m.weight.numel()) * float(m.bit_penalty_scale())
        b = m.wq.expected_bits()
        bit_sum = bit_sum + n * b
        p_sum += n
        device = m.weight.device
    if p_sum == 0:
        return torch.tensor(0.0, device=device or "cpu")
    avg_bit = bit_sum / p_sum
    return lam * (avg_bit - target_avg_bit) ** 2


# ------------------------------------------------------------
# Fisher-style initialization for layer-wise bit allocation
# ------------------------------------------------------------
def _take_batch(batch):
    if isinstance(batch, (list, tuple)):
        x = batch[0]
        y = batch[1] if len(batch) > 1 else None
        return x, y
    return batch, None


def _default_type_scale():
    return {
        "attn": 1.40,
        "head": 1.50,
        "embed": 1.25,
        "mlp": 1.05,
        "dwconv": 0.80,
        "conv": 1.00,
        "other": 1.00,
    }


def layerwise_fisher_init(
    model: nn.Module,
    dataloader: Iterable,
    criterion: nn.Module,
    device: torch.device,
    num_batches: int = 8,
    candidate_bits: Tuple[int, ...] = (2, 4, 6, 8),
    target_avg_bit: float = 4.0,
):
    layers = list(iter_quant_layers(model))
    if not layers:
        return

    type_scale = _default_type_scale()
    scores: Dict[str, float] = {n: 0.0 for n, _ in layers}
    params = {n: float(m.weight.numel()) for n, m in layers}

    model.train()
    model.to(device)

    for p in model.parameters():
        if p.grad is not None:
            p.grad = None

    used = 0
    for batch in dataloader:
        if used >= num_batches:
            break
        x, y = _take_batch(batch)
        if y is None:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        used += 1

        for name, m in layers:
            g = m.weight.grad
            if g is None:
                continue
            t = _module_type(name)
            scores[name] += float((g.detach() ** 2).sum().item()) * float(type_scale.get(t, 1.0))
        model.zero_grad(set_to_none=True)

    if used == 0:
        return

    bits = tuple(sorted(set(int(b) for b in candidate_bits)))
    b_min = min(bits)
    mapping = {n: b_min for n, _ in layers}

    total_params = sum(params.values()) + 1e-12
    target_total_bits = float(target_avg_bit) * total_params
    cur_total_bits = sum(params[n] * mapping[n] for n in mapping)

    # greedy upgrade: allocate more bits to high-Fisher layers first
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    upgraded = True
    while upgraded and cur_total_bits < target_total_bits:
        upgraded = False
        for name, _ in ranked:
            current = mapping[name]
            higher = [b for b in bits if b > current]
            if not higher:
                continue
            nxt = higher[0]
            extra = params[name] * (nxt - current)
            if cur_total_bits + extra > target_total_bits + 1e-9:
                continue
            mapping[name] = nxt
            cur_total_bits += extra
            upgraded = True
            if cur_total_bits >= target_total_bits:
                break

    for name, m in layers:
        if tuple(m.wq.candidate_bits) != bits:
            # keep current candidate set if different from global bits
            local_bits = tuple(m.wq.candidate_bits)
        else:
            local_bits = bits
        target_bit = mapping[name]
        if target_bit not in local_bits:
            # choose nearest valid bit >= target, else fallback to max
            valid = [b for b in local_bits if b >= target_bit]
            target_bit = valid[0] if len(valid) > 0 else max(local_bits)
        logits = torch.full_like(m.wq.logits.data, -6.0)
        logits[local_bits.index(target_bit)] = 6.0
        m.wq.logits.data.copy_(logits)
