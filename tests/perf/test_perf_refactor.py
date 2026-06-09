"""Performance-refactor tests for ``src/adota`` (backlog items #11, #12, #13).

These tests guard the optimizations that accelerate computation without
changing behaviour:

* #11 -- the causal attention mask is built once and cached, instead of being
  rebuilt on CPU and copied to the device on every forward pass.
* #12 -- the positional indices are a registered (non-persistent) buffer
  instead of a fresh ``torch.arange`` every forward.
* #13 -- ``token_size`` is a native ``int``.

Strategy (no committed golden blobs -- everything is regenerated in-run):

* **Correctness** -- an in-run A/B. We compare the optimized code against a
  faithful reimplementation of the *pre-refactor* mask path (built on CPU then
  copied to the device). Same seed and weights -> outputs must match.

* **Performance** -- the same A/B, timed, with *strict* assertions (a test fails
  if the optimized path is not faster). The strict gates live at the level the
  optimization actually affects: the causal-mask provision and the transformer
  block step (single record). The full DoTA3D_v3 step is dominated by the 3D
  conv encoder/decoder, so the mask saving is below its timing noise -- there we
  assert correctness and *report* timing rather than gate on it.

Device is auto-selected via ``nvidia-smi`` (the GPU with the most free memory),
so the tests do not collide with training jobs on other GPUs.
"""

from __future__ import annotations

import statistics
import subprocess
import types
from time import perf_counter

import torch

from src.adota.layers import PositionalEmbedding, TransformerEncoderLayerDoTA
from src.adota.models import DoTA3D_v3

# Model config mirrors scripts/config_train_adota.yaml (the trained models).
INPUT_SHAPE = (2, 160, 30, 30)
MODEL_KWARGS = dict(
    num_transformers=1,
    num_heads=4,
    num_levels=4,
    enc_features=32,
    kernel_size=3,
    convolutional_steps=2,
    conv_hidden_channels=64,
    dropout_rate=0.1,
    causal=True,
    zero_padding=True,
    last_activation=False,
    num_forward=2,
)
TOKEN_SIZE = 128  # latent (2 x 2 x enc_features=32) for the config above
SEQ_LEN = INPUT_SHAPE[1] + 1  # slices + energy token
SEED = 1234


# ── Device selection ─────────────────────────────────────────────────────────


def pick_device() -> torch.device:
    """Return the freest visible CUDA device (most free memory), else CPU."""
    if not torch.cuda.is_available():
        return torch.device("cpu")
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=10,
        )
    except Exception:
        return torch.device("cuda:0")

    best_idx, best_free, best_util = None, -1, 101
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        idx, free, util = int(parts[0]), int(parts[1]), int(parts[2])
        if idx >= torch.cuda.device_count():
            continue  # not visible to torch (e.g. CUDA_VISIBLE_DEVICES)
        if free > best_free or (free == best_free and util < best_util):
            best_idx, best_free, best_util = idx, free, util
    return torch.device(f"cuda:{best_idx}" if best_idx is not None else "cuda:0")


DEVICE = pick_device()


# ── Builders ─────────────────────────────────────────────────────────────────


def build_model(device: torch.device) -> DoTA3D_v3:
    torch.manual_seed(SEED)
    return DoTA3D_v3(input_shape=INPUT_SHAPE, **MODEL_KWARGS).to(device)


def build_layer(device: torch.device) -> TransformerEncoderLayerDoTA:
    torch.manual_seed(SEED)
    return TransformerEncoderLayerDoTA(
        embeded_dim=TOKEN_SIZE, num_heads=4, causal=True, num_forward=2
    ).to(device)


def single_record(device: torch.device):
    """One normalized (x, energy, target) record (batch size 1)."""
    gen = torch.Generator().manual_seed(SEED)
    x = torch.rand(1, *INPUT_SHAPE, generator=gen)
    e = torch.rand(1, 1, generator=gen)
    y = torch.rand(1, 1, INPUT_SHAPE[1], INPUT_SHAPE[2], INPUT_SHAPE[3], generator=gen)
    return x.to(device), e.to(device), y.to(device)


def single_token_sequence(device: torch.device) -> torch.Tensor:
    gen = torch.Generator().manual_seed(SEED)
    return torch.rand(1, SEQ_LEN, TOKEN_SIZE, generator=gen).to(device)


def _original_causal_mask(self, sequence_length: int, device: torch.device):
    """Faithful pre-refactor mask: build on CPU, then copy to the device."""
    mask = (
        torch.triu(
            torch.ones(sequence_length, sequence_length),
            diagonal=(-1) * self.num_forward,
        )
        == 1
    ).transpose(0, 1)
    mask = (
        mask.float()
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
    )
    return mask.to(device)  # host -> device copy on every call (the old cost)


def install_original_mask_path(module):
    """Swap every transformer layer in ``module`` onto the pre-refactor mask path.

    ``module`` may be a DoTA3D_v3 (iterates its transformer blocks) or a single
    TransformerEncoderLayerDoTA. Returns a restore callable.
    """
    layers = (
        list(module.transformer_layer)
        if isinstance(module, DoTA3D_v3)
        else [module]
    )
    for layer in layers:
        layer._causal_mask = types.MethodType(_original_causal_mask, layer)
        layer._mask_cache = {}

    def restore():
        for layer in layers:
            layer.__dict__.pop("_causal_mask", None)  # fall back to class method
            layer._mask_cache = {}

    return restore


def bench(fn, iters: int, warmup: int) -> float:
    """Median wall-time per call (seconds), with CUDA sync around each call."""
    cuda = DEVICE.type == "cuda"
    for _ in range(warmup):
        fn()
    if cuda:
        torch.cuda.synchronize(DEVICE)
    times = []
    for _ in range(iters):
        if cuda:
            torch.cuda.synchronize(DEVICE)
        t0 = perf_counter()
        fn()
        if cuda:
            torch.cuda.synchronize(DEVICE)
        times.append(perf_counter() - t0)
    return statistics.median(times)


def _report(name: str, t_orig: float, t_opt: float, unit: str = "ms") -> float:
    scale = 1e3 if unit == "ms" else 1e6
    speedup = t_orig / t_opt if t_opt > 0 else float("inf")
    print(
        f"\n[PERF] {name} on {DEVICE}\n"
        f"       original : {t_orig * scale:9.3f} {unit}/iter\n"
        f"       optimized: {t_opt * scale:9.3f} {unit}/iter\n"
        f"       speedup  : {speedup:6.2f}x"
    )
    return speedup


# ── Unit equivalence (#11, #12, #13) ─────────────────────────────────────────


def test_causal_mask_cached_equals_original():
    """Cached mask is bit-identical to the original CPU-built mask, and reused."""
    layer = build_layer(DEVICE)
    cached = layer._causal_mask(SEQ_LEN, DEVICE)
    original = _original_causal_mask(layer, SEQ_LEN, DEVICE)
    assert torch.equal(cached, original)
    # Second call returns the very same cached tensor (no rebuild).
    assert layer._causal_mask(SEQ_LEN, DEVICE) is cached


def test_positions_buffer_equals_arange_and_not_persistent():
    """Positions buffer equals arange, is int32, and stays out of state_dict."""
    pe = PositionalEmbedding(SEQ_LEN, token_size=TOKEN_SIZE).to(DEVICE)
    expected = torch.arange(0, SEQ_LEN, dtype=torch.int32, device=DEVICE)
    assert torch.equal(pe.positions, expected)
    assert pe.positions.dtype == torch.int32
    assert pe.positions.device.type == DEVICE.type
    # persistent=False -> not serialized (checkpoint backward-compat preserved).
    assert "positions" not in pe.state_dict()


def test_token_size_is_native_int_and_no_leaked_buffers():
    model = build_model(DEVICE)
    assert type(model.token_size) is int
    assert not any(k.endswith("positions") for k in model.state_dict())


# ── Strict perf: causal-mask provision (#11) ─────────────────────────────────


def test_causal_mask_provision_is_faster():
    layer = build_layer(DEVICE)
    layer._mask_cache = {}
    t_orig = bench(lambda: _original_causal_mask(layer, SEQ_LEN, DEVICE), iters=200, warmup=20)
    t_opt = bench(lambda: layer._causal_mask(SEQ_LEN, DEVICE), iters=200, warmup=20)
    _report("causal-mask provision", t_orig, t_opt, unit="us")
    assert t_opt < t_orig


# ── Training step (single record): transformer block, strict perf ────────────


def test_training_step_equivalence_and_perf():
    """Single-record train step through the transformer block: equal + faster."""
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    layer = build_layer(DEVICE).train()
    x = single_token_sequence(DEVICE)

    # ── Correctness: optimized vs original mask path, same dropout draws ──
    restore = install_original_mask_path(layer)
    torch.manual_seed(SEED)
    ref = layer(x).detach().clone()
    restore()
    torch.manual_seed(SEED)
    opt = layer(x).detach().clone()
    print(f"\n[CORRECTNESS] train step max|orig-opt| = {(ref - opt).abs().max():.3e}")
    assert torch.allclose(ref, opt, atol=1e-6, rtol=1e-5)

    # Gradients finite through the optimized path.
    layer.zero_grad(set_to_none=True)
    layer(x).pow(2).mean().backward()
    grads = [p.grad for p in layer.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)

    # ── Performance: forward + backward, strict speedup ──
    def step():
        layer.zero_grad(set_to_none=True)
        layer(x).pow(2).mean().backward()

    restore = install_original_mask_path(layer)
    t_orig = bench(step, iters=100, warmup=20)
    restore()
    t_opt = bench(step, iters=100, warmup=20)
    _report("training step (transformer block, fwd+bwd, single record)", t_orig, t_opt)
    assert t_opt < t_orig, (
        f"training step not faster: optimized {t_opt*1e3:.3f} ms vs "
        f"original {t_orig*1e3:.3f} ms"
    )


# ── Inference step (single record): transformer block, strict perf ───────────


def test_inference_step_equivalence_and_perf():
    """Single-record inference step through the transformer block: equal + faster."""
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    layer = build_layer(DEVICE).eval()
    x = single_token_sequence(DEVICE)

    # ── Correctness (eval is deterministic) ──
    with torch.no_grad():
        restore = install_original_mask_path(layer)
        out_orig, attn_orig = layer(x)
        out_orig, attn_orig = out_orig.clone(), attn_orig.clone()
        restore()
        out_opt, attn_opt = layer(x)
    print(f"\n[CORRECTNESS] inference step max|orig-opt| = {(out_orig - out_opt).abs().max():.3e}")
    assert out_opt.shape == out_orig.shape
    assert attn_opt.shape == attn_orig.shape  # attention output preserved
    assert torch.equal(out_orig, out_opt)

    # ── Performance: forward only, strict speedup ──
    def step():
        with torch.no_grad():
            layer(x)

    restore = install_original_mask_path(layer)
    t_orig = bench(step, iters=100, warmup=20)
    restore()
    t_opt = bench(step, iters=100, warmup=20)
    _report("inference step (transformer block, fwd, single record)", t_orig, t_opt)
    assert t_opt < t_orig, (
        f"inference step not faster: optimized {t_opt*1e3:.3f} ms vs "
        f"original {t_orig*1e3:.3f} ms"
    )


# ── Full-model end-to-end: correctness + reported (non-gated) timing ─────────


def test_full_model_equivalence_and_reported_timing():
    """End-to-end DoTA3D_v3 single-record train & infer: behaviour preserved.

    The full step is dominated by the 3D conv encoder/decoder, so the mask
    saving is below its timing noise -- timing is reported, not gated.
    """
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    model = build_model(DEVICE)
    x, e, y = single_record(DEVICE)

    # Training-step equivalence (re-seed for identical dropout draws).
    model.train()
    restore = install_original_mask_path(model)
    torch.manual_seed(SEED)
    ref = model(x, e).detach().clone()
    restore()
    torch.manual_seed(SEED)
    opt = model(x, e).detach().clone()
    print(f"\n[CORRECTNESS] full train step max|orig-opt| = {(ref - opt).abs().max():.3e}")
    assert torch.allclose(ref, opt, atol=1e-5, rtol=1e-4)

    # Inference-step equivalence.
    model.eval()
    with torch.no_grad():
        restore = install_original_mask_path(model)
        dose_orig, attn_orig = model(x, e)
        dose_orig, attn_orig = dose_orig.clone(), attn_orig.clone()
        restore()
        dose_opt, attn_opt = model(x, e)
    print(f"[CORRECTNESS] full infer step max|orig-opt| = {(dose_orig - dose_opt).abs().max():.3e}")
    assert dose_opt.shape == dose_orig.shape
    assert attn_opt.shape == attn_orig.shape
    assert torch.allclose(dose_orig, dose_opt, atol=1e-5, rtol=1e-4)

    # Reported (informational) full-step timing.
    model.train()

    def train_step():
        model.zero_grad(set_to_none=True)
        ((model(x, e) - y) ** 2).mean().backward()

    t_train = bench(train_step, iters=20, warmup=5)

    model.eval()

    def infer_step():
        with torch.no_grad():
            model(x, e)

    t_infer = bench(infer_step, iters=20, warmup=5)
    print(
        f"\n[PERF] full DoTA3D_v3 single record on {DEVICE} (informational)\n"
        f"       train step (fwd+bwd): {t_train * 1e3:9.3f} ms/iter\n"
        f"       infer step (fwd)    : {t_infer * 1e3:9.3f} ms/iter"
    )
