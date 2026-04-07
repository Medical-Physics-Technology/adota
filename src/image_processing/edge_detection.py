from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi


# -------------------------
# Core LoG / DoG responses
# -------------------------

def log_response(
    img: np.ndarray,
    sigma_mm: float | tuple[float, ...],
    spacing_mm: tuple[float, ...] | None = None,
    mode: str = "reflect",
    cval: float = 0.0,
) -> np.ndarray:
    img = np.asarray(img, dtype=np.float32)

    if isinstance(sigma_mm, (int, float)):
        sigma_mm = (float(sigma_mm),) * img.ndim
    if len(sigma_mm) != img.ndim:
        raise ValueError("sigma_mm must be scalar or have length img.ndim")

    if spacing_mm is not None:
        if len(spacing_mm) != img.ndim:
            raise ValueError("spacing_mm must have length img.ndim")
        sigma_px = tuple(float(s) / float(sp) for s, sp in zip(sigma_mm, spacing_mm))
    else:
        sigma_px = tuple(float(s) for s in sigma_mm)

    return ndi.gaussian_laplace(img, sigma=sigma_px, mode=mode, cval=cval)


def dog_response(
    img: np.ndarray,
    sigma_mm: float | tuple[float, ...],
    k: float = 1.6,
    spacing_mm: tuple[float, ...] | None = None,
    mode: str = "reflect",
) -> np.ndarray:
    img = np.asarray(img, dtype=np.float32)

    if isinstance(sigma_mm, (int, float)):
        sigma_mm = (float(sigma_mm),) * img.ndim

    if spacing_mm is not None:
        if len(spacing_mm) != img.ndim:
            raise ValueError("spacing_mm must have length img.ndim")
        sigma_px = tuple(float(s) / float(sp) for s, sp in zip(sigma_mm, spacing_mm))
    else:
        sigma_px = tuple(float(s) for s in sigma_mm)

    blur1 = ndi.gaussian_filter(img, sigma=sigma_px, mode=mode)
    blur2 = ndi.gaussian_filter(img, sigma=tuple(k * s for s in sigma_px), mode=mode)
    return blur2 - blur1


# -------------------------
# Thresholding helpers
# -------------------------

def _robust_mad(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float32).ravel()
    med = np.median(x)
    return 1.4826 * np.median(np.abs(x - med))


def _sigma_px_and_scalar(
    img_ndim: int,
    sigma_mm: float | tuple[float, ...],
    spacing_mm: tuple[float, ...] | None,
) -> tuple[tuple[float, ...], float]:
    if isinstance(sigma_mm, (int, float)):
        sigma_mm = (float(sigma_mm),) * img_ndim
    if len(sigma_mm) != img_ndim:
        raise ValueError("sigma_mm must be scalar or have length img.ndim")

    if spacing_mm is not None:
        if len(spacing_mm) != img_ndim:
            raise ValueError("spacing_mm must have length img.ndim")
        sigma_px = tuple(float(s) / float(sp) for s, sp in zip(sigma_mm, spacing_mm))
        sigma_scalar = float(np.mean(sigma_mm))  # mm-domain scale for σ^2 normalization
    else:
        sigma_px = tuple(float(s) for s in sigma_mm)
        sigma_scalar = float(np.mean(sigma_px))  # pixel-domain fallback
    return sigma_px, sigma_scalar


def _neighbor_offsets(ndim: int, connectivity: int) -> list[tuple[int, ...]]:
    if connectivity == 1:
        offsets = []
        for ax in range(ndim):
            off = [0] * ndim
            off[ax] = 1
            offsets.append(tuple(off))
            off[ax] = -1
            offsets.append(tuple(off))
        return offsets
    if connectivity == 2:
        offsets = [tuple(o) for o in np.ndindex(*([3] * ndim))]
        offsets = [tuple(oi - 1 for oi in o) for o in offsets]
        return [o for o in offsets if any(v != 0 for v in o)]
    raise ValueError("connectivity must be 1 or 2")


def _kill_roll_borders(mask: np.ndarray) -> None:
    ndim = mask.ndim
    for ax in range(ndim):
        sl = [slice(None)] * ndim
        sl[ax] = slice(0, 1)
        mask[tuple(sl)] = False
        sl[ax] = slice(-1, None)
        mask[tuple(sl)] = False


# -------------------------
# Edge extraction variants
# -------------------------

def zero_crossing_edges(
    log_img: np.ndarray,
    thresh: float = 0.0,
    connectivity: int = 1,
) -> np.ndarray:
    L = np.asarray(log_img, dtype=np.float32)
    ndim = L.ndim
    if ndim not in (2, 3):
        raise ValueError("zero_crossing_edges expects 2D or 3D input")

    offsets = _neighbor_offsets(ndim, connectivity)
    edges = np.zeros(L.shape, dtype=bool)

    for off in offsets:
        Ls = np.roll(L, shift=off, axis=tuple(range(ndim)))
        edges |= ((L * Ls) < 0.0) & (np.abs(L - Ls) >= float(thresh))

    _kill_roll_borders(edges)
    return edges


class LogEdgeParams:
    """
    Parameters controlling how "important" edges are selected.

    - scale_normalized: use L_norm = σ^2 * LoG (recommended for stable thresholding across σ)
    - log_k_mad: LoG jump threshold = log_k_mad * MAD(L_norm)
    - grad_percentile: gradient gate threshold is this percentile of |∇(Gσ*I)| within the image/ROI
    - hysteresis: keep weak edges only if connected to strong edges
    - low_high_ratio: low = ratio * high for hysteresis
    - connectivity: 1 (axis) or 2 (full neighborhood)
    """
    def __init__(
        self,
        *,
        scale_normalized: bool = True,
        log_k_mad: float = 6.0,
        grad_percentile: float = 95.0,
        hysteresis: bool = True,
        low_high_ratio: float = 0.5,
        connectivity: int = 1,
        mode: str = "reflect",
        cval: float = 0.0,
    ):
        self.scale_normalized = scale_normalized
        self.log_k_mad = float(log_k_mad)
        self.grad_percentile = float(grad_percentile)
        self.hysteresis = bool(hysteresis)
        self.low_high_ratio = float(low_high_ratio)
        self.connectivity = int(connectivity)
        self.mode = str(mode)
        self.cval = float(cval)


def log_edges_significant(
    img: np.ndarray,
    sigma_mm: float | tuple[float, ...],
    spacing_mm: tuple[float, ...] | None = None,
    params: LogEdgeParams | None = None,
    roi_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Integrated, parameterized LoG edge detector that returns only "important" edges.

    Outputs:
      - L_used: LoG response used for detection (scale-normalized if enabled)
      - edges: boolean edge map (after LoG jump gate + gradient gate + optional hysteresis)

    roi_mask (optional):
      - boolean mask to compute thresholds (MAD and gradient percentile) only within an ROI.
      - detection still happens on the full image; you can AND the result with roi_mask if desired.
    """
    if params is None:
        params = LogEdgeParams()

    img = np.asarray(img, dtype=np.float32)
    ndim = img.ndim
    if ndim not in (2, 3):
        raise ValueError("log_edges_significant expects 2D or 3D input")

    sigma_px, sigma_scalar = _sigma_px_and_scalar(ndim, sigma_mm, spacing_mm)

    # Smooth once for gradient gate
    I_s = ndi.gaussian_filter(img, sigma=sigma_px, mode=params.mode, cval=params.cval)

    # LoG
    L = ndi.gaussian_laplace(img, sigma=sigma_px, mode=params.mode, cval=params.cval)

    # Scale normalization for comparable amplitudes across σ
    L_used = (sigma_scalar ** 2) * L if params.scale_normalized else L

    # Threshold statistics domain (ROI-aware)
    if roi_mask is not None:
        roi_mask = np.asarray(roi_mask, dtype=bool)
        if roi_mask.shape != img.shape:
            raise ValueError("roi_mask must have the same shape as img")
        stat_vals = L_used[roi_mask]
    else:
        stat_vals = L_used

    tau_log = params.log_k_mad * _robust_mad(stat_vals)

    # Gradient magnitude gate
    grads = np.gradient(I_s)
    gmag = np.sqrt(np.sum([g.astype(np.float32) ** 2 for g in grads], axis=0))

    if roi_mask is not None:
        tau_g = np.percentile(gmag[roi_mask], params.grad_percentile)
    else:
        tau_g = np.percentile(gmag, params.grad_percentile)

    offsets = _neighbor_offsets(ndim, params.connectivity)

    zc = np.zeros(img.shape, dtype=bool)
    jump = np.zeros(img.shape, dtype=np.float32)

    for off in offsets:
        Ln = np.roll(L_used, shift=off, axis=tuple(range(ndim)))
        sc = (L_used * Ln) < 0.0
        dj = np.abs(L_used - Ln).astype(np.float32)
        zc |= sc & (dj >= tau_log)
        jump = np.maximum(jump, dj)

    _kill_roll_borders(zc)

    # Candidate edges must be zero-crossing + strong gradient
    candidates = zc & (gmag >= tau_g)

    if not params.hysteresis:
        edges = candidates
    else:
        # Hysteresis on LoG jump magnitude: seed "strong", grow into "weak"
        high = candidates & (jump >= tau_log)
        low = candidates & (jump >= (params.low_high_ratio * tau_log))

        struct = ndi.generate_binary_structure(ndim, 1 if params.connectivity == 1 else ndim)
        labels, nlab = ndi.label(low, structure=struct)
        if nlab == 0:
            edges = np.zeros_like(candidates)
        else:
            keep = np.zeros(nlab + 1, dtype=bool)
            keep_ids = np.unique(labels[high])
            keep[keep_ids] = True
            keep[0] = False
            edges = keep[labels]

    return L_used, edges


def log_edges(
    img: np.ndarray,
    sigma_mm: float | tuple[float, ...],
    spacing_mm: tuple[float, ...] | None = None,
    zero_cross_thresh: float = 0.0,
    connectivity: int = 1,
    mode: str = "reflect",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Backward-compatible wrapper.
    If you want "important edges", call log_edges_significant(...).
    """
    L = log_response(img, sigma_mm=sigma_mm, spacing_mm=spacing_mm, mode=mode)
    E = zero_crossing_edges(L, thresh=zero_cross_thresh, connectivity=connectivity)
    return L, E