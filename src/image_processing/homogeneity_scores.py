from __future__ import annotations

import numpy as np


def glcm_homogeneity_idm(
    img: np.ndarray,
    *,
    levels: int = 64,
    value_range: tuple[float, float] | None = None,
    distances: tuple[int, ...] = (1,),
    angles: tuple[float, ...] = (0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4),
    symmetric: bool = True,
    normed: bool = True,
    variant: str = "idm",  # "idm" or "idmn"
    eps: float = 1e-12,
    mask: np.ndarray | None = None,
) -> float:
    """
    Compute GLCM Homogeneity (Inverse Difference Moment family) from a 2D image.

    Definitions (on normalized GLCM p(i,j)):
      IDM  = sum_{i,j} p(i,j) / (1 + (i-j)^2)
      IDMN = sum_{i,j} p(i,j) / (1 + ((i-j)/Ng)^2)

    Parameters
    ----------
    img : np.ndarray
        2D input (e.g., CT slice in HU or already discretized).
    levels : int
        Number of gray levels after discretization.
    value_range : (min, max) or None
        Range used for discretization. If None, uses min/max of img within mask.
    distances : tuple[int]
        Pixel distances for GLCM offsets.
    angles : tuple[float]
        Angles in radians (0, pi/4, pi/2, 3pi/4 typical).
    symmetric : bool
        If True, add transpose counts P(i,j)+=P(j,i).
    normed : bool
        If True, normalize each GLCM to probabilities.
    variant : str
        "idm" or "idmn".
    eps : float
        Numerical stabilizer for normalization.
    mask : np.ndarray or None
        Optional boolean mask defining valid pixels.

    Returns
    -------
    float
        Homogeneity score averaged over (distances, angles).
    """
    img = np.asarray(img)
    if img.ndim != 2:
        raise ValueError("img must be 2D")

    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != img.shape:
            raise ValueError("mask must have same shape as img")
        valid = mask
    else:
        valid = np.ones_like(img, dtype=bool)

    # Discretize to {0, ..., levels-1}
    if value_range is None:
        vals = img[valid]
        if vals.size == 0:
            raise ValueError("mask selects no pixels")
        vmin, vmax = float(vals.min()), float(vals.max())
    else:
        vmin, vmax = map(float, value_range)

    if vmax <= vmin:
        # constant image -> perfectly homogeneous under any reasonable definition
        return 1.0

    # Clip then quantize
    x = np.clip(img.astype(np.float32), vmin, vmax)
    q = np.floor((x - vmin) / (vmax - vmin) * (levels - 1 + 1e-7)).astype(np.int32)

    H, W = q.shape

    # Precompute weights denominator grid for IDM/IDMN
    i = np.arange(levels, dtype=np.float32)[:, None]
    j = np.arange(levels, dtype=np.float32)[None, :]
    if variant.lower() == "idm":
        denom = 1.0 + (i - j) ** 2
    elif variant.lower() == "idmn":
        denom = 1.0 + ((i - j) / float(levels)) ** 2
    else:
        raise ValueError("variant must be 'idm' or 'idmn'")
    w = 1.0 / denom  # shape (levels, levels)

    # Angle -> (dy, dx) unit offsets (rounded to nearest integer)
    # For standard 4 angles these are exact.
    offsets = []
    for a in angles:
        dy = int(np.rint(np.sin(a)))
        dx = int(np.rint(np.cos(a)))
        if dy == 0 and dx == 0:
            raise ValueError("angle produced zero offset; choose valid angles")
        offsets.append((dy, dx))

    scores = []

    for d in distances:
        if d <= 0:
            raise ValueError("distances must be positive integers")
        for dy1, dx1 in offsets:
            dy, dx = dy1 * d, dx1 * d

            # Define overlapping windows for pairs (p -> p_shift)
            y0 = max(0, -dy)
            y1 = min(H, H - dy)
            x0 = max(0, -dx)
            x1 = min(W, W - dx)

            if y1 <= y0 or x1 <= x0:
                continue

            a_patch = q[y0:y1, x0:x1]
            b_patch = q[y0 + dy : y1 + dy, x0 + dx : x1 + dx]

            m_patch = valid[y0:y1, x0:x1] & valid[y0 + dy : y1 + dy, x0 + dx : x1 + dx]
            if not np.any(m_patch):
                continue

            a_vals = a_patch[m_patch].ravel()
            b_vals = b_patch[m_patch].ravel()

            # Accumulate GLCM counts efficiently via bincount on flattened indices
            idx = a_vals * levels + b_vals
            P = (
                np.bincount(idx, minlength=levels * levels)
                .astype(np.float64)
                .reshape(levels, levels)
            )

            if symmetric:
                P = P + P.T

            if normed:
                s = P.sum()
                if s <= eps:
                    continue
                P = P / s

            # Homogeneity = sum p(i,j) * w(i,j)
            score = float(np.sum(P * w))
            scores.append(score)

    if len(scores) == 0:
        raise ValueError("No valid GLCM pairs found (check mask / distances / angles).")

    return float(np.mean(scores))
