"""Sobel-based edge metrics on CT volumes around the Bragg peak."""

import numpy as np
from scipy.ndimage import sobel as ndimage_sobel


def _structure_tensor_metrics(J: np.ndarray) -> dict:
    """Eigendecompose a 3×3 symmetric structure tensor; return scalar descriptors.

    Returns:
        anisotropy   -- A = (λ1 − λ3) / (λ1 + λ2 + λ3) ∈ [0, 1]
        beam_angle_deg -- θ = arccos(|v1 · ẑ|) in degrees, ẑ = z-axis
        edge_energy  -- tr(J) = λ1 + λ2 + λ3
    """
    eigenvalues, eigenvectors = np.linalg.eigh(J)
    # eigh returns ascending order; index 2 is the largest eigenvalue
    lam = eigenvalues[::-1].copy()          # [λ1, λ2, λ3], λ1 ≥ λ2 ≥ λ3
    v1 = eigenvectors[:, 2]                 # eigenvector of λ1, in (z,y,x) space

    lam_sum = float(lam.sum())
    if lam_sum < 1e-12:
        return dict(anisotropy=0.0, beam_angle_deg=0.0, edge_energy=0.0)

    anisotropy = float((lam[0] - lam[2]) / lam_sum)
    edge_energy = lam_sum
    # beam travels along z-axis: ẑ = (1,0,0) in (z,y,x); v1[0] is z-component
    cos_theta = min(1.0, abs(float(v1[0])))
    beam_angle_deg = float(np.degrees(np.arccos(cos_theta)))
    return dict(anisotropy=anisotropy, beam_angle_deg=beam_angle_deg, edge_energy=edge_energy)


def compute_sobel_metrics(
    ct_hu: np.ndarray,
    flux: np.ndarray,
    z_min: float,
    z_max: float,
    flux_threshold_frac: float = 0.10,
    sobel_percentile: float = 95.0,
) -> dict:
    """Compute 3-D Sobel-based edge metrics in the Bragg-peak zone.

    The CT volume is filtered with 3-D Sobel operators along each axis.
    Gradient magnitudes are aggregated using the flux as a spatial
    weight mask so that only edges traversed by the beam contribute.

    Returns a dict with keys:
        mean_sobel_axial  -- flux-weighted mean of |G_z| in BP zone.
        p95_sobel_bp      -- 95th percentile of the total gradient
                             magnitude among flux-weighted voxels.
    """
    k_start = int(np.ceil(z_min))
    k_end = int(np.floor(z_max))

    defaults = dict(mean_sobel_axial=0.0, p95_sobel_bp=0.0, sum_sobel_bp=0.0)
    if k_end <= k_start:
        return defaults

    # -- 3-D Sobel on the full CT volume (cheap: ~ms for 160x30x30) -------
    g_z = ndimage_sobel(ct_hu, axis=0)  # axial
    g_y = ndimage_sobel(ct_hu, axis=1)
    g_x = ndimage_sobel(ct_hu, axis=2)
    grad_mag = np.sqrt(g_z**2 + g_y**2 + g_x**2)

    # -- Restrict to BP zone ----------------------------------------------
    g_z_bp = np.abs(g_z[k_start : k_end + 1])  # (N, H, W)
    grad_mag_bp = grad_mag[k_start : k_end + 1]  # (N, H, W)
    flux_bp = np.abs(flux[k_start : k_end + 1])  # (N, H, W)

    # -- Build flux weight mask (per-slice 10% threshold) -----------------
    weights = np.zeros_like(flux_bp)
    for i in range(flux_bp.shape[0]):
        f_max = flux_bp[i].max()
        if f_max < 1e-12:
            continue
        mask = flux_bp[i] >= flux_threshold_frac * f_max
        weights[i][mask] = flux_bp[i][mask]

    w_sum = weights.sum()
    if w_sum < 1e-12:
        return defaults

    # -- mean_sobel_axial: flux-weighted mean of |G_z| --------------------
    mean_sobel_axial = float(np.sum(weights * g_z_bp) / w_sum)

    # -- p95_sobel_bp: 95th percentile of grad magnitude (weighted) -------
    #    Collect gradient magnitudes at voxels with non-zero weight.
    active = weights > 0
    if active.sum() == 0:
        return defaults
    active_grads = grad_mag_bp[active]
    p95_sobel_bp = float(np.percentile(active_grads, sobel_percentile))
    sum_sobel_bp = float(np.sum(active_grads))

    return dict(
        mean_sobel_axial=mean_sobel_axial,
        p95_sobel_bp=p95_sobel_bp,
        sum_sobel_bp=sum_sobel_bp,
    )


def compute_sobel_metrics_sphere(
    ct_hu: np.ndarray,
    gt_dose: np.ndarray,
    radius_mm: float,
    resolution: tuple,
    flux: np.ndarray,
    flux_threshold_frac: float = 0.10,
    sobel_percentile: float = 95.0,
) -> dict:
    """Compute 3-D Sobel metrics inside a sphere around the Bragg peak.

    The Bragg peak is located from the ground-truth dose grid.  A
    spherical neighbourhood of the given radius (in mm) is extracted
    and the 3-D Sobel gradient magnitude is computed on the raw CT.

    Returns a dict with keys:
        mean_sobel_axial  -- mean of |G_z| inside the sphere.
        p95_sobel_bp      -- percentile of gradient magnitude inside sphere.
        sum_sobel_bp      -- sum of gradient magnitudes inside sphere.
    """
    D, H, W = ct_hu.shape
    dz, dy, dx = resolution

    # Locate Bragg peak from dose
    idd = gt_dose.sum(axis=(1, 2))
    bp_k = int(np.argmax(idd))
    # Lateral BP position from dose at BP depth
    dose_bp_slice = gt_dose[bp_k]
    if dose_bp_slice.max() > 1e-9:
        bp_y = int(np.argmax(dose_bp_slice.max(axis=1)))
        bp_x = int(np.argmax(dose_bp_slice.max(axis=0)))
    else:
        bp_y, bp_x = H // 2, W // 2

    # Build sphere mask
    rz = int(np.ceil(radius_mm / dz))
    ry = int(np.ceil(radius_mm / dy))
    rx = int(np.ceil(radius_mm / dx))

    z_lo, z_hi = max(0, bp_k - rz), min(D, bp_k + rz + 1)
    y_lo, y_hi = max(0, bp_y - ry), min(H, bp_y + ry + 1)
    x_lo, x_hi = max(0, bp_x - rx), min(W, bp_x + rx + 1)

    zz, yy, xx = np.mgrid[z_lo:z_hi, y_lo:y_hi, x_lo:x_hi]
    dist_sq = (
        ((zz - bp_k) * dz) ** 2 + ((yy - bp_y) * dy) ** 2 + ((xx - bp_x) * dx) ** 2
    )
    sphere_mask = dist_sq <= radius_mm**2

    defaults = dict(mean_sobel_axial=0.0, p95_sobel_bp=0.0, sum_sobel_bp=0.0)
    if sphere_mask.sum() == 0:
        return defaults

    # 3-D Sobel on raw CT
    g_z = ndimage_sobel(ct_hu, axis=0)
    g_y = ndimage_sobel(ct_hu, axis=1)
    g_x = ndimage_sobel(ct_hu, axis=2)
    grad_mag = np.sqrt(g_z**2 + g_y**2 + g_x**2)

    # Extract values inside sphere
    g_z_sphere = np.abs(g_z[z_lo:z_hi, y_lo:y_hi, x_lo:x_hi])[sphere_mask]
    grad_sphere = grad_mag[z_lo:z_hi, y_lo:y_hi, x_lo:x_hi][sphere_mask]

    mean_sobel_axial = float(np.mean(g_z_sphere))
    p95_sobel_bp = float(np.percentile(grad_sphere, sobel_percentile))
    sum_sobel_bp = float(np.sum(grad_sphere))

    return dict(
        mean_sobel_axial=mean_sobel_axial,
        p95_sobel_bp=p95_sobel_bp,
        sum_sobel_bp=sum_sobel_bp,
    )


def compute_structure_tensor_metrics_sphere(
    ct_hu: np.ndarray,
    gt_dose: np.ndarray,
    radius_mm: float,
    resolution: tuple,
    dose_threshold_frac: float = 0.05,
) -> dict:
    """Compute dose-weighted (DW) and threshold-masked (TH) structure-tensor
    Sobel metrics inside a sphere around the Bragg peak.

    Method DW weights each voxel by its dose value (continuous weighting).
    Method TH uses a binary mask: voxels with dose >= dose_threshold_frac *
    max(gt_dose over the full grid) contribute equally; others are excluded.

    Returns a dict with keys:
        sobel_dw_mean         -- dose-weighted mean |g| (Method DW)
        sobel_dw_anisotropy   -- edge anisotropy A in [0,1] from J_dw
        sobel_dw_beam_angle   -- dominant-edge-to-beam angle theta [degrees]
        sobel_dw_edge_energy  -- tr(J_dw) = dose-weighted sum of |g|^2
        sobel_th_mean         -- unweighted mean |g| over threshold mask
        sobel_th_anisotropy   -- edge anisotropy A from J_th
        sobel_th_beam_angle   -- dominant-edge-to-beam angle theta [degrees]
        sobel_th_edge_energy  -- tr(J_th) = sum of |g|^2 over active voxels
    """
    _zero = dict(
        sobel_dw_mean=0.0, sobel_dw_anisotropy=0.0,
        sobel_dw_beam_angle=0.0, sobel_dw_edge_energy=0.0,
        sobel_th_mean=0.0, sobel_th_anisotropy=0.0,
        sobel_th_beam_angle=0.0, sobel_th_edge_energy=0.0,
    )

    D, H, W = ct_hu.shape
    dz, dy, dx = resolution

    # Locate Bragg peak from depth-integrated dose
    idd = gt_dose.sum(axis=(1, 2))
    bp_k = int(np.argmax(idd))
    dose_bp_slice = gt_dose[bp_k]
    if dose_bp_slice.max() > 1e-9:
        bp_y = int(np.argmax(dose_bp_slice.max(axis=1)))
        bp_x = int(np.argmax(dose_bp_slice.max(axis=0)))
    else:
        bp_y, bp_x = H // 2, W // 2

    # Sphere bounding box and boolean mask
    rz = int(np.ceil(radius_mm / dz))
    ry = int(np.ceil(radius_mm / dy))
    rx = int(np.ceil(radius_mm / dx))
    z_lo, z_hi = max(0, bp_k - rz), min(D, bp_k + rz + 1)
    y_lo, y_hi = max(0, bp_y - ry), min(H, bp_y + ry + 1)
    x_lo, x_hi = max(0, bp_x - rx), min(W, bp_x + rx + 1)

    zz, yy, xx = np.mgrid[z_lo:z_hi, y_lo:y_hi, x_lo:x_hi]
    dist_sq = (
        ((zz - bp_k) * dz) ** 2
        + ((yy - bp_y) * dy) ** 2
        + ((xx - bp_x) * dx) ** 2
    )
    sphere_mask = dist_sq <= radius_mm ** 2
    if sphere_mask.sum() == 0:
        return _zero

    # 3D Sobel computed once; reused by both methods
    g_z = ndimage_sobel(ct_hu, axis=0)
    g_y = ndimage_sobel(ct_hu, axis=1)
    g_x = ndimage_sobel(ct_hu, axis=2)
    grad_mag = np.sqrt(g_z**2 + g_y**2 + g_x**2)

    # Flatten sphere voxels to 1-D arrays
    gz_sph  = g_z[z_lo:z_hi, y_lo:y_hi, x_lo:x_hi][sphere_mask]    # (N,)
    gy_sph  = g_y[z_lo:z_hi, y_lo:y_hi, x_lo:x_hi][sphere_mask]
    gx_sph  = g_x[z_lo:z_hi, y_lo:y_hi, x_lo:x_hi][sphere_mask]
    mag_sph = grad_mag[z_lo:z_hi, y_lo:y_hi, x_lo:x_hi][sphere_mask]
    d_sph   = gt_dose[z_lo:z_hi, y_lo:y_hi, x_lo:x_hi][sphere_mask]  # (N,)

    g_mat = np.stack([gz_sph, gy_sph, gx_sph], axis=1)  # (N, 3)

    # ── Method DW: continuous dose weighting ─────────────────────────────
    dose_sum = float(d_sph.sum())
    if dose_sum >= 1e-12:
        s_dw = float(np.dot(d_sph, mag_sph) / dose_sum)
        J_dw = np.einsum("i,ij,ik->jk", d_sph, g_mat, g_mat)  # (3, 3)
        tm = _structure_tensor_metrics(J_dw)
        dw = dict(
            sobel_dw_mean=s_dw,
            sobel_dw_anisotropy=tm["anisotropy"],
            sobel_dw_beam_angle=tm["beam_angle_deg"],
            sobel_dw_edge_energy=tm["edge_energy"],
        )
    else:
        dw = dict(sobel_dw_mean=0.0, sobel_dw_anisotropy=0.0,
                  sobel_dw_beam_angle=0.0, sobel_dw_edge_energy=0.0)

    # ── Method TH: binary mask at global 5 % dose threshold ──────────────
    d_grid_max = float(gt_dose.max())
    if d_grid_max > 1e-12:
        active = d_sph >= dose_threshold_frac * d_grid_max
    else:
        active = np.zeros(len(d_sph), dtype=bool)

    if active.sum() >= 1:
        g_mat_th = g_mat[active]                # (M, 3)
        s_th = float(mag_sph[active].mean())
        J_th = g_mat_th.T @ g_mat_th            # (3, 3) unweighted sum
        tm = _structure_tensor_metrics(J_th)
        th = dict(
            sobel_th_mean=s_th,
            sobel_th_anisotropy=tm["anisotropy"],
            sobel_th_beam_angle=tm["beam_angle_deg"],
            sobel_th_edge_energy=tm["edge_energy"],
        )
    else:
        th = dict(sobel_th_mean=0.0, sobel_th_anisotropy=0.0,
                  sobel_th_beam_angle=0.0, sobel_th_edge_energy=0.0)

    return {**dw, **th}
