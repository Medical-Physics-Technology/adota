"""Figures for the DoseCUDA pencil-beam commissioning and the MC / PBA / ADoTA comparison.

Extends the pattern of :mod:`src.figures.mc_dose_comparison` (two doses over the
CT, shared colour scale, difference panel, depth profiles) to N engines, and adds
the commissioning QC panels. Saving goes through
:func:`src.figures.axes_utils.save_figure_as_publication_formats` and colour bars
through :func:`~src.figures.axes_utils.aligned_colorbar`, as everywhere else.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.figures.axes_utils import aligned_colorbar, save_figure_as_publication_formats

__all__ = ["commissioning_run_figure", "commissioning_summary_figure", "engine_comparison_figure"]

ENGINE_STYLE = {
    "MC": dict(color="black", ls="-"),
    "PBA": dict(color="tab:blue", ls="--"),
    "ADoTA": dict(color="tab:red", ls=":"),
}


def _style(label: str, i: int) -> dict:
    return ENGINE_STYLE.get(label, dict(color=f"C{i}", ls="-."))


def commissioning_run_figure(
    wet_mm: np.ndarray,
    idd: Mapping[str, np.ndarray],
    sigma: Mapping[str, np.ndarray],
    figure_path: str,
    title: str = "",
) -> List[Path]:
    """IDD and lateral sigma vs water-equivalent depth for one water-phantom beamlet.

    ``idd`` / ``sigma`` map an engine label (``"MC"``, ``"PBA"``) to per-slice
    arrays on the shared ``wet_mm`` axis; the bottom row is the relative IDD
    difference to MC and the sigma difference, in mm.
    """
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True, layout="constrained",
                             gridspec_kw={"height_ratios": [3, 1]})
    ref_idd, ref_sig = idd.get("MC"), sigma.get("MC")
    peak = float(np.nanmax(ref_idd)) if ref_idd is not None else 1.0
    for i, label in enumerate(idd):
        st = _style(label, i)
        axes[0, 0].plot(wet_mm, idd[label] / peak, label=label, **st)
        axes[0, 1].plot(wet_mm, sigma[label], label=label, **st)
        if label != "MC" and ref_idd is not None:
            axes[1, 0].plot(wet_mm, 100 * (idd[label] - ref_idd) / peak, **st)
            axes[1, 1].plot(wet_mm, sigma[label] - ref_sig, **st)
    axes[0, 0].set_ylabel("IDD / MC peak")
    axes[0, 1].set_ylabel("lateral sigma (mm)")
    axes[1, 0].set_ylabel("dIDD (% of MC peak)")
    axes[1, 1].set_ylabel("d sigma (mm)")
    for ax in axes[1]:
        ax.set_xlabel("water-equivalent depth (mm)")
        ax.axhline(0, color="grey", lw=0.8)
    for ax in axes.ravel():
        ax.grid(alpha=0.3)
    axes[0, 0].legend()
    if title:
        fig.suptitle(title)
    paths = save_figure_as_publication_formats(fig, figure_path)
    plt.close(fig)
    return paths


def commissioning_summary_figure(
    rows: Sequence[Mapping[str, float]],
    figure_path: str,
    gamma_key: Optional[str] = None,
) -> List[Path]:
    """Validation metrics vs energy: R80 error, peak-IDD ratio, sigma error, gamma.

    ``rows`` are the per-run dicts of the validation CSV (keys ``energy``,
    ``dr80_mm``, ``idd_peak_ratio``, ``max_abs_dsigma_mm``, ``node`` and
    optionally ``gamma_key``); node energies and interpolated energies are drawn
    with different markers.
    """
    panels = [("dr80_mm", "R80 PBA - MC (mm)"), ("idd_peak_ratio", "IDD peak PBA / MC"),
              ("max_abs_dsigma_mm", "max |d sigma| (mm)")]
    if gamma_key:
        panels.append((gamma_key, f"gamma pass rate (%), {gamma_key.replace('gpr_', '')}"))
    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 4.2), layout="constrained")
    for ax, (key, label) in zip(np.atleast_1d(axes), panels):
        for node, marker, name in ((True, "o", "commissioned node"), (False, "s", "interpolated energy")):
            pts = [(r["energy"], r[key]) for r in rows if bool(r.get("node")) == node and key in r]
            if pts:
                e, v = zip(*sorted(pts))
                ax.plot(e, v, marker=marker, ls="none", label=name)
        ax.set_xlabel("energy (MeV)")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
    np.atleast_1d(axes)[0].legend()
    paths = save_figure_as_publication_formats(fig, figure_path)
    plt.close(fig)
    return paths


def _bev_planes(volume: np.ndarray, iz: int, iy: int):
    """``(z, y, x)`` crop -> the two longitudinal planes through the peak (beam along x)."""
    return volume[iz, :, :], volume[:, iy, :]


def engine_comparison_figure(
    ct: np.ndarray,
    doses: Dict[str, np.ndarray],
    figure_path: str,
    reference: str = "MC",
    spacing_mm: float = 1.0,
    title: str = "",
    metrics_text: Optional[Mapping[str, str]] = None,
    dose_scale: float = 1.0,
    dose_unit: str = "MCsquare eV/g per proton",
) -> List[Path]:
    """N dose engines over the same CT crop, with differences to ``reference``.

    Rows: one per engine (dose over CT, two longitudinal planes through the
    reference's peak voxel, a single shared colour scale), one difference row per
    non-reference engine (symmetric scale shared across rows), and a profile row
    (integrated depth dose and the lateral profile at the reference's Bragg peak).
    All arrays are ``(z, y, x)`` with the beam along x.

    Doses arrive on MCsquare's beamlet scale (eV/g per simulated proton: ADoTA is
    trained on it, the pencil beam runs at :meth:`~src.pencil_beam.engine.PencilBeamEngine.mcsquare_spot_weight`)
    and are displayed as ``dose * dose_scale`` in ``dose_unit``; for Gy per MU pass
    ``dose_scale = mcsquare_to_gy_per_mu(1.0, energy, bdl)`` and ``dose_unit = "Gy per MU"``.
    """
    doses = {label: np.asarray(dose, dtype=np.float64) * dose_scale for label, dose in doses.items()}
    labels = list(doses)
    others = [k for k in labels if k != reference]
    ref = doses[reference]
    iz, iy, ix = np.unravel_index(int(np.argmax(ref)), ref.shape)
    vmax = max(float(np.max(d)) for d in doses.values()) or 1.0
    dmax = max([float(np.max(np.abs(doses[k] - ref))) for k in others] + [1e-12])
    n_rows = len(labels) + len(others) + 1
    extent = [0, ct.shape[2] * spacing_mm, ct.shape[1] * spacing_mm, 0]
    fig, axes = plt.subplots(n_rows, 2, figsize=(16, 2.6 * n_rows + 1), layout="constrained",
                             gridspec_kw={"height_ratios": [1] * (n_rows - 1) + [1.6]})
    plane_names = (f"axial (z = {iz})", f"coronal (y = {iy})")
    for r, label in enumerate(labels):
        for c, (ct_plane, d_plane) in enumerate(zip(_bev_planes(ct, iz, iy), _bev_planes(doses[label], iz, iy))):
            ax = axes[r, c]
            ax.imshow(ct_plane, cmap="gray", aspect="auto", extent=extent)
            im = ax.imshow(np.ma.masked_less(d_plane, 0.01 * vmax), cmap="jet", alpha=0.6, vmin=0, vmax=vmax,
                           aspect="auto", extent=extent)
            ax.set_title(f"{label} | {plane_names[c]}", fontsize=11)
            if c == 1:
                aligned_colorbar(fig, im, ax, f"dose ({dose_unit})", label_fontsize=10, tick_fontsize=8,
                                 label_coords=(3.5, 0.5))
    for k, label in enumerate(others):
        r = len(labels) + k
        diff = doses[label] - ref
        for c, plane in enumerate(_bev_planes(diff, iz, iy)):
            ax = axes[r, c]
            im = ax.imshow(plane, cmap="RdBu_r", vmin=-dmax, vmax=dmax, aspect="auto", extent=extent)
            ax.set_title(f"{label} - {reference} | {plane_names[c]}", fontsize=11)
            if c == 1:
                aligned_colorbar(fig, im, ax, f"dose diff. ({dose_unit})", label_fontsize=10, tick_fontsize=8,
                                 label_coords=(3.5, 0.5))
    depth = (np.arange(ref.shape[2]) + 0.5) * spacing_mm
    lateral = (np.arange(ref.shape[1]) + 0.5 - (iy + 0.5)) * spacing_mm
    ax_idd, ax_lat = axes[-1]
    for i, label in enumerate(labels):
        st = _style(label, i)
        ax_idd.plot(depth, doses[label].sum(axis=(0, 1)) * spacing_mm ** 2, label=label, **st)
        ax_lat.plot(lateral, doses[label][iz, :, ix], label=label, **st)
    ax_idd.set_xlabel("depth along the beam (mm)")
    ax_idd.set_ylabel(f"integrated depth dose ({dose_unit} x mm$^2$)")
    ax_lat.set_xlabel(f"lateral position at the {reference} Bragg peak (mm)")
    ax_lat.set_ylabel(f"dose ({dose_unit})")
    for ax in (ax_idd, ax_lat):
        ax.grid(alpha=0.3)
        ax.legend()
    if metrics_text:
        ax_idd.text(0.01, 0.98, "\n".join(f"{k}: {v}" for k, v in metrics_text.items()),
                    transform=ax_idd.transAxes, va="top", fontsize=9, family="monospace",
                    bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))
    if title:
        fig.suptitle(title)
    paths = save_figure_as_publication_formats(fig, figure_path)
    plt.close(fig)
    return paths
