"""ADoTA vs MCsquare dose-volume histogram (DVH) comparison.

Two outputs, kept separate:

* :func:`dvh_comparison_figure` -- the DVH curve plot only (no metric table):
  structures by colour, the two doses by line style (solid = ADoTA, dashed =
  MCsquare).
* :func:`write_dvh_metrics_json` -- a per-plan ``dvh_metrics.json`` with
  **structure-type-dependent** clinical metrics (target: Dmin/Dmean/Dmax/D2/D95/D98;
  OARs: Dmin/Dmean/Dmax/D2) for both doses plus their difference, with embedded
  descriptions and units.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from src.beamlets.dvh import DVH
from src.figures.single_beam import save_figure_as_publication_formats

__all__ = [
    "compute_structure_dvhs",
    "dvh_metrics",
    "write_dvh_metrics_json",
    "dvh_comparison_figure",
]

_PALETTE = ["tab:blue", "tab:red", "tab:green", "tab:orange", "tab:purple", "tab:brown"]

# Clinically meaningful metrics per structure type.
_TARGET_METRICS = ("Dmin", "Dmean", "Dmax", "D2", "D95", "D98")
_OAR_METRICS = ("Dmin", "Dmean", "Dmax", "D2")

_METRIC_DEFINITIONS = {
    "Dmin": "Minimum dose in the structure (Gy).",
    "Dmean": "Mean dose in the structure (Gy).",
    "Dmax": "Maximum dose in the structure (Gy).",
    "D2": (
        "Near-maximum dose: the dose received by the hottest 2% of the structure "
        "volume (Gy). A robust surrogate for Dmax (far less single-voxel noise) and "
        "the standard hot-spot / high-dose indicator for both targets and OARs."
    ),
    "D95": (
        "Dose covering at least 95% of the target volume (Gy); a minimum-coverage "
        "indicator, commonly required to be 90-95% of the prescribed dose."
    ),
    "D98": (
        "Near-minimum dose covering 98% of the target volume (Gy); indicates "
        "adequate dose to the edges of the target."
    ),
}

# Print-readable font sizes.
_FS_TICK = 16
_FS_LEGEND = 16
_FS_AXIS = 17
_FS_TITLE = 18


def _structure_type(name: str, target_keyword: str) -> str:
    """Classify a structure as ``"target"`` or ``"OAR"`` by its name."""
    return "target" if target_keyword.lower() in name.lower() else "OAR"


def _ordered_names(structures, target_keyword: str) -> list:
    """Target first, then the remaining structures in name order."""
    names = list(structures)
    target = [n for n in names if target_keyword.lower() in n.lower()]
    others = sorted(n for n in names if n not in target)
    return target + others


def compute_structure_dvhs(
    structures: Dict[str, np.ndarray],
    dose_a: np.ndarray,
    dose_b: np.ndarray,
    spacing: Sequence[float],
    max_dvh: Optional[float] = None,
) -> Dict[str, Tuple[DVH, DVH]]:
    """Compute ``(DVH_a, DVH_b)`` per structure on a shared dose axis.

    Args:
        structures: Oriented boolean masks ``name -> (z, y, x)``.
        dose_a: First dose grid (e.g. ADoTA), Gy, ``(z, y, x)``.
        dose_b: Reference dose grid (e.g. MCsquare), Gy, same shape.
        spacing: Voxel spacing ``(sx, sy, sz)`` in mm.
        max_dvh: Shared histogram upper dose; defaults to ``1.05 * max(both)``.

    Returns:
        ``name -> (DVH_a, DVH_b)``.
    """
    if max_dvh is None:
        max_dvh = 1.05 * max(float(dose_a.max()), float(dose_b.max()))
    return {
        name: (
            DVH(mask, dose_a, spacing, name=name, max_dvh=max_dvh),
            DVH(mask, dose_b, spacing, name=name, max_dvh=max_dvh),
        )
        for name, mask in structures.items()
    }


def dvh_metrics(
    structures: Dict[str, np.ndarray],
    dose_a: np.ndarray,
    dose_b: np.ndarray,
    spacing: Sequence[float],
    labels: Tuple[str, str] = ("ADoTA", "MCsquare"),
    target_keyword: str = "target",
) -> dict:
    """Build the structure-type-dependent DVH metrics report (dict).

    Target structures report ``Dmin/Dmean/Dmax/D2/D95/D98``; OARs report
    ``Dmin/Dmean/Dmax/D2``. Each structure carries both doses and their difference
    (``labels[0] - labels[1]``).

    Returns:
        A JSON-serialisable dict with ``units``, ``doses``,
        ``metric_definitions`` and ``structures``.
    """
    dvhs = compute_structure_dvhs(structures, dose_a, dose_b, spacing)
    report = {
        "units": "Gy",
        "doses": list(labels),
        "metric_definitions": dict(_METRIC_DEFINITIONS),
        "structures": {},
    }
    for name in _ordered_names(structures, target_keyword):
        dvh_a, dvh_b = dvhs[name]
        stype = _structure_type(name, target_keyword)
        keys = _TARGET_METRICS if stype == "target" else _OAR_METRICS
        metrics_a = {k: round(float(getattr(dvh_a, k)), 4) for k in keys}
        metrics_b = {k: round(float(getattr(dvh_b, k)), 4) for k in keys}
        difference = {k: round(metrics_a[k] - metrics_b[k], 4) for k in keys}
        report["structures"][name] = {
            "type": stype,
            "n_voxels": dvh_a.n_voxels,
            labels[0]: metrics_a,
            labels[1]: metrics_b,
            "difference": difference,
        }
    return report


def write_dvh_metrics_json(
    json_path: Path,
    structures: Dict[str, np.ndarray],
    dose_a: np.ndarray,
    dose_b: np.ndarray,
    spacing: Sequence[float],
    labels: Tuple[str, str] = ("ADoTA", "MCsquare"),
    target_keyword: str = "target",
) -> dict:
    """Compute the DVH metrics and write them to ``json_path``; return the dict."""
    report = dvh_metrics(structures, dose_a, dose_b, spacing, labels, target_keyword)
    Path(json_path).write_text(json.dumps(report, indent=2) + "\n")
    return report


def dvh_comparison_figure(
    structures: Dict[str, np.ndarray],
    dose_a: np.ndarray,
    dose_b: np.ndarray,
    spacing: Sequence[float],
    figure_path: str,
    labels: Tuple[str, str] = ("ADoTA", "MCsquare"),
    target_keyword: str = "target",
    dpi: int = 300,
) -> list[Path]:
    """Render the ADoTA vs MCsquare DVH curves (no metric table).

    Args:
        structures: Oriented boolean masks ``name -> (z, y, x)``.
        dose_a: First dose grid (ADoTA), Gy.
        dose_b: Reference dose grid (MCsquare), Gy.
        spacing: Voxel spacing ``(sx, sy, sz)`` mm.
        figure_path: Output stem (``.svg``/``.pdf``/``.png`` written).
        labels: ``(name_a, name_b)``.
        target_keyword: Substring identifying the target (drawn first).
        dpi: Output resolution.

    Returns:
        The written figure paths.
    """
    dvhs = compute_structure_dvhs(structures, dose_a, dose_b, spacing)
    names = _ordered_names(structures, target_keyword)
    colors = {name: _PALETTE[i % len(_PALETTE)] for i, name in enumerate(names)}

    fig = plt.figure(layout="constrained", figsize=(12, 9), dpi=dpi)
    ax = fig.add_subplot(111)
    for name in names:
        dvh_a, dvh_b = dvhs[name]
        ax.plot(*dvh_a.histogram, color=colors[name], lw=2.2, ls="-")
        ax.plot(*dvh_b.histogram, color=colors[name], lw=2.2, ls="--")

    ax.set_xlabel("Dose [Gy]", fontsize=_FS_AXIS)
    ax.set_ylabel("Volume [%]", fontsize=_FS_AXIS)
    ax.set_xlim(left=0.0)
    ax.set_ylim(0.0, 100.0)
    ax.grid(True, linestyle=":", linewidth=0.6)
    ax.tick_params(labelsize=_FS_TICK)
    ax.set_title("Dose-volume histograms: ADoTA vs MCsquare", fontsize=_FS_TITLE, weight="bold")

    struct_handles = [Line2D([0], [0], color=colors[n], lw=2.8) for n in names]
    style_handles = [
        Line2D([0], [0], color="black", lw=2.2, ls="-"),
        Line2D([0], [0], color="black", lw=2.2, ls="--"),
    ]
    leg1 = ax.legend(
        struct_handles, names, title="Structure", loc="upper right",
        fontsize=_FS_LEGEND, title_fontsize=_FS_LEGEND,
    )
    ax.add_artist(leg1)
    ax.legend(
        style_handles, list(labels), title="Dose", loc="center right",
        fontsize=_FS_LEGEND, title_fontsize=_FS_LEGEND,
    )

    paths = save_figure_as_publication_formats(fig, figure_path)
    plt.close(fig)
    return paths
