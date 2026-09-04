"""CT Hounsfield-unit visualisation and simple HU-based segmentation.

Provides a lookup-table segmentation based on the Hounsfield scale
(https://en.wikipedia.org/wiki/Hounsfield_scale) and a 2×2 figure
showing original HU maps alongside the segmented result.

The Bragg-peak estimator diagnostic figure lives in
:mod:`src.figures.bp_diagnostic`.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from scipy.ndimage import gaussian_filter, median_filter

from src.figures.axes_utils import aligned_colorbar

# ── HU Lookup Table ─────────────────────────────────────────────────────────
# Each entry: (label, HU_lower_inclusive, HU_upper_exclusive, colour)
# Boundaries are chosen from the Wikipedia Hounsfield scale table and
# simplified for a coarse tissue segmentation relevant to proton therapy.

HU_LUT: list[tuple[str, float, float, str]] = [
    ("Air", -1024, -950, "#B3E5FC"),  # light blue
    ("Lung", -950, -500, "#CE93D8"),  # soft purple
    ("Fat", -500, -100, "#FFF9C4"),  # pastel yellow
    ("Soft tissue", -100, 100, "#FFCCBC"),  # pastel peach
    ("Contrast / blood", 100, 300, "#EF9A9A"),  # pastel red
    ("Cancellous bone", 300, 500, "#A5D6A7"),  # pastel green
    ("Cortical bone", 500, 3072, "#90CAF9"),  # pastel blue
]

N_CLASSES = len(HU_LUT)
_COLORS = [entry[3] for entry in HU_LUT]

# Colourmap & norm that map integer class indices 0..N-1 to colours
SEGMENT_CMAP = ListedColormap(_COLORS)
SEGMENT_NORM = BoundaryNorm(
    boundaries=list(range(N_CLASSES + 1)),  # [0, 1, 2, …, N]
    ncolors=N_CLASSES,
)


def smooth_ct(
    ct_hu: np.ndarray,
    method: str = "gaussian",
    sigma: float = 1.0,
    median_size: int = 3,
) -> np.ndarray:
    """Apply spatial smoothing to a CT volume in Hounsfield Units.

    Smoothing reduces voxel-level noise before HU-based segmentation,
    producing cleaner tissue boundaries.

    Args:
        ct_hu: 3-D CT volume in HU, shape ``(D, H, W)``.
        method: Smoothing method – ``"gaussian"`` for a Gaussian filter
            or ``"median"`` for a median filter.
        sigma: Standard deviation for the Gaussian kernel (only used
            when *method* is ``"gaussian"``).
        median_size: Side length of the cubic structuring element (only
            used when *method* is ``"median"``).

    Returns:
        Smoothed CT volume with the same shape and dtype semantics as
        the input.

    Raises:
        ValueError: If *method* is not ``"gaussian"`` or ``"median"``.
    """
    if method == "gaussian":
        return gaussian_filter(ct_hu, sigma=sigma)
    elif method == "median":
        return median_filter(ct_hu, size=median_size)
    else:
        raise ValueError(
            f"Unknown smoothing method '{method}'. " f"Choose 'gaussian' or 'median'."
        )


def segment_hu(ct_hu: np.ndarray) -> np.ndarray:
    """Segment a CT volume by HU lookup table.

    Each voxel is assigned an integer class index (0 … N-1) according
    to :data:`HU_LUT`.

    Args:
        ct_hu: Array of Hounsfield-unit values (any shape).

    Returns:
        Integer array of the same shape with class indices.
    """
    seg = np.zeros_like(ct_hu, dtype=np.int8)
    for idx, (_, lo, hi, _) in enumerate(HU_LUT):
        seg[(ct_hu >= lo) & (ct_hu < hi)] = idx
    return seg


def plot_ct_with_segmentation(
    ct_hu: np.ndarray,
    sample_id: str,
    output_path: Path,
    ct_hu_unsmoothed: np.ndarray | None = None,
    gt_dose: np.ndarray | None = None,
    bp_range_slices: tuple[float, float] | None = None,
    voxel_spacing_mm: float = 2.0,
    sobel_magnitude: np.ndarray | None = None,
    sobel_magnitude_raw: np.ndarray | None = None,
) -> None:
    """Save a figure with CT slices, segmentation, and optionally the IDD.

    When *ct_hu_unsmoothed* is provided the image grid has **3 rows**
    (raw HU → smoothed HU → segmentation); otherwise **2 rows**.

    When *sobel_magnitude_raw* is provided an additional row showing
    the Sobel gradient magnitude on the **original** (unsmoothed) CT
    is inserted after the segmentation row.

    When *sobel_magnitude* is provided an additional row showing the
    Sobel gradient magnitude on the **smoothed** CT is inserted after
    the raw-Sobel row (or after segmentation if raw Sobel is absent).

    When *gt_dose* is provided an additional **IDD row** (full-width)
    is appended at the bottom showing the normalised Integrated Depth
    Dose together with the estimated Bragg-peak range.

    Args:
        ct_hu: 3-D CT volume in HU ``(D, H, W)``.  When smoothing is
            used this should be the **smoothed** volume.
        sample_id: Identifier used in the figure title / filename.
        output_path: Where to save the PNG figure.
        ct_hu_unsmoothed: Optional raw (unsmoothed) CT volume.
        gt_dose: Optional ground-truth dose grid ``(D, H, W)``.
        bp_range_slices: ``(z_min, z_max)`` in slice indices.
        voxel_spacing_mm: Voxel spacing along the depth axis [mm].
        sobel_magnitude: Optional pre-computed 3-D Sobel gradient
            magnitude volume ``(D, H, W)`` on the smoothed CT.
        sobel_magnitude_raw: Optional pre-computed 3-D Sobel gradient
            magnitude volume ``(D, H, W)`` on the original CT.
    """
    smoothed = ct_hu_unsmoothed is not None
    has_idd = gt_dose is not None
    has_sobel = sobel_magnitude is not None
    has_sobel_raw = sobel_magnitude_raw is not None
    img_rows = 3 if smoothed else 2
    if has_sobel_raw:
        img_rows += 1
    if has_sobel:
        img_rows += 1

    D, H, W = ct_hu.shape
    axial_idx = H // 2
    sagittal_idx = W // 2

    # ── Helper: extract & rotate a pair of slices ───────────────────────
    def _slices(vol: np.ndarray):
        ax_sl = np.rot90(vol[:, axial_idx, :])  # (D, W) → (W, D)
        sg_sl = np.rot90(vol[:, :, sagittal_idx])  # (D, H) → (H, D)
        return ax_sl, sg_sl

    # Smoothed (or only) HU slices
    axial_slice, sagittal_slice = _slices(ct_hu)

    # Segmentation (always on ct_hu, which is smoothed when applicable)
    seg = segment_hu(ct_hu)
    seg_axial, seg_sagittal = _slices(seg)

    # Unsmoothed slices (if available)
    if smoothed:
        raw_axial, raw_sagittal = _slices(ct_hu_unsmoothed)

    # Sobel slices (if available)
    if has_sobel_raw:
        sobel_raw_axial, sobel_raw_sagittal = _slices(sobel_magnitude_raw)
    if has_sobel:
        sobel_axial, sobel_sagittal = _slices(sobel_magnitude)

    # ── Build mosaic layout ─────────────────────────────────────────────
    # Image rows use pairs (left=axial, right=sagittal).
    # IDD row spans full width.
    row_labels = []
    if smoothed:
        row_labels.append("AB")  # raw HU
    row_labels.append("CD")  # (smoothed) HU
    row_labels.append("EF")  # segmentation
    if has_sobel_raw:
        row_labels.append("JK")  # Sobel on raw CT
    if has_sobel:
        row_labels.append("GH")  # Sobel on smoothed CT
    if has_idd:
        row_labels.append("II")  # IDD (full-width)

    mosaic = "\n".join(row_labels)

    height_ratios = [1.0] * img_rows
    if has_idd:
        height_ratios.append(1.2)  # IDD row slightly taller

    fig_height = 2.5 * img_rows + (3.0 if has_idd else 0.0)

    fig, ax_dict = plt.subplot_mosaic(
        mosaic,
        figsize=(16, fig_height),
        dpi=200,
        gridspec_kw={
            "hspace": 0.35,
            "wspace": 0.25,
            "height_ratios": height_ratios,
        },
    )

    # ── Row: unsmoothed HU (only when smoothing is applied) ─────────────
    if smoothed:
        ax_dict["A"].imshow(raw_axial, cmap="gray", aspect="auto")
        ax_dict["A"].set_title("HU (raw) – centre axial slice", fontsize=11)

        im_raw = ax_dict["B"].imshow(raw_sagittal, cmap="gray", aspect="auto")
        ax_dict["B"].set_title("HU (raw) – centre sagittal slice", fontsize=11)
        aligned_colorbar(fig, im_raw, ax_dict["B"], "HU", label_coords=(6.0, 0.5))

    # ── Row: (smoothed) HU ──────────────────────────────────────────────
    hu_label = "HU (smoothed)" if smoothed else "HU"
    ax_dict["C"].imshow(axial_slice, cmap="gray", aspect="auto")
    ax_dict["C"].set_title(f"{hu_label} – centre axial slice", fontsize=11)

    im_hu = ax_dict["D"].imshow(sagittal_slice, cmap="gray", aspect="auto")
    ax_dict["D"].set_title(f"{hu_label} – centre sagittal slice", fontsize=11)
    aligned_colorbar(fig, im_hu, ax_dict["D"], "HU", label_coords=(6.0, 0.5))

    # ── Row: segmentation (pastel palette) ──────────────────────────────
    ax_dict["E"].imshow(
        seg_axial,
        cmap=SEGMENT_CMAP,
        norm=SEGMENT_NORM,
        aspect="auto",
    )
    ax_dict["E"].set_title("Segmentation – centre axial slice", fontsize=11)

    seg_im = ax_dict["F"].imshow(
        seg_sagittal,
        cmap=SEGMENT_CMAP,
        norm=SEGMENT_NORM,
        aspect="auto",
    )
    ax_dict["F"].set_title("Segmentation – centre sagittal slice", fontsize=11)

    # Discrete colorbar with tissue-class labels
    seg_cbar = aligned_colorbar(
        fig, seg_im, ax_dict["F"], "Tissue", label_coords=(8.5, 0.5)
    )
    tick_locs = [i + 0.5 for i in range(N_CLASSES)]
    seg_cbar.set_ticks(tick_locs)
    seg_cbar.set_ticklabels(
        [entry[0] for entry in HU_LUT],
        fontsize=8,
    )

    # ── Row: Sobel on raw CT (optional) ──────────────────────────────────
    if has_sobel_raw:
        # Share colour scale between raw and smoothed Sobel rows
        sobel_vmax = max(
            sobel_raw_axial.max(),
            sobel_raw_sagittal.max(),
            sobel_axial.max() if has_sobel else 0.0,
            sobel_sagittal.max() if has_sobel else 0.0,
        )
        ax_dict["J"].imshow(
            sobel_raw_axial,
            cmap="hot",
            aspect="auto",
            vmin=0,
            vmax=sobel_vmax,
        )
        ax_dict["J"].set_title("Sobel (raw CT) – centre axial slice", fontsize=11)

        im_sobel_raw = ax_dict["K"].imshow(
            sobel_raw_sagittal,
            cmap="hot",
            aspect="auto",
            vmin=0,
            vmax=sobel_vmax,
        )
        ax_dict["K"].set_title("Sobel (raw CT) – centre sagittal slice", fontsize=11)
        aligned_colorbar(
            fig,
            im_sobel_raw,
            ax_dict["K"],
            "Gradient",
            label_coords=(6.0, 0.5),
        )
    else:
        sobel_vmax = None

    # ── Row: Sobel on smoothed CT (optional) ────────────────────────────
    if has_sobel:
        sobel_kw = {}
        if sobel_vmax is not None:
            sobel_kw = dict(vmin=0, vmax=sobel_vmax)
        ax_dict["G"].imshow(
            sobel_axial,
            cmap="hot",
            aspect="auto",
            **sobel_kw,
        )
        ax_dict["G"].set_title("Sobel (smoothed CT) – centre axial slice", fontsize=11)

        im_sobel = ax_dict["H"].imshow(
            sobel_sagittal,
            cmap="hot",
            aspect="auto",
            **sobel_kw,
        )
        ax_dict["H"].set_title(
            "Sobel (smoothed CT) – centre sagittal slice", fontsize=11
        )
        aligned_colorbar(
            fig,
            im_sobel,
            ax_dict["H"],
            "Gradient",
            label_coords=(6.0, 0.5),
        )

    # Remove ticks from all image axes
    image_keys = set(ax_dict.keys()) - {"I"}
    for key in image_keys:
        ax_dict[key].set_xticks([])
        ax_dict[key].set_yticks([])

    # ── Row: Integrated Depth Dose ───────────────────────────────────────
    if has_idd:
        idd_gt = gt_dose.sum(axis=(1, 2))  # (D,)
        idd_max = np.max(idd_gt)
        if idd_max > 0:
            idd_norm = idd_gt / idd_max * 100.0
        else:
            idd_norm = idd_gt

        bp_idx = int(np.argmax(idd_gt))

        ax = ax_dict["I"]
        ax.plot(idd_norm, label="GT IDD", color="blue")
        ax.axvline(
            x=bp_idx,
            color="red",
            linestyle="--",
            label=f"Bragg peak ({bp_idx * voxel_spacing_mm:.0f} mm)",
        )

        if bp_range_slices is not None:
            z_min_sl, z_max_sl = bp_range_slices
            ax.axvline(
                x=z_min_sl,
                color="green",
                linestyle=":",
                linewidth=1.5,
                label=f"BP start ({z_min_sl * voxel_spacing_mm:.0f} mm)",
            )
            ax.axvline(
                x=z_max_sl,
                color="orange",
                linestyle=":",
                linewidth=1.5,
                label=f"BP end ({z_max_sl * voxel_spacing_mm:.0f} mm)",
            )
            ax.axvspan(z_min_sl, z_max_sl, alpha=0.10, color="green")

        ax.set_xticks(np.arange(0, D + 1, 10))
        ax.set_xticklabels(
            (np.arange(0, D + 1, 10) * voxel_spacing_mm).astype(int),
            rotation=45,
        )
        ax.set_xlabel("Depth [mm]", fontsize=12)
        ax.set_ylabel("Normalised IDD [%]", fontsize=12, color="blue")
        ax.set_xlim(0, D - 1)
        ax.tick_params(axis="y", labelcolor="blue", labelsize=10)
        ax.tick_params(axis="x", labelsize=10)
        ax.grid(linestyle="--", linewidth=0.5)

        # ── Twin y-axis: IDD gradient & second derivative ──────────────
        idd_grad = np.gradient(idd_norm)
        idd_grad2 = np.gradient(idd_grad)  # second derivative

        ax2 = ax.twinx()
        ax2.plot(
            idd_grad,
            label="dIDD/dz",
            color="darkviolet",
            alpha=0.7,
            linewidth=1.0,
        )
        ax2.plot(
            idd_grad2,
            label=r"d$^2$IDD/dz$^2$",
            color="darkorange",
            alpha=0.7,
            linewidth=1.0,
            linestyle="-.",
        )
        ax2.set_ylabel(
            r"dIDD/dz,  d$^2$IDD/dz$^2$  [%/slice$^n$]",
            fontsize=11,
        )
        ax2.tick_params(axis="y", labelsize=10)

        # Combine legends from both axes
        lines_1, labels_1 = ax.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax.legend(
            lines_1 + lines_2,
            labels_1 + labels_2,
            fontsize=9,
            loc="upper left",
        )

    fig.suptitle(
        f"CT HU & Segmentation – {sample_id}",
        fontsize=13,
        fontweight="bold",
    )
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
