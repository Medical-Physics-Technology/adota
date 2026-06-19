"""Stage 3: accumulate per-spot beamlets back into the full scoring grid.

The inverse of extraction, and consistent with it: extraction rotates the CT
around the **physical isocenter** (SimpleITK), so accumulation inverts *that*
rotation -- not the old flipped-pivot torch rotation. Each beamlet array (the
flux projection as a dose stand-in now; the ADoTA prediction later) is deposited
into a zeroed copy of the CT grid, scaled by ``relative_weight``:

1. per field, deposit every spot's crop into a zeroed **rotated-frame** grid at
   its ``crp`` window (the same in-bounds overlap as the crop), summing overlaps;
2. **de-rotate** that grid back to the patient frame
   (``rotate_ct_around_isocenter(grid, -A, iso)``, out-of-bounds filled with
   ``0.0`` -- this is dose, not CT);
3. sum across fields, clip negatives, and write ``Dose_ADoTA.mhd`` next to the
   MC ``Dose.mhd`` (which is never touched), carrying the CT grid's metadata.

Because the deposit reuses :func:`src.beamlets.cropping.clip_axis_window` and the
de-rotation is the exact inverse of the extraction rotation, a beamlet lands back
in exactly the voxels it was cropped from (an extract -> accumulate round trip is
exact at rotation angle 0; rotated fields incur the usual double-interpolation
smoothing).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import SimpleITK as sitk

from src.beamlets import ROI_SIZE
from src.beamlets.cropping import clip_axis_window
from src.beamlets.isocenter import isocenter_physical
from src.beamlets.rotation import rotate_ct_around_isocenter
from src.loaders.plan_directory import PlanDirectory

logger = logging.getLogger(__name__)

__all__ = ["AccumulationConfig", "deposit_crop", "accumulate_dose", "run_accumulation"]

DOSE_SUFFIX = {"flux": "_flux.npy", "prediction": "_ds_pred.npy"}


@dataclass
class AccumulationConfig:
    """Configuration for :func:`run_accumulation`.

    Attributes:
        dose_source: Which per-spot array to deposit -- ``"flux"`` (the
            constructed flux projection, used as a dose stand-in for now) or
            ``"prediction"`` (the ADoTA output ``{id}_ds_pred.npy``, added with
            the inference stage).
        clip_negative: Clip negative values to zero after summing.
        output_name: Output filename written at the plan-dir level.
        calibration_factor: Multiplicative dose calibration applied to the final
            accumulated dose before writing. Defaults to ``1.0`` (no-op, the
            pipeline is unaffected). A value > 1 corrects the model's systematic
            per-beamlet under-prediction (~2.8% measured on held-out beamlets), so
            the written ``Dose_ADoTA.mhd`` and every downstream consumer (figure,
            DVH, gamma) use the calibrated dose consistently.
    """

    dose_source: str = "flux"
    clip_negative: bool = True
    output_name: str = "Dose_ADoTA.mhd"
    calibration_factor: float = 1.0


def deposit_crop(
    grid: np.ndarray,
    crop: np.ndarray,
    crp: Sequence[int],
    weight: float,
    roi_size: Tuple[int, int, int] = ROI_SIZE,
) -> None:
    """Add ``weight * crop`` into ``grid`` at the crop's original window.

    The window is the inverse of the extraction crop: lateral ``iz +/- H//2``,
    ``iy +/- W//2`` and depth ``x in [0, D)``, clipped to the grid (so edge
    beamlets deposit only their in-bounds part). ``grid`` is modified in place.

    Args:
        grid: Full-grid array ``(z, y, x)`` to deposit into.
        crop: Beamlet array ``(H, W, D)`` (z, y, x).
        crp: Centre voxel ``(iz, iy, ix)`` (the saved ``crp_numpy_ct``).
        weight: Scalar weight (``relative_weight``).
        roi_size: ``(H, W, D)`` ROI size.
    """
    height, width, depth = roi_size
    iz, iy, _ix = crp
    nz, ny, nx = grid.shape

    z_lo, z_hi, z_clo, z_chi = clip_axis_window(iz - height // 2, height, nz)
    y_lo, y_hi, y_clo, y_chi = clip_axis_window(iy - width // 2, width, ny)
    x_lo, x_hi, x_clo, x_chi = clip_axis_window(0, depth, nx)

    grid[z_lo:z_hi, y_lo:y_hi, x_lo:x_hi] += weight * crop[
        z_clo:z_chi, y_clo:y_chi, x_clo:x_chi
    ]


def accumulate_dose(
    plan_directory: PlanDirectory,
    beamlets_dir: Path,
    config: AccumulationConfig,
) -> Tuple[sitk.Image, dict]:
    """Accumulate all extracted beamlets into a full-grid dose image.

    Args:
        plan_directory: The loaded plan directory (provides the CT grid).
        beamlets_dir: Directory of extracted ``{id}_*.npy`` / ``{id}_sim_res.json``.
        config: Accumulation options.

    Returns:
        Tuple ``(dose_image, summary)`` -- the accumulated dose as a SimpleITK
        image on the CT grid, and a summary dict.

    Raises:
        FileNotFoundError: If no beamlet metadata is found.
        NotImplementedError: If ``dose_source="prediction"`` (added later).
    """
    if config.dose_source not in DOSE_SUFFIX:
        raise ValueError(
            f"Unknown dose_source {config.dose_source!r}; "
            f"expected one of {sorted(DOSE_SUFFIX)}"
        )

    ct = plan_directory.ct
    ref_arr = sitk.GetArrayFromImage(ct)  # (z, y, x)
    grouped = _load_records_by_field(beamlets_dir)

    started = perf_counter()
    deposit_s = 0.0
    derotate_s = 0.0
    total = np.zeros(ref_arr.shape, dtype=np.float32)
    n_spots = 0

    for beam, records in grouped.items():
        rec0 = records[0]
        angle = rec0["gantry_angle"]
        iso_index = rec0["simulation_log"]["isocenter"]
        # Plan isocenter x is flipped relative to the CT (S3) -- same correction
        # as extraction so the de-rotation pivots on the true target.
        iso_phys = isocenter_physical(iso_index, ct)

        # Rebuild the field's EXPANDED rotated grid (the frame the crops live in;
        # geometry stored per field in sim_res), so crp indices land correctly.
        ex_nx, ex_ny, ex_nz = rec0["image_size"]  # sitk (x, y, z) order
        rotated_grid = np.zeros((ex_nz, ex_ny, ex_nx), dtype=np.float32)  # (z, y, x)
        deposit_t = perf_counter()
        for record in records:
            arr = _load_beamlet_array(
                beamlets_dir, record["id"], config.dose_source
            )
            deposit_crop(
                rotated_grid,
                arr,
                record["crp_numpy_ct"],
                float(record["relative_weight"]),
                tuple(record["roi_size"]),
            )
            n_spots += 1
        deposit_s += perf_counter() - deposit_t

        # De-rotate the field's deposited (expanded) grid back into the patient
        # frame AND crop to the original CT grid in one step: resample with the
        # CT as the reference. This is the exact inverse of the extraction
        # rotation around the isocenter, with the output at the original size.
        derotate_t = perf_counter()
        rotated_image = sitk.GetImageFromArray(rotated_grid)
        rotated_image.SetOrigin(tuple(rec0["image_origin"]))
        rotated_image.SetSpacing(tuple(rec0["image_spacing"]))
        rotated_image.SetDirection(ct.GetDirection())
        derotated = rotate_ct_around_isocenter(
            rotated_image, -angle, iso_phys, reference=ct, default_value=0.0
        )
        total += sitk.GetArrayFromImage(derotated)
        derotate_s += perf_counter() - derotate_t
        logger.info(
            "Field beam=%d: deposited %d spots, de-rotated by %.1f deg",
            beam,
            len(records),
            -angle,
        )

    if config.clip_negative:
        total = np.clip(total, 0.0, None)

    # Optional dose calibration (default 1.0 = no-op, bit-identical to before).
    # A pure multiplicative scalar, so it commutes with the Gy conversion applied
    # downstream; guarded so the factor-1.0 path leaves ``total`` untouched.
    if config.calibration_factor != 1.0:
        total *= np.float32(config.calibration_factor)

    dose_image = sitk.GetImageFromArray(total)
    dose_image.CopyInformation(ct)

    summary = {
        "n_spots": n_spots,
        "n_fields": len(grouped),
        "dose_source": config.dose_source,
        "calibration_factor": float(config.calibration_factor),
        "dose_max": float(total.max()),
        "dose_sum": float(total.sum()),
        "grid_size": list(ct.GetSize()),
        "deposit_s": deposit_s,
        "derotate_s": derotate_s,
        "elapsed_s": perf_counter() - started,
    }
    return dose_image, summary


def run_accumulation(
    plan_directory: PlanDirectory,
    beamlets_dir: Path,
    output_path: Path,
    config: Optional[AccumulationConfig] = None,
) -> dict:
    """Accumulate beamlets and write the dose image to ``output_path``.

    Args:
        plan_directory: The loaded plan directory.
        beamlets_dir: Directory of extracted beamlets.
        output_path: Destination ``.mhd`` (e.g. ``<plan_dir>/Dose_ADoTA.mhd``).
        config: Accumulation options.

    Returns:
        A summary dict (also includes the written ``output_path``).

    Raises:
        ValueError: If ``output_path`` would overwrite the MC reference dose.
    """
    config = config or AccumulationConfig()
    output_path = Path(output_path)
    if output_path.name.lower() in {"dose.mhd", "dose.raw"}:
        raise ValueError(
            f"Refusing to write to {output_path.name}: the MC reference dose is "
            "read-only. Choose a different output name."
        )

    dose_image, summary = accumulate_dose(plan_directory, beamlets_dir, config)
    write_t = perf_counter()
    sitk.WriteImage(dose_image, str(output_path))
    summary["write_s"] = perf_counter() - write_t
    summary["output_path"] = str(output_path)
    logger.info(
        "Accumulated %d spots across %d field(s) -> %s (max=%.4g)",
        summary["n_spots"],
        summary["n_fields"],
        output_path,
        summary["dose_max"],
    )
    return summary


def _load_beamlet_array(beamlets_dir: Path, spot_id: str, dose_source: str) -> np.ndarray:
    """Load a spot's array as ``(z, y, x)`` crop order, ready for deposit.

    ``flux`` is already saved in ``(z, y, x)`` crop order. The ADoTA
    ``prediction`` is the model output ``{id}_ds_pred.npy``, depth-first
    ``(1, 1, D, H, W)`` (matching the network's permuted input), so it is squeezed
    and ``moveaxis(0, -1)`` brings the depth axis back to ``x``. The shape follows
    the extraction's ``grid_factor``: ``(1,1,320,60,60) -> (60,60,320)`` at gf=1,
    ``(1,1,160,30,30) -> (30,30,160)`` at gf=2 (deposited on the 2mm field grid).
    """
    arr = np.load(
        Path(beamlets_dir) / f"{spot_id}{DOSE_SUFFIX[dose_source]}"
    ).astype(np.float32)
    if dose_source == "prediction":
        arr = np.moveaxis(np.squeeze(arr), 0, -1)
    return arr


def _load_records_by_field(beamlets_dir: Path) -> Dict[int, List[dict]]:
    """Load every ``*_sim_res.json`` and group by beam (field) index, in order."""
    beamlets_dir = Path(beamlets_dir)
    meta_files = sorted(beamlets_dir.glob("*_sim_res.json"))
    if not meta_files:
        raise FileNotFoundError(f"No beamlet metadata found in {beamlets_dir}")

    grouped: Dict[int, List[dict]] = {}
    for meta_file in meta_files:
        record = json.loads(meta_file.read_text())
        grouped.setdefault(record["beam"], []).append(record)
    return dict(sorted(grouped.items()))
