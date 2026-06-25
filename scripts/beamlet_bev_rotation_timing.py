"""Per-beamlet BEV reinterpolation: DoTA-like vs ADoTA (standalone, YAML-driven).

The paper's claim is that ADoTA removes the per-beamlet *reinterpolation to the
BEV* that a DoTA-like pipeline needs: instead of rotating each cropped subvolume
so the beamlet is perpendicular to the entrance face (and rotating the predicted
dose back afterwards), ADoTA keeps the crop axis-aligned and encodes the beam
direction analytically with the proton-flux projection.

This script runs a single OpenTPS plan through the **staged CPU** extraction
(storing every beamlet on the 2mm per-field grid, so the per-beamlet down/up-
sample is bypassed for both pipelines) and then, for a sample of beamlets,
produces the validation figure that proves the DoTA reinterpolation is correct:

    row 1: CT + flux            (col1 axial, col2 sagittal, col3 coronal)
    row 2: rotated CT + centerline (the lateral-centre depth line the DoTA model
           assumes the protons enter on; the rotated flux ridge is overlaid to
           show it now lies on that centerline)

This is the STEP 1+2 deliverable. The flux/rotation timing comparison and the
ADoTA inference + dose back-rotation are layered on next, reusing the same
extraction outputs and :func:`src.image_processing.rotation.rotate_beamlet_crop`.
"""

from __future__ import annotations

import json
import logging
import random
import sys
from pathlib import Path
from time import perf_counter
from typing import Annotated, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import typer

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.adota.config import load_yaml_config, setup_logging, setup_run_directory
from src.adota.utils import load_model
from src.beamlets import roi_for_factor
from src.beamlets.bdl import BeamDataLibrary
from src.beamlets.extraction import ExtractionConfig, run_extraction
from src.beamlets.flux import flux_projection, flux_spatial_spread
from src.beamlets.streaming import StreamingConfig, run_streaming_pipeline
from src.evaluation.cli import resolve_device
from src.image_processing.rotation import rotate_beamlet_crop
from src.loaders.dir_based import (
    DEFAULT_SCALE,
    postprocess_prediction,
    prepare_input_from_arrays,
)
from src.loaders.plan_directory import PlanDirectory, load_plan_directory

logger = logging.getLogger(__name__)

app = typer.Typer(
    help="Per-beamlet BEV reinterpolation timing/validation (DoTA-like vs ADoTA)."
)


def _coerce_beams(beams: Optional[object]) -> Optional[list[int]]:
    """Accept ``null``, an int, a list, or a comma-separated string of beams."""
    if beams is None or beams == "":
        return None
    if isinstance(beams, int):
        return [beams]
    if isinstance(beams, (list, tuple)):
        return [int(b) for b in beams]
    return [int(part) for part in str(beams).split(",") if part.strip()]


def run_staged_cpu_extraction(
    plan_directory: PlanDirectory,
    output_dir: Path,
    *,
    bdl_path: Optional[Path],
    grid_factor: int,
    n_spots: Optional[int],
    beams: Optional[list[int]],
    overwrite: bool,
) -> dict:
    """Stage 1: extract + store every beamlet (CPU flux) on the 2mm field grid.

    Reuses the trusted serial extraction (:func:`run_extraction`) with
    ``flux_on_gpu=False`` so the flux is built on the CPU, as the study requires.
    """
    config = ExtractionConfig(
        roi_size=roi_for_factor(grid_factor),
        n_spots=n_spots,
        beams=beams,
        overwrite=overwrite,
        save_overlays=False,
        bdl_path=bdl_path,
        flux_on_gpu=False,
        grid_factor=grid_factor,
    )
    logger.info("Staged CPU extraction (grid_factor=%d) -> %s", grid_factor, output_dir)
    return run_extraction(plan_directory, output_dir, config)


def _load_beamlet(beamlets_dir: Path, spot_id: str) -> tuple[np.ndarray, np.ndarray, dict]:
    ct = np.load(beamlets_dir / f"{spot_id}_ct.npy")
    flux = np.load(beamlets_dir / f"{spot_id}_flux.npy")
    sim_res = json.loads((beamlets_dir / f"{spot_id}_sim_res.json").read_text())
    return ct, flux, sim_res


def _flux_overlay(ax: plt.Axes, flux_slice: np.ndarray, *, dashed: bool = False) -> None:
    """Overlay the flux ridge (translucent fill + contour) on a CT slice axes."""
    fmax = float(flux_slice.max())
    if fmax <= 1e-12:
        return
    norm = flux_slice / fmax
    masked = np.ma.masked_less(norm, 0.05)
    ax.imshow(masked, origin="lower", aspect="auto", cmap="inferno", alpha=0.4, vmin=0.0, vmax=1.0)
    ax.contour(
        norm,
        levels=[0.5],
        colors="yellow",
        linewidths=1.2,
        linestyles="dashed" if dashed else "solid",
    )


def _make_validation_figure(
    ct_zyx: np.ndarray,
    flux_zyx: np.ndarray,
    rotated_ct_zyx: np.ndarray,
    rotated_flux_zyx: np.ndarray,
    *,
    spot_id: str,
    beamlet_angles: tuple[float, float],
    energy_mev: float,
    out_path: Path,
) -> None:
    """2x3 figure: (CT+flux) over (rotated CT + centerline), in 3 orthogonal views.

    The crop is ``(z, y, x)`` with the beam/depth axis along ``x``:
        axial    = mid-z plane (y vertical, x=depth horizontal),
        sagittal = mid-x plane (z vertical, y horizontal) -- the beam cross-section,
        coronal  = mid-y plane (z vertical, x=depth horizontal).
    """
    nz, ny, nx = ct_zyx.shape
    zc, yc, xc = nz // 2, ny // 2, nx // 2
    ct_kw = dict(origin="lower", cmap="gray", vmin=-1000.0, vmax=1000.0)

    # (slice extractor, aspect, x/y axis labels, column title)
    views = [
        (lambda v: v[zc, :, :], "auto", ("depth x", "lateral y"), "axial (mid-z)"),
        (lambda v: v[:, :, xc], "equal", ("lateral y", "lateral z"), "sagittal (mid-x)"),
        (lambda v: v[:, yc, :], "auto", ("depth x", "lateral z"), "coronal (mid-y)"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 7), dpi=130)
    fig.suptitle(
        f"Beamlet {spot_id}: DoTA reinterpolation to the BEV\n"
        f"beamlet_angles=(theta_x={beamlet_angles[0]:+.2f} deg, "
        f"theta_y={beamlet_angles[1]:+.2f} deg), E={energy_mev:.1f} MeV",
        fontsize=13,
        weight="bold",
    )

    for col, (slicer, aspect, (xlabel, ylabel), title) in enumerate(views):
        # Row 1: original CT + angled flux.
        ax = axes[0, col]
        ax.imshow(slicer(ct_zyx), aspect=aspect, **ct_kw)
        _flux_overlay(ax, slicer(flux_zyx))
        ax.set_title(f"CT + flux | {title}", fontsize=10)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)

        # Row 2: rotated CT + centerline (+ rotated flux ridge for proof).
        ax = axes[1, col]
        ax.imshow(slicer(rotated_ct_zyx), aspect=aspect, **ct_kw)
        _flux_overlay(ax, slicer(rotated_flux_zyx), dashed=True)
        if col == 0:  # axial: centerline at y = yc across the depth
            ax.axhline(yc, color="cyan", lw=1.4, ls="-", label="centerline")
        elif col == 1:  # sagittal: the beam entrance centre point
            ax.plot([yc], [zc], marker="+", color="cyan", ms=14, mew=2.0, label="centerline")
        else:  # coronal: centerline at z = zc across the depth
            ax.axhline(zc, color="cyan", lw=1.4, ls="-", label="centerline")
        ax.legend(loc="upper right", fontsize=8)
        ax.set_title(f"rotated CT + centerline | {title}", fontsize=10)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _select_spot_ids(
    beamlets_dir: Path, n_figures: int, selection: str, figure_seed: int
) -> list[str]:
    """Pick which beamlets to visualise.

    ``"max_angle"`` (default) takes the most-tilted beamlets, where the BEV
    reinterpolation is most visible; ``"random"`` samples uniformly.
    """
    spot_ids = sorted(p.name.removesuffix("_sim_res.json") for p in beamlets_dir.glob("*_sim_res.json"))
    if not spot_ids:
        raise FileNotFoundError(f"No beamlets found under {beamlets_dir}")
    if n_figures >= len(spot_ids):
        return spot_ids
    if selection == "random":
        return sorted(random.Random(figure_seed).sample(spot_ids, n_figures))
    if selection == "max_angle":
        def magnitude(spot_id: str) -> float:
            meta = json.loads((beamlets_dir / f"{spot_id}_sim_res.json").read_text())
            ba = meta["simulation_log"]["beamlet_angles"]
            return float(np.hypot(float(ba[0]), float(ba[1])))

        return sorted(sorted(spot_ids, key=magnitude, reverse=True)[:n_figures])
    raise ValueError(f"Unknown selection {selection!r}; expected 'max_angle' or 'random'.")


def generate_validation_figures(
    beamlets_dir: Path,
    figures_dir: Path,
    *,
    n_figures: int,
    figure_seed: int,
    rotation_backend: str,
    device: str,
    selection: str = "max_angle",
) -> list[dict]:
    """Render the rotation-validation figure for a sample of stored beamlets."""
    selected = _select_spot_ids(beamlets_dir, n_figures, selection, figure_seed)
    n_total = len(list(beamlets_dir.glob("*_sim_res.json")))
    logger.info(
        "Rendering %d validation figure(s) of %d beamlets (selection=%s)",
        len(selected),
        n_total,
        selection,
    )

    summary: list[dict] = []
    for spot_id in selected:
        ct, flux, sim_res = _load_beamlet(beamlets_dir, spot_id)
        beamlet_angles = tuple(float(a) for a in sim_res["simulation_log"]["beamlet_angles"])
        energy_mev = float(sim_res["initial_energy"])

        rotated_ct, ct_rot_s = rotate_beamlet_crop(
            ct, beamlet_angles, backend=rotation_backend, device=device
        )
        rotated_flux, _ = rotate_beamlet_crop(
            flux, beamlet_angles, backend=rotation_backend, device=device
        )

        out_path = figures_dir / f"{spot_id}_bev_rotation.png"
        _make_validation_figure(
            ct,
            flux,
            rotated_ct,
            rotated_flux,
            spot_id=spot_id,
            beamlet_angles=beamlet_angles,
            energy_mev=energy_mev,
            out_path=out_path,
        )
        summary.append(
            {
                "spot_id": spot_id,
                "beamlet_angles": list(beamlet_angles),
                "energy_mev": energy_mev,
                "ct_shape_zyx": list(ct.shape),
                "ct_forward_rotation_s": ct_rot_s,
                "figure": str(out_path),
            }
        )
        logger.info(
            "  %s: angles=(%.2f, %.2f) deg, ct_rotation=%.4fs -> %s",
            spot_id,
            beamlet_angles[0],
            beamlet_angles[1],
            ct_rot_s,
            out_path.name,
        )
    return summary


def _time_flux_construction(
    sim_res: dict,
    ct_shape_zyx: tuple[int, int, int],
    bdl: BeamDataLibrary,
    grid_factor: int,
    repeats: int,
) -> tuple[np.ndarray, float]:
    """Rebuild + time the analytical CPU flux projection (the ADoTA cost).

    Reconstructs the exact :func:`flux_projection` call the extraction used (same
    entrance projection, beamlet angles, nearest-energy sigmas, shape and 2mm
    spacing) so the measured time is the genuine per-beamlet flux-construction cost.
    """
    energy = float(sim_res["initial_energy"])
    re_proj = sim_res["rays_entrence_point_proj"]
    angles = tuple(float(a) for a in sim_res["simulation_log"]["beamlet_angles"])
    sigmas = flux_spatial_spread(bdl, energy)
    spacing = np.asarray([grid_factor, grid_factor, grid_factor], dtype=np.float32)

    flux = flux_projection(re_proj, angles, sigmas, ct_shape_zyx, spacing=spacing)  # warmup + result
    times: list[float] = []
    for _ in range(repeats):
        start = perf_counter()
        flux_projection(re_proj, angles, sigmas, ct_shape_zyx, spacing=spacing)
        times.append(perf_counter() - start)
    return flux, float(np.median(times))


def _infer_bev_dose(
    model: torch.nn.Module,
    device: torch.device,
    rotated_ct: np.ndarray,
    rotated_flux: np.ndarray,
    energy_mev: float,
    *,
    scale: dict,
    normalize_flux: bool,
    downsampling_method: str,
    grid_factor: int,
) -> tuple[np.ndarray, float, float, float]:
    """Run ADoTA on the BEV (perpendicular) inputs; return the dose crop + step times.

    The prep / forward / postprocess steps are shared with the ADoTA pipeline (the
    only difference is the rotated vs axis-aligned input), so their times are
    reported as "shared" and cancel out of the DoTA-vs-ADoTA comparison.
    """
    resize = grid_factor == 1
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    prep_t = perf_counter()
    x, e = prepare_input_from_arrays(
        rotated_ct, rotated_flux, energy_mev, scale=scale, normalize_flux=normalize_flux,
        downsampling_method=downsampling_method, device=device, resize=resize,
    )
    x_batch = x.unsqueeze(0).to(device)
    e_batch = e.unsqueeze(0).to(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    prep_s = perf_counter() - prep_t

    fwd_t = perf_counter()
    with torch.no_grad():
        pred = model(x_batch, e_batch)[0]
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    forward_s = perf_counter() - fwd_t

    post_t = perf_counter()
    dose_pred = postprocess_prediction(pred, scale, upsample=resize)
    post_s = perf_counter() - post_t
    dose_bev_zyx = np.moveaxis(np.squeeze(dose_pred), 0, -1)  # (D,H,W) -> (z,y,x)
    return dose_bev_zyx, prep_s, forward_s, post_s


def run_timing_comparison(
    beamlets_dir: Path,
    model: torch.nn.Module,
    device: torch.device,
    *,
    bdl: BeamDataLibrary,
    grid_factor: int,
    rotation_backend: str,
    n_timing: Optional[int],
    timing_seed: int,
    repeats: int,
    scale: dict,
    normalize_flux: bool,
    downsampling_method: str,
) -> dict:
    """Per-beamlet timing: ADoTA flux construction vs DoTA-like BEV rotations.

    Both pipelines feed the SAME ADoTA model. ADoTA keeps the crop axis-aligned and
    pays the analytical flux construction; the DoTA-like path rotates the CT patch
    into the BEV (forward), runs the model, and rotates the predicted dose back to
    the field-aligned CT patch (inverse). The flux is rotated with the grid so the
    shared ADoTA model can run, but -- per the study design -- that flux rotation is
    NOT charged to the DoTA-like pipeline (a true DoTA model is single-channel).
    """
    spot_ids = sorted(p.name.removesuffix("_sim_res.json") for p in beamlets_dir.glob("*_sim_res.json"))
    if not spot_ids:
        raise FileNotFoundError(f"No beamlets found under {beamlets_dir}")
    if n_timing is not None and n_timing < len(spot_ids):
        spot_ids = sorted(random.Random(timing_seed).sample(spot_ids, n_timing))
    logger.info(
        "Timing %d beamlet(s) (repeats=%d, rotation_backend=%s) ...",
        len(spot_ids), repeats, rotation_backend,
    )

    metrics: dict[str, list[float]] = {
        k: [] for k in ("flux", "ct_to_bev", "dose_to_ct", "prep", "forward", "post")
    }
    for spot_id in spot_ids:
        ct, _flux_stored, sim_res = _load_beamlet(beamlets_dir, spot_id)
        angles = tuple(float(a) for a in sim_res["simulation_log"]["beamlet_angles"])
        energy = float(sim_res["initial_energy"])

        # ADoTA reinterpretation: analytical flux construction (CPU).
        flux, t_flux = _time_flux_construction(sim_res, ct.shape, bdl, grid_factor, repeats)

        # DoTA reinterpretation (1/2): rotate the CT patch into the BEV.
        rotated_ct, t_ct_to_bev = rotate_beamlet_crop(
            ct, angles, backend=rotation_backend, device=str(device), repeats=repeats
        )
        # Flux rotated with the grid so the shared ADoTA model gets a consistent
        # (perpendicular) input -- NOT timed (a real DoTA model has no flux channel).
        rotated_flux, _ = rotate_beamlet_crop(
            flux, angles, backend=rotation_backend, device=str(device), repeats=1
        )

        # Shared model path; also yields the BEV dose to rotate back.
        dose_bev, t_prep, t_forward, t_post = _infer_bev_dose(
            model, device, rotated_ct, rotated_flux, energy,
            scale=scale, normalize_flux=normalize_flux,
            downsampling_method=downsampling_method, grid_factor=grid_factor,
        )

        # DoTA reinterpretation (2/2): rotate the BEV dose back to the field-aligned patch.
        _, t_dose_to_ct = rotate_beamlet_crop(
            dose_bev, angles, inverse=True, backend=rotation_backend,
            device=str(device), repeats=repeats,
        )

        metrics["flux"].append(t_flux)
        metrics["ct_to_bev"].append(t_ct_to_bev)
        metrics["dose_to_ct"].append(t_dose_to_ct)
        metrics["prep"].append(t_prep)
        metrics["forward"].append(t_forward)
        metrics["post"].append(t_post)

    return _build_timing_report(
        metrics,
        n_beamlets=len(spot_ids),
        grid_factor=grid_factor,
        rotation_backend=rotation_backend,
        repeats=repeats,
        device=str(device),
    )


def _build_timing_report(
    metrics: dict[str, list[float]],
    *,
    n_beamlets: int,
    grid_factor: int,
    rotation_backend: str,
    repeats: int,
    device: str,
) -> dict:
    """Aggregate per-beamlet step times into totals + per-beamlet medians/means."""
    def agg(values: list[float]) -> dict:
        arr = np.asarray(values, dtype=float)
        return {
            "total_s": float(arr.sum()),
            "ms_median": float(np.median(arr) * 1000.0),
            "ms_mean": float(np.mean(arr) * 1000.0),
        }

    dota = np.asarray(metrics["ct_to_bev"], float) + np.asarray(metrics["dose_to_ct"], float)
    adota = np.asarray(metrics["flux"], float)
    report = {
        "n_beamlets": n_beamlets,
        "grid_factor": grid_factor,
        "rotation_backend": rotation_backend,
        "repeats": repeats,
        "device": device,
        "steps": {name: agg(values) for name, values in metrics.items()},
        "comparison": {
            "adota_reinterpretation_ms_median": float(np.median(adota) * 1000.0),
            "dota_reinterpretation_ms_median": float(np.median(dota) * 1000.0),
            "dota_reinterpretation_total_s": float(dota.sum()),
            "speedup_dota_over_adota": float(np.median(dota) / max(np.median(adota), 1e-12)),
        },
    }
    return report


def _format_timing_table(report: dict) -> str:
    """Render the DoTA-vs-ADoTA per-beamlet timing as a PrettyTable.

    Mirrors the sectioned, content-sized layout of
    ``scripts/run_plan_opentps.py::_format_timing_report``.
    """
    from prettytable import PrettyTable

    steps = report["steps"]
    cmp = report["comparison"]
    gf = report["grid_factor"]
    prep_lbl = "input prep (downsample)" if gf == 1 else "input prep (no resize)"
    post_lbl = "postprocess (upsample)" if gf == 1 else "postprocess (no resize)"

    def fs(step: dict) -> str:
        return f"{step['total_s']:.2f}"

    def fms(step: dict) -> str:
        return f"{step['ms_median']:.3f}"

    sections: list[list[list[str]]] = []

    # ADoTA reinterpretation.
    sections.append([
        ["ADoTA (ours)", "flux construction (CPU)", fs(steps["flux"]), fms(steps["flux"]),
         "analytical projection, no rotation"],
    ])

    # DoTA-like reinterpretation.
    dota_total_s = steps["ct_to_bev"]["total_s"] + steps["dose_to_ct"]["total_s"]
    sections.append([
        ["DoTA-like", "CT patch -> BEV (forward rot)", fs(steps["ct_to_bev"]), fms(steps["ct_to_bev"]),
         "beamlet patch rotation to BEV"],
        ["", "BEV dose -> CT patch (back rot)", fs(steps["dose_to_ct"]), fms(steps["dose_to_ct"]),
         "field-aligned dose rotation"],
        ["", "reinterpretation subtotal", f"{dota_total_s:.2f}",
         f"{cmp['dota_reinterpretation_ms_median']:.3f}", "flux rotation not charged"],
    ])

    # Shared model path (identical for both; excluded from the comparison).
    sections.append([
        ["Shared", prep_lbl, fs(steps["prep"]), fms(steps["prep"]), "same input for both"],
        ["", "ADoTA forward", fs(steps["forward"]), fms(steps["forward"]), "same model for both"],
        ["", post_lbl, fs(steps["post"]), fms(steps["post"]), ""],
    ])

    # Head-to-head.
    ratio = cmp["speedup_dota_over_adota"]
    sections.append([
        ["Comparison", "ADoTA reinterpretation", "", f"{cmp['adota_reinterpretation_ms_median']:.3f}",
         "flux construction"],
        ["", "DoTA reinterpretation", "", f"{cmp['dota_reinterpretation_ms_median']:.3f}",
         "CT->BEV + dose->CT"],
        ["", "speedup (DoTA / ADoTA)", "", f"{ratio:.1f}x", "x cheaper with ADoTA"],
    ])

    table = PrettyTable()
    table.field_names = ["Pipeline", "Step", "Total [s]", "ms/beamlet", "Notes"]
    table.align["Pipeline"] = "l"
    table.align["Step"] = "l"
    table.align["Total [s]"] = "r"
    table.align["ms/beamlet"] = "r"
    table.align["Notes"] = "l"
    for si, sec in enumerate(sections):
        for ri, row in enumerate(sec):
            divider = (ri == len(sec) - 1) and (si != len(sections) - 1)
            table.add_row(row, divider=divider)

    header = (
        f"PER-BEAMLET REINTERPRETATION TIMING (DoTA-like vs ADoTA)\n"
        f"  {report['n_beamlets']} beamlets, grid_factor={gf}, "
        f"rotation={report['rotation_backend']} (CPU), repeats={report['repeats']}, "
        f"inference on {report['device']}, flux + rotations on CPU"
    )
    return header + "\n" + table.get_string()


def estimate_end_to_end(
    plan_directory: PlanDirectory,
    model: torch.nn.Module,
    device: torch.device,
    *,
    output_path: Path,
    grid_factor: int,
    batch_size: int,
    ct_to_bev_median_s: float,
    dose_to_ct_median_s: float,
) -> dict:
    """Estimate the full-plan time of both pipelines from one real ADoTA run.

    Runs the realistic batched ADoTA streaming pipeline (every shared step --
    per-field rotation, crop, flux, prep, forward, post, deposit, de-rotation --
    measured on the whole plan) and derives the DoTA-like plan time by
    substitution: a true DoTA model is single-channel, so the analytical **flux is
    removed** and the **two per-beamlet BEV rotations** (CT->BEV + dose->CT,
    measured in the harness) are **added**. Everything else is identical, so the
    only swapped component is the per-beamlet reinterpretation. ``write`` (disk) is
    excluded so the comparison is compute-only.

    Device split: the ADoTA forward runs on ``device`` (GPU -- the shared,
    legitimately accelerated component), while the per-beamlet reinterpretation
    stays on the CPU for BOTH pipelines -- the analytical flux (CPU) and the DoTA
    rotations (CPU) -- because the original DoTA authors provided no GPU path for
    the BEV reinterpolation.
    """
    config = StreamingConfig(
        grid_factor=grid_factor,
        batch_size=batch_size,
        flux_on_gpu=False,
        precision="fp32",
    )
    summary = run_streaming_pipeline(plan_directory, model, device, output_path, config)

    timing = summary["timing"]
    n_spots = int(summary["n_spots"])
    n_fields = int(summary["n_fields"])
    compute_steps = {k: float(v) for k, v in timing.items() if k != "write"}
    adota_compute_s = float(sum(compute_steps.values()))
    flux_total_s = float(timing.get("flux", 0.0))

    ct_to_bev_total_s = n_spots * ct_to_bev_median_s
    dose_to_ct_total_s = n_spots * dose_to_ct_median_s
    # DoTA-like: drop the flux channel, add the two per-beamlet rotations.
    dota_compute_s = adota_compute_s - flux_total_s + ct_to_bev_total_s + dose_to_ct_total_s

    return {
        "n_spots": n_spots,
        "n_fields": n_fields,
        "grid_factor": grid_factor,
        "batch_size": batch_size,
        "device": str(device),
        "adota_steps_s": compute_steps,
        "adota_compute_s": adota_compute_s,
        "flux_total_s": flux_total_s,
        "ct_to_bev_total_s": ct_to_bev_total_s,
        "dose_to_ct_total_s": dose_to_ct_total_s,
        "dota_compute_s": dota_compute_s,
        "speedup_dota_over_adota": dota_compute_s / max(adota_compute_s, 1e-9),
        "ct_to_bev_median_ms": ct_to_bev_median_s * 1000.0,
        "dose_to_ct_median_ms": dose_to_ct_median_s * 1000.0,
    }


def _format_end_to_end_table(e2e: dict) -> str:
    """Render the full-plan DoTA-vs-ADoTA timing as a PrettyTable."""
    from prettytable import PrettyTable

    n = max(int(e2e["n_spots"]), 1)
    steps = e2e["adota_steps_s"]
    # Step label, ADoTA seconds, DoTA seconds (None -> blank). Shared steps appear
    # in both columns; the reinterpretation rows differ.
    order = [
        ("rotation (per field, gantry)", "rotation"),
        ("CT cropping", "crop"),
        ("input prep", "prep"),
        ("ADoTA forward (batched)", "forward"),
        ("postprocess", "post"),
        ("deposit", "deposit"),
        ("de-rotate (per field)", "derotate"),
    ]

    def ms(total_s: float) -> str:
        return f"{total_s / n * 1000.0:.3f}"

    rows: list[list[str]] = []
    # Shared steps (same in both pipelines).
    for label, key in order:
        if key in steps:
            s = steps[key]
            rows.append([label, f"{s:.2f}", f"{s:.2f}", ms(s)])
    shared_divider_idx = len(rows)

    # Reinterpretation rows (the only difference).
    flux_s = e2e["flux_total_s"]
    ct_s = e2e["ct_to_bev_total_s"]
    dose_s = e2e["dose_to_ct_total_s"]
    rows.append(["flux construction (ADoTA only)", f"{flux_s:.2f}", "-", ms(flux_s)])
    rows.append(["CT patch -> BEV (DoTA only)", "-", f"{ct_s:.2f}", ms(ct_s)])
    rows.append(["BEV dose -> CT patch (DoTA only)", "-", f"{dose_s:.2f}", ms(dose_s)])

    table = PrettyTable()
    table.field_names = ["Step", "ADoTA [s]", "DoTA-like [s]", "ms/beamlet"]
    table.align["Step"] = "l"
    table.align["ADoTA [s]"] = "r"
    table.align["DoTA-like [s]"] = "r"
    table.align["ms/beamlet"] = "r"
    for i, row in enumerate(rows):
        table.add_row(row, divider=(i == shared_divider_idx - 1))

    adota_total = e2e["adota_compute_s"]
    dota_total = e2e["dota_compute_s"]
    table.add_row(["PLAN TOTAL (compute)", f"{adota_total:.2f}", f"{dota_total:.2f}", ""], divider=True)
    table.add_row(
        ["SPEEDUP (DoTA / ADoTA)", "", f"{e2e['speedup_dota_over_adota']:.2f}x", ""]
    )

    header = (
        f"END-TO-END PLAN TIMING (DoTA-like vs ADoTA)\n"
        f"  {e2e['n_spots']} spots, {e2e['n_fields']} fields, grid_factor={e2e['grid_factor']}, "
        f"batch={e2e['batch_size']}, inference on {e2e['device']}, "
        f"flux + rotations on CPU\n"
        f"  ADoTA reinterpretation = analytical flux; DoTA reinterpretation = "
        f"CT->BEV + dose->CT per beamlet"
    )
    return header + "\n" + table.get_string()


@app.command()
def main(
    config: Annotated[
        Optional[Path], typer.Option(help="Path to YAML configuration file.")
    ] = None,
    plan_dir: Annotated[Optional[Path], typer.Option(help="OpenTPS plan directory.")] = None,
    model_name: Annotated[Optional[str], typer.Option(help="Model dir under models/.")] = None,
    model_fname: Annotated[Optional[str], typer.Option(help="Model weights filename.")] = None,
    bdl_path: Annotated[Optional[Path], typer.Option(help="BDL path (default: plan-local).")] = None,
    device_index: Annotated[Optional[int], typer.Option(help="CUDA device index (-1 = CPU).")] = None,
    runs_dir: Annotated[Optional[Path], typer.Option(help="Base directory for run outputs.")] = None,
    grid_factor: Annotated[Optional[int], typer.Option(help="Field resampling factor (1 or 2).")] = None,
    rotation_backend: Annotated[Optional[str], typer.Option(help="scipy or torch.")] = None,
    n_spots: Annotated[Optional[int], typer.Option(help="Extract only the first N spots.")] = None,
    beams: Annotated[Optional[str], typer.Option(help="Comma-separated beam indices.")] = None,
    n_figures: Annotated[Optional[int], typer.Option(help="Beamlets to visualise.")] = None,
    figure_selection: Annotated[
        Optional[str], typer.Option(help="Beamlet selection: 'max_angle' or 'random'.")
    ] = None,
    figure_seed: Annotated[Optional[int], typer.Option(help="Figure sampling seed.")] = None,
    run_timing: Annotated[
        Optional[bool], typer.Option(help="Run the DoTA-vs-ADoTA timing comparison.")
    ] = None,
    n_timing: Annotated[
        Optional[int], typer.Option(help="Beamlets to time (null = all).")
    ] = None,
    timing_repeats: Annotated[
        Optional[int], typer.Option(help="Timed repeats per CPU op (median reported).")
    ] = None,
    timing_seed: Annotated[Optional[int], typer.Option(help="Timing sampling seed.")] = None,
    run_end_to_end: Annotated[
        Optional[bool], typer.Option(help="Estimate full-plan DoTA-vs-ADoTA time (CPU).")
    ] = None,
    batch_size: Annotated[
        Optional[int], typer.Option(help="Streaming batch size for the end-to-end run.")
    ] = None,
    overwrite: Annotated[Optional[bool], typer.Option(help="Allow non-empty beamlets dir.")] = None,
    verbose: Annotated[Optional[bool], typer.Option(help="Verbose/debug logging.")] = None,
) -> None:
    """Run staged CPU extraction, render rotation figures, and time both pipelines."""
    cfg: dict = load_yaml_config(config) if config is not None else {}

    def pick(cli, key, default=None):
        return cli if cli is not None else cfg.get(key, default)

    plan_dir = Path(pick(plan_dir, "plan_dir"))
    model_name = pick(model_name, "model_name", "DoTA_v3_grid_search_v11")
    model_fname = pick(model_fname, "model_fname", "best_model.pth")
    bdl_value = pick(bdl_path, "bdl_path")
    bdl_path = Path(bdl_value) if bdl_value else None
    device_index = pick(device_index, "device_index", 0)
    runs_dir = Path(pick(runs_dir, "runs_dir", ROOT_DIR / "runs"))
    grid_factor = int(pick(grid_factor, "grid_factor", 2))
    rotation_backend = pick(rotation_backend, "rotation_backend", "scipy")
    n_spots = pick(n_spots, "n_spots")
    beams_list = _coerce_beams(pick(beams, "beams"))
    n_figures = int(pick(n_figures, "n_figures", 6))
    figure_selection = pick(figure_selection, "figure_selection", "max_angle")
    figure_seed = int(pick(figure_seed, "figure_seed", 0))
    run_timing = bool(pick(run_timing, "run_timing", True))
    n_timing = pick(n_timing, "n_timing", 300)
    n_timing = int(n_timing) if n_timing is not None else None
    timing_repeats = int(pick(timing_repeats, "timing_repeats", 3))
    timing_seed = int(pick(timing_seed, "timing_seed", 0))
    run_end_to_end = bool(pick(run_end_to_end, "run_end_to_end", True))
    batch_size = int(pick(batch_size, "batch_size", 56))
    overwrite = bool(pick(overwrite, "overwrite", True))
    verbose = bool(pick(verbose, "verbose", False))

    if rotation_backend not in {"scipy", "torch"}:
        raise typer.BadParameter("--rotation-backend must be 'scipy' or 'torch'.")
    if grid_factor not in {1, 2}:
        raise typer.BadParameter("--grid-factor must be 1 or 2.")

    run_dir = setup_run_directory(runs_dir, prefix="beamlet_bev_rot_", subdirs=("figures",))
    setup_logging(run_dir, verbose=verbose, log_filename="run.log")
    device = resolve_device(device_index)
    beamlets_dir = run_dir / "beamlets"

    logger.info("Run directory: %s", run_dir)
    logger.info("Plan: %s", plan_dir)
    logger.info("Device: %s for inference; flux + rotations on CPU", device)

    run_config = {
        "plan_dir": str(plan_dir),
        "model_name": model_name,
        "model_fname": model_fname,
        "bdl_path": str(bdl_path) if bdl_path else None,
        "device": str(device),
        "grid_factor": grid_factor,
        "rotation_backend": rotation_backend,
        "n_spots": n_spots,
        "beams": beams_list,
        "n_figures": n_figures,
        "figure_seed": figure_seed,
    }
    (run_dir / "config.json").write_text(json.dumps(run_config, indent=2))

    plan_directory = load_plan_directory(plan_dir, bdl_path=bdl_path)
    manifest = run_staged_cpu_extraction(
        plan_directory,
        beamlets_dir,
        bdl_path=bdl_path,
        grid_factor=grid_factor,
        n_spots=n_spots,
        beams=beams_list,
        overwrite=overwrite,
    )
    logger.info(
        "Extracted %d spots across %d field(s); CPU flux stored under %s",
        manifest["n_spots"],
        manifest["n_fields"],
        beamlets_dir,
    )

    figures = generate_validation_figures(
        beamlets_dir,
        run_dir / "figures",
        n_figures=n_figures,
        figure_seed=figure_seed,
        rotation_backend=rotation_backend,
        device=str(device),
        selection=figure_selection,
    )
    (run_dir / "figure_summary.json").write_text(json.dumps(figures, indent=2))
    logger.info("Wrote %d validation figure(s) -> %s", len(figures), run_dir / "figures")

    if run_timing:
        model_path = ROOT_DIR / "models" / model_name / model_fname
        hyperparams_path = ROOT_DIR / "models" / model_name / "hyperparams.json"
        logger.info("Loading ADoTA model: %s", model_path)
        model = load_model(model_path, hyperparams_path, device)
        bdl = BeamDataLibrary.from_file(plan_directory.bdl_path)
        report = run_timing_comparison(
            beamlets_dir,
            model,
            device,
            bdl=bdl,
            grid_factor=grid_factor,
            rotation_backend=rotation_backend,
            n_timing=n_timing,
            timing_seed=timing_seed,
            repeats=timing_repeats,
            scale=DEFAULT_SCALE,
            normalize_flux=True,
            downsampling_method="interpolation",
        )
        (run_dir / "timing_report.json").write_text(json.dumps(report, indent=2))
        logger.info("\n%s", _format_timing_table(report))
        logger.info("Timing report written to %s", run_dir / "timing_report.json")

        if run_end_to_end:
            logger.info("Estimating end-to-end plan time (real ADoTA streaming run) ...")
            e2e = estimate_end_to_end(
                plan_directory,
                model,
                device,
                output_path=run_dir / "Dose_ADoTA_e2e.mhd",
                grid_factor=grid_factor,
                batch_size=batch_size,
                ct_to_bev_median_s=report["steps"]["ct_to_bev"]["ms_median"] / 1000.0,
                dose_to_ct_median_s=report["steps"]["dose_to_ct"]["ms_median"] / 1000.0,
            )
            (run_dir / "end_to_end_report.json").write_text(json.dumps(e2e, indent=2))
            logger.info("\n%s", _format_end_to_end_table(e2e))
            logger.info("End-to-end report written to %s", run_dir / "end_to_end_report.json")

    logger.info("Done. Outputs in %s", run_dir)


if __name__ == "__main__":
    app()
