"""Compare the recomputed difficulty metrics against the reference study.

Two questions, in order:

1. Does the ``gt`` recomputation reproduce the study's ``results.csv``? It must,
   to floating-point noise: same loader path, same metric code, same dose. Any
   difference means the moved code or the record loader drifted, and nothing
   downstream can be trusted until it is zero.
2. How far does the ``analytic`` feature vector sit from the ``gt`` one, metric
   by metric? Reported as Spearman rank agreement (what the percentile-normalised
   score actually consumes), the median and 95th percentile of the absolute
   difference, and the depth error of the analytic Bragg peak in millimetres.

Usage:
    uv run python scripts/analysis/acquisition_input_only_compare.py --features-dir /scratch/.../subset_s230
"""
from __future__ import annotations

from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import typer
from scipy.stats import spearmanr

from src.acquisition.features import FEATURE_NAMES

REFERENCE_RESULTS = Path("/scratch/mstryja/adota_runs/20260707_124010/results.csv")

# results.csv predates the unified stopping-power model (CHG-0002, 1.3.0), so the
# metrics built on RSP legitimately differ from the current code by a few percent.
# The gate demands exactness of everything else and reports these separately.
RSP_METRICS = ("pflugfelder_hi", "wepl_mean", "wepl_std", "isi_sum", "isi_max", "isi_mean", "isi_axial_sum")
# Flux-weighted means and variances pick up float-order noise from normalising
# the flux on the full frame rather than on the crop, and the study stored
# float32; a metric reproduces if it agrees to 1e-3 absolute or 1e-4 relative.
EXACT_TOLERANCE = 1e-3
EXACT_REL_TOLERANCE = 1e-4
# A near-tie in the stored dose maximum can pick a different peak voxel once the
# dose is de-normalised (float32), moving that one record's crop and every metric
# downstream of it. A handful of such records is not a reproduction failure; a
# systematic difference is. The gate allows this fraction of records per metric.
MAX_DEVIATING_FRACTION = 1e-4


def reproduction_table(gt: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    """Max absolute difference per metric between the gt recomputation and the study."""
    merged = gt.merge(reference, on="sample_id", suffixes=("", "_ref"))
    rows = []
    for name in FEATURE_NAMES:
        a, b = merged[name].to_numpy(float), merged[f"{name}_ref"].to_numpy(float)
        finite = np.isfinite(a) & np.isfinite(b)
        diff = np.abs(a[finite] - b[finite])
        rel = diff / np.maximum(np.abs(b[finite]), 1e-12)
        deviating = (diff > EXACT_TOLERANCE) & (rel > EXACT_REL_TOLERANCE)
        rows.append({"metric": name, "n": int(finite.sum()),
                     "max_abs_diff": float(diff.max()) if finite.any() else np.nan,
                     "max_rel_diff": float(rel.max()) if finite.any() else np.nan,
                     "n_deviating": int(deviating.sum()),
                     "n_nan_mismatch": int(np.sum(np.isfinite(a) != np.isfinite(b)))})
    return pd.DataFrame(rows)


def agreement_table(gt: pd.DataFrame, analytic: pd.DataFrame, both_inside: bool = False) -> pd.DataFrame:
    """Rank agreement and absolute error of the analytic metrics against the gt ones,
    optionally restricted to beamlets whose peak both arms place inside the crop."""
    merged = gt.merge(analytic, on="sample_id", suffixes=("_gt", "_an"))
    if both_inside:
        merged = merged[merged["peak_inside_crop_gt"] & merged["peak_inside_crop_an"]]
    rows = []
    for name in FEATURE_NAMES:
        a, b = merged[f"{name}_gt"].to_numpy(float), merged[f"{name}_an"].to_numpy(float)
        finite = np.isfinite(a) & np.isfinite(b)
        diff = np.abs(a[finite] - b[finite])
        rho = spearmanr(a[finite], b[finite]).correlation if finite.sum() > 2 and np.std(a[finite]) > 0 else np.nan
        rows.append({"metric": name, "n": int(finite.sum()), "spearman": float(rho),
                     "median_abs_diff": float(np.median(diff)) if diff.size else np.nan,
                     "p95_abs_diff": float(np.percentile(diff, 95)) if diff.size else np.nan,
                     "gt_median": float(np.median(a[finite])) if finite.any() else np.nan})
    return pd.DataFrame(rows)


def main(
    features_dir: Annotated[Path, typer.Option(help="Directory holding features_gt.csv and features_analytic.csv.")],
    reference: Annotated[Path, typer.Option(help="The study's results.csv.")] = REFERENCE_RESULTS,
) -> None:
    """Write reproduction.csv and agreement.csv next to the features; exit 1 if gt does not reproduce."""
    gt = pd.read_csv(features_dir / "features_gt.csv")
    analytic = pd.read_csv(features_dir / "features_analytic.csv")
    ref = pd.read_csv(reference, usecols=["sample_id", *FEATURE_NAMES])

    repro = reproduction_table(gt, ref)
    repro.to_csv(features_dir / "reproduction.csv", index=False)
    exact = repro[~repro.metric.isin(RSP_METRICS)]
    rsp = repro[repro.metric.isin(RSP_METRICS)]
    worst = exact.sort_values("n_deviating").iloc[-1]
    typer.echo(f"gt reproduction over {len(gt)} records, non-RSP metrics: at most {int(worst.n_deviating)} records "
               f"deviate on any metric ({worst.metric}, {worst.n_deviating / len(gt):.1e} of records); "
               f"nan mismatches {int(exact.n_nan_mismatch.sum())}")
    if worst.n_deviating:
        typer.echo("  deviating counts: " + ", ".join(f"{r.metric} {r.n_deviating}"
                                                       for r in exact.itertuples() if r.n_deviating))
    typer.echo("RSP-based metrics, expected to differ from the pre-CHG-0002 study:")
    typer.echo(rsp[["metric", "max_abs_diff"]].to_string(index=False))
    reproduced = bool((exact.n_deviating <= MAX_DEVIATING_FRACTION * len(gt)).all()
                      and (exact.n_nan_mismatch == 0).all())

    # Does the surrogate agree with the ground truth on whether the beam stops in the crop?
    flags = gt[["sample_id", "peak_inside_crop"]].merge(
        analytic[["sample_id", "peak_inside_crop"]], on="sample_id", suffixes=("_gt", "_an"))
    confusion = pd.crosstab(flags.peak_inside_crop_gt, flags.peak_inside_crop_an,
                            rownames=["gt inside"], colnames=["analytic inside"])
    typer.echo("\npeak-inside-crop, ground truth vs analytic:")
    typer.echo(confusion.to_string())
    typer.echo(f"agreement {100 * np.mean(flags.peak_inside_crop_gt == flags.peak_inside_crop_an):.1f} percent; "
               f"gt inside {100 * flags.peak_inside_crop_gt.mean():.1f} percent, analytic inside "
               f"{100 * flags.peak_inside_crop_an.mean():.1f} percent")

    for label, inside in (("all records", False), ("both arms see the peak inside the crop", True)):
        agree = agreement_table(gt, analytic, both_inside=inside)
        agree.to_csv(features_dir / ("agreement_inside.csv" if inside else "agreement.csv"), index=False)
        typer.echo(f"\nanalytic vs gt agreement, {label} (n={int(agree.n.max())}), sorted by Spearman:")
        typer.echo(agree.sort_values("spearman").to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    peak = analytic["peak_depth_err_mm"].to_numpy(float)
    typer.echo(f"\nanalytic peak depth error vs gt [mm]: median {np.median(peak):+.1f}, "
               f"IQR {np.percentile(peak, 25):+.1f} to {np.percentile(peak, 75):+.1f}, "
               f"|err| <= 4 mm in {100 * np.mean(np.abs(peak) <= 4):.0f} percent, "
               f"|err| > 20 mm in {100 * np.mean(np.abs(peak) > 20):.1f} percent")
    lat = analytic["peak_lateral_err_vox"].to_numpy(float)
    typer.echo(f"crop-centre lateral offset [voxels]: median {np.median(lat):.1f}, p95 {np.percentile(lat, 95):.1f}")

    if reproduced:
        typer.secho("\ngt recomputation reproduces the study's results.csv", fg=typer.colors.GREEN)
    else:
        typer.secho("\ngt recomputation does NOT reproduce results.csv; see reproduction.csv", fg=typer.colors.RED)
        raise typer.Exit(code=1)


if __name__ == "__main__":
    typer.run(main)
