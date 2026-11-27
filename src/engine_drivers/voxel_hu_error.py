"""Dose error per voxel against the voxel's own HU (the ``voxel_hu_error`` stage).

For every analysed beamlet, the voxels where the Monte Carlo dose reaches at least
``cutoff`` (0.1%) of the beamlet's Monte Carlo maximum are binned by their HU (50 HU
bins on round edges, so -1000, -500, 0 and 300 are edges) and by a dose band relative
to that maximum. Per voxel and engine:

* the **global error** ``e = 100 (D_engine - D_MC) / max(D_MC)``, as a share of the
  beamlet's maximum (the normalisation of global gamma), kept signed and absolute;
* the **local error** ``100 |D_engine - D_MC| / D_MC``, only where the Monte Carlo dose
  is at least 10% of its maximum. It differs from the repository's MAPE
  (``record_metrics``), which divides by the engine's dose and selects voxels on it.

Only sums and counts are stored per beamlet, engine, band and HU bin, so a voxel-pooled
mean is exact and its interval comes from resampling patients (a weight matrix times
per-patient sums). A per-beamlet median over cells with enough voxels sits beside it.

The three arrays share one layout: the export's CT and Monte Carlo dose are stored
``(H, W, D)`` and transposed ``(2, 0, 1)`` exactly as :func:`mc_model_grid` does; the
engine doses are saved in that model layout by the earlier stages.
"""
from __future__ import annotations

import logging
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Mapping, Sequence, Tuple

import h5py
import numpy as np
import pandas as pd

from src.engine_drivers.hu_histogram import export_hu
from src.engine_drivers.scoring import export_scale, mc_model_grid

logger = logging.getLogger(__name__)

__all__ = ["BANDS", "HU_EDGES", "TISSUE_CLASSES", "accumulate", "band_order", "beamlet_medians", "pooled",
           "run_voxel_hu_error", "tissue_pooled", "voxel_summary"]

HU_EDGES = np.arange(-1050.0, 3101.0, 50.0)
BANDS: Tuple[Tuple[str, float, float], ...] = (("0.1-10%", 0.001, 0.10), ("10-50%", 0.10, 0.50),
                                               ("50-100%", 0.50, np.inf))
"""Dose bands relative to the beamlet's Monte Carlo maximum, ``lo <= d < hi``."""
TISSUE_CLASSES: Tuple[Tuple[str, float, float], ...] = (("below -500 HU", -np.inf, -500.0),
                                                        ("-500 to 300 HU", -500.0, 300.0),
                                                        ("above 300 HU", 300.0, np.inf))
"""Tissue classes on whole bins: ``lo <= bin_low_hu < hi`` (the edges -500 and 300 are bin edges)."""
SUMS = ("n", "sum_abs", "sum_signed", "n_local", "sum_local")


def _beamlet_sums(hu: np.ndarray, mc: np.ndarray, dose: np.ndarray, cutoff: float, local_min: float) -> np.ndarray:
    """``(len(BANDS) * n_hu_bins, 5)`` sums for one beamlet and engine, columns as :data:`SUMS`."""
    n_hu = HU_EDGES.size - 1
    peak = float(mc.max())
    rel = mc / peak
    sel = rel >= cutoff
    band_lo = np.array([b[1] for b in BANDS])
    band = np.searchsorted(band_lo, rel[sel], side="right") - 1
    hu_bin = np.clip(np.searchsorted(HU_EDGES, hu[sel], side="right") - 1, 0, n_hu - 1)
    cell = band * n_hu + hu_bin
    diff = (dose[sel] - mc[sel]).astype(np.float64)
    e = 100.0 * diff / peak
    local = rel[sel] >= local_min
    loc_val = np.where(local, 100.0 * np.abs(diff) / np.where(local, mc[sel], 1.0), 0.0)
    size = len(BANDS) * n_hu
    return np.stack([np.bincount(cell, minlength=size).astype(np.float64),
                     np.bincount(cell, weights=np.abs(e), minlength=size),
                     np.bincount(cell, weights=e, minlength=size),
                     np.bincount(cell, weights=local.astype(np.float64), minlength=size),
                     np.bincount(cell, weights=loc_val, minlength=size)], axis=1)


def accumulate(dataset_path: Path, dose_dirs: Mapping[str, Path], ids: Sequence[str], cutoff: float = 0.001,
               local_min: float = 0.10, log_every: int = 1000) -> pd.DataFrame:
    """Per beamlet, engine, band and HU bin: the sums of :data:`SUMS` (occupied cells only)."""
    n_hu = HU_EDGES.size - 1
    band_names = np.repeat([b[0] for b in BANDS], n_hu)
    bin_low = np.tile(HU_EDGES[:-1], len(BANDS))
    scale = export_scale(Path(dataset_path))
    frames: List[pd.DataFrame] = []
    start = perf_counter()
    with h5py.File(dataset_path, "r") as handle:
        ct_scale = {"min_ct": scale["min_ct"], "max_ct": scale["max_ct"]}
        for i, sid in enumerate(ids, 1):
            mc = mc_model_grid(handle, sid, scale)
            hu = np.ascontiguousarray(export_hu(np.asarray(handle[sid]["ct"][...]), ct_scale).transpose(2, 0, 1))
            for engine, folder in dose_dirs.items():
                dose = np.load(Path(folder) / f"{sid}.npy")
                if dose.shape != mc.shape:
                    raise ValueError(f"{engine} dose of {sid} has shape {dose.shape}, Monte Carlo {mc.shape}")
                sums = _beamlet_sums(hu, mc, dose, cutoff, local_min)
                keep = sums[:, 0] > 0
                frames.append(pd.DataFrame({"sample_id": sid, "engine": engine, "band": band_names[keep],
                                            "bin_low_hu": bin_low[keep],
                                            **{k: sums[keep, j] for j, k in enumerate(SUMS)}}))
            if i % log_every == 0 or i == len(ids):
                logger.info("voxel_hu_error: %d/%d beamlets (%.0f s)", i, len(ids), perf_counter() - start)
    out = pd.concat(frames, ignore_index=True)
    for k in ("n", "n_local"):
        out[k] = out[k].astype(np.int64)
    return out


def _with_site(cells: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    return cells.merge(meta[["sample_id", "patient_id", "site"]], on="sample_id", how="left", validate="m:1")


def _pooled_cells(frame: pd.DataFrame, keys: Sequence[str], n_boot: int, seed: int) -> pd.DataFrame:
    """Voxel-pooled means per ``keys`` cell with patient-resampling 95% intervals."""
    rng = np.random.default_rng(seed)
    rows = []
    per_patient = frame.groupby([*keys, "patient_id"], observed=True)[list(SUMS)].sum().reset_index()
    for (engine, site), part in per_patient.groupby(["engine", "site"], observed=True):
        patients = np.sort(part["patient_id"].unique())
        weights = rng.multinomial(len(patients), np.full(len(patients), 1.0 / len(patients)), size=n_boot)
        cell_keys = [k for k in keys if k not in ("engine", "site")]
        for cell, cpart in part.groupby(cell_keys, observed=True):
            mats = {k: cpart.set_index("patient_id")[k].reindex(patients, fill_value=0.0).to_numpy(float)
                    for k in SUMS}
            row = dict(zip(["engine", "site", *cell_keys], [engine, site, *np.atleast_1d(cell)]))
            row.update(voxels=int(mats["n"].sum()), patients=int((mats["n"] > 0).sum()))
            for name, num, den in (("abs", "sum_abs", "n"), ("signed", "sum_signed", "n"),
                                   ("local", "sum_local", "n_local")):
                total = mats[den].sum()
                row[f"{name}_mean"] = mats[num].sum() / total if total > 0 else np.nan
                with np.errstate(invalid="ignore", divide="ignore"):
                    boot = (weights @ mats[num]) / (weights @ mats[den])
                lo, hi = np.nanpercentile(boot, [2.5, 97.5]) if np.isfinite(boot).any() else (np.nan, np.nan)
                row[f"{name}_ci_low"], row[f"{name}_ci_high"] = lo, hi
            row["local_voxels"] = int(mats["n_local"].sum())
            rows.append(row)
    return pd.DataFrame(rows)


def pooled(cells: pd.DataFrame, meta: pd.DataFrame, n_boot: int = 500, seed: int = 0) -> pd.DataFrame:
    """Per engine, site, band and HU bin: voxel-pooled abs, signed and local error with intervals."""
    return _pooled_cells(_with_site(cells, meta), ["engine", "site", "band", "bin_low_hu"], n_boot, seed)


def tissue_pooled(cells: pd.DataFrame, meta: pd.DataFrame, n_boot: int = 500, seed: int = 0) -> pd.DataFrame:
    """The same per tissue class (:data:`TISSUE_CLASSES`) instead of HU bin."""
    frame = _with_site(cells, meta)
    labels = np.full(len(frame), "", dtype=object)
    for name, lo, hi in TISSUE_CLASSES:
        labels[(frame["bin_low_hu"] >= lo).to_numpy() & (frame["bin_low_hu"] < hi).to_numpy()] = name
    frame = frame.assign(tissue=labels)
    return _pooled_cells(frame, ["engine", "site", "band", "tissue"], n_boot, seed)


def beamlet_medians(cells: pd.DataFrame, meta: pd.DataFrame, min_voxels: int = 10) -> pd.DataFrame:
    """Per engine, site, band and HU bin: median and quartiles of the per-beamlet mean absolute
    error over beamlets with at least ``min_voxels`` voxels in that cell."""
    frame = _with_site(cells[cells["n"] >= min_voxels], meta)
    frame = frame.assign(beamlet_abs=frame["sum_abs"] / frame["n"])
    grouped = frame.groupby(["engine", "site", "band", "bin_low_hu"], observed=True)["beamlet_abs"]
    out = grouped.agg(beamlets="size", median="median", q25=lambda s: s.quantile(0.25),
                      q75=lambda s: s.quantile(0.75)).reset_index()
    return out


def band_order() -> Dict[str, int]:
    return {b[0]: i for i, b in enumerate(BANDS)}


def _ci(mean: float, lo: float, hi: float) -> str:
    return "n/a" if not pd.notna(mean) else f"{mean:+.2f} [{lo:+.2f}, {hi:+.2f}]"


def voxel_summary(tissue: pd.DataFrame, n_beamlets: int, cutoff: float, local_min: float) -> List[str]:
    """``summary.md`` lines: method, one table per dose band (tissue classes), caveats."""
    lines = ["# Dose error per voxel against the voxel's HU", "",
             f"{n_beamlets} beamlets (the analysis population: analytic peak inside the crop), every voxel where "
             f"the Monte Carlo dose is at least {100 * cutoff:g}% of the beamlet's Monte Carlo maximum. Generated by "
             "the `voxel_hu_error` stage; numbers in `voxel_error_by_tissue.csv` and `voxel_error_by_hu.csv`.", "",
             "- **global error** e = 100 (D_engine - D_MC) / max(D_MC), in % of the beamlet's maximum dose "
             "(the normalisation of global gamma); |e| is its absolute value, e keeps the sign (positive = engine "
             "too high).",
             f"- **local error** = 100 |D_engine - D_MC| / D_MC, only where D_MC >= {100 * local_min:g}% of the "
             "maximum. This is not the repository's MAPE, which divides by the engine's dose and selects voxels on "
             "the engine's dose (10% of the engine's maximum).",
             "- Values are voxel-pooled means; [ ] are 95% intervals from resampling patients (500 resamples). The "
             "per-beamlet median of the absolute error is in `voxel_error_beamlet_medians.csv`.", ""]
    order = band_order()
    for band in sorted(tissue["band"].unique(), key=order.get):
        part = tissue[tissue["band"] == band]
        lines += [f"## Dose {band} of the maximum", "",
                  "| engine | site | tissue | voxels | patients | mean abs error |e| | mean signed error e | "
                  "mean local error |", "|---|---|---|---|---|---|---|---|"]
        for row in part.sort_values(["engine", "site", "tissue"]).itertuples(index=False):
            local = "n/a" if not pd.notna(row.local_mean) else f"{row.local_mean:.2f}"
            lines.append(f"| {row.engine} | {row.site} | {row.tissue} | {row.voxels} | {row.patients} | "
                         f"{_ci(row.abs_mean, row.abs_ci_low, row.abs_ci_high)} | "
                         f"{_ci(row.signed_mean, row.signed_ci_low, row.signed_ci_high)} | {local} |")
        lines.append("")
    lines += [
        "## Caveats", "",
        "- **Dose to medium against dose to water.** MCsquare runs with its default `Dose_to_Water_conversion "
        "Disabled` (the config writer never sets it), so the Monte Carlo, and ADoTA trained on it, give dose to "
        "medium. The DoseCUDA kernel deposits the water depth-dose at the voxel's water-equivalent depth with no "
        "stopping-power or density factor (`pencilBeamKernel`, IMPTClasses.cu), i.e. effectively dose to water. "
        "The pencil beam's error in bone (and to a smaller extent in other non-water tissue) therefore contains a "
        "dose-definition term, not only algorithm error. The same holds for its EXP-0023 errors.",
        "- **The lowest band is the noisiest.** Below 10% of the maximum the Monte Carlo (1e6 primaries) is noisy, "
        "and selecting voxels on the noisy reference biases the signed error negative; no noise floor was "
        "measured here, so this band cannot be separated from Monte Carlo noise.",
        "- **Association, not cause.** A range error caused upstream (lung, air, bone the beam crossed) lands in "
        "the peak and fall-off voxels whatever their own HU, so a curve says where errors sit, not why.",
        "- **HU are 2 mm averages** of the stored 1 mm CT, so voxels at an interface take intermediate HU; the "
        "bins between about -500 and -100 HU hold many such mixed voxels.",
        "- Voxels are counted once per beamlet; crops of one field overlap, so a CT voxel can count more than "
        "once.", ""]
    return lines


def run_voxel_hu_error(cfg: Mapping, run: Path) -> List[str]:
    """The stage: per-beamlet sums, pooled tables and figures under ``<run>/analysis/voxel_hu_error``."""
    from src.figures.voxel_hu_error import hu_error_grid

    vcfg = cfg.get("voxel_hu_error", {})
    cutoff, local_min = float(vcfg.get("dose_cutoff", 0.001)), float(vcfg.get("local_min", 0.10))
    n_boot, seed = int(vcfg.get("n_boot", 500)), int(vcfg.get("seed", 20261126))
    min_patients, formats = int(vcfg.get("min_patients", 5)), tuple(vcfg.get("formats", ["png", "pdf"]))
    out = run / "analysis" / "voxel_hu_error"
    out.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(run / "analysis" / "dataset_overview" / "beamlet_metadata.csv", dtype={"sample_id": str})
    ids = meta.loc[meta["in_analysis"].astype(bool), "sample_id"].tolist()
    dirs = {"ADoTA": run / "doses" / "adota", "Pencil beam": run / "doses" / "pba"}

    cells = accumulate(Path(cfg["dataset_path"]), dirs, ids, cutoff, local_min)
    cells.to_csv(out / "voxel_sums_per_beamlet.csv.gz", index=False)
    pooled(cells, meta, n_boot, seed).to_csv(out / "voxel_error_by_hu.csv", index=False)
    tissue_pooled(cells, meta, n_boot, seed).to_csv(out / "voxel_error_by_tissue.csv", index=False)
    beamlet_medians(cells, meta).to_csv(out / "voxel_error_beamlet_medians.csv", index=False)
    logger.info("voxel_hu_error: %d beamlets, %d cells", len(ids), len(cells))

    table = pd.read_csv(out / "voxel_error_by_hu.csv")
    bands = [b[0] for b in BANDS]
    note = (f"Voxels with Monte Carlo dose >= {100 * cutoff:g}% of the beamlet maximum; HU of the 2 mm model-grid "
            f"voxel; bins of 50 HU seen in at least {min_patients} patients; bands: 95% intervals from resampling "
            "patients. Grey: voxels per bin. Dotted: -500 and 300 HU.")
    hu_error_grid(table, value="abs", ylabel="|e| [% of max]", bands=bands, note=note,
                  title="Absolute dose error per voxel against its HU (% of the beamlet's maximum dose)",
                  figure_path=out / "voxel_abs_error_vs_hu", min_patients=min_patients, formats=formats)
    hu_error_grid(table, value="signed", ylabel="e [% of max]", bands=bands, zero_line=True, note=note,
                  title="Signed dose error per voxel against its HU (positive: engine too high)",
                  figure_path=out / "voxel_signed_error_vs_hu", min_patients=min_patients, formats=formats)
    hu_error_grid(table, value="local", ylabel="local [%]", bands=[b for b in bands if b != BANDS[0][0]],
                  note=note.replace("Voxels with", f"Local error, voxels with >= {100 * local_min:g}%; all with"),
                  title="Local dose error per voxel against its HU (% of the Monte Carlo dose in that voxel)",
                  figure_path=out / "voxel_local_error_vs_hu", min_patients=min_patients, formats=formats)
    tissue = pd.read_csv(out / "voxel_error_by_tissue.csv")
    return voxel_summary(tissue, len(ids), cutoff, local_min)
