"""
Bragg Peak Estimation – Multi-Method Comparison

Estimates the Bragg-peak depth for every beamlet in the training set
using multiple methods (GT-based IDD, CSDA range in water, density-
corrected CSDA, CT density gradient).  Compares all pre-inference
methods against the GT-based IDD ground truth.

Output: runs/{timestamp}/ with CSV, scatter plots, error histograms,
and energy-stratified breakdown.
"""

import csv
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Annotated, Callable, Dict, List, Optional, Protocol, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import typer
import yaml
from scipy.interpolate import interp1d
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.adota.config import DEFAULT_SCALE
from src.figures.ct_visualizations import plot_bp_estimation_diagnostic
from src.loaders.generator import H5PYGenerator
from src.processing.rsp import (
    DENSITY_WATER,
    hu_to_density,
    hu_to_rsp,
    hu_to_rsp_density,
)
from src.utils.scallers import inverse_minmax

logger = logging.getLogger(__name__)

app = typer.Typer(help="Bragg-peak estimation – multi-method comparison")


from src.schemas.results import BPRecord

# ═══════════════════════════════════════════════════════════════════════════
#  Estimator protocol & registry
# ═══════════════════════════════════════════════════════════════════════════


class BPEstimator(Protocol):
    """Protocol every BP estimation method must satisfy."""

    name: str

    def estimate(
        self,
        ct_hu: np.ndarray,
        flux: np.ndarray,
        energy_mev: float,
        dose: Optional[np.ndarray],
        resolution_mm: Tuple[float, float, float],
    ) -> float:
        """Return estimated BP depth in mm (along axis 0)."""
        ...


# Global method registry
_REGISTRY: Dict[str, Callable[..., "BPEstimator"]] = {}


def register(name: str):
    """Decorator that adds a factory to the registry."""

    def wrapper(cls):
        _REGISTRY[name] = cls
        return cls

    return wrapper


def build_estimator(method_cfg: dict, global_cfg: dict) -> "BPEstimator":
    """Instantiate an estimator from its config dict."""
    name = method_cfg["name"]
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown BP method '{name}'. Available: {list(_REGISTRY.keys())}"
        )
    return _REGISTRY[name](method_cfg=method_cfg, global_cfg=global_cfg)


# ═══════════════════════════════════════════════════════════════════════════
#  Lookup tables
# ═══════════════════════════════════════════════════════════════════════════


def load_pstar_table(path: Path) -> interp1d:
    """Load PSTAR CSV → interpolator  energy_MeV → CSDA_range_cm."""
    df = pd.read_csv(path, comment="#")
    return interp1d(
        df["kinetic_energy_MeV"].values,
        df["csda_range_cm"].values,
        kind="cubic",
        fill_value="extrapolate",
    )


def load_schneider_calibration(path: Path) -> dict:
    """Load Schneider HU→RSP piecewise-linear calibration YAML."""
    with open(path) as f:
        return yaml.safe_load(f)


def hu_to_rsp(ct_hu: np.ndarray, calibration: dict) -> np.ndarray:
    """Convert HU volume to relative stopping power (RSP).

    NOTE: This is a thin wrapper kept for backward compatibility.
    The canonical implementation lives in ``src.processing.rsp``.
    """
    from src.processing.rsp import hu_to_rsp as _hu_to_rsp

    return _hu_to_rsp(ct_hu, calibration=calibration)


def energy_to_r80_mm(energy_mev: float) -> float:
    """Grevillot et al. (2011) analytical fit: energy [MeV] → r80 [mm].

    Returns the 80 % distal dose fall-off depth in water (r80),
    which is the clinically relevant range definition used by most TPS.
    """
    ln_e = np.log(energy_mev)
    r80_cm = np.exp(-5.5064 + 1.2193 * ln_e + 0.15248 * ln_e**2 - 0.013296 * ln_e**3)
    return float(r80_cm) * 10.0  # cm → mm


# Schneider 1996 HU → density table and helpers are now imported from
# src.processing.rsp (hu_to_density, hu_to_rsp_density, DENSITY_WATER).


# ═══════════════════════════════════════════════════════════════════════════
#  Method 1: GT IDD (ground truth – reference)
# ═══════════════════════════════════════════════════════════════════════════


@register("gt_idd")
class GTIDDEstimator:
    """BP depth from the ground-truth Integrated Depth Dose."""

    name = "gt_idd"

    def __init__(self, method_cfg: dict, global_cfg: dict):
        self.proximal_fraction = method_cfg.get("proximal_fraction", 0.50)
        self.fall_fraction = method_cfg.get("fall_fraction", 0.10)

    def estimate(
        self,
        ct_hu: np.ndarray,
        flux: np.ndarray,
        energy_mev: float,
        dose: Optional[np.ndarray],
        resolution_mm: Tuple[float, float, float],
    ) -> float:
        if dose is None:
            return float("nan")
        idd = dose.sum(axis=(1, 2))
        idd_max = idd.max()
        if idd_max < 1e-9:
            return 0.0
        bp_idx = int(np.argmax(idd))
        return float(bp_idx) * resolution_mm[0]


# ═══════════════════════════════════════════════════════════════════════════
#  Method 2: CSDA range in pure water
# ═══════════════════════════════════════════════════════════════════════════


@register("csda_water")
class CSDAWaterEstimator:
    """BP depth = CSDA range in water (energy lookup only)."""

    name = "csda_water"

    def __init__(self, method_cfg: dict, global_cfg: dict):
        pstar_path = PROJECT_ROOT / global_cfg.get(
            "pstar_table", "data/proton_tables/pstar_water.csv"
        )
        self._interp = load_pstar_table(pstar_path)

    def estimate(
        self,
        ct_hu: np.ndarray,
        flux: np.ndarray,
        energy_mev: float,
        dose: Optional[np.ndarray],
        resolution_mm: Tuple[float, float, float],
    ) -> float:
        range_cm = float(self._interp(energy_mev))
        return range_cm * 10.0  # cm → mm


# ═══════════════════════════════════════════════════════════════════════════
#  Method 3: CSDA density-corrected (Schneider RSP)
# ═══════════════════════════════════════════════════════════════════════════


@register("csda_density_corrected")
class CSDACorrectedEstimator:
    """CSDA range in water scaled by mean RSP along the beam path."""

    name = "csda_density_corrected"

    def __init__(self, method_cfg: dict, global_cfg: dict):
        pstar_path = PROJECT_ROOT / global_cfg.get(
            "pstar_table", "data/proton_tables/pstar_water.csv"
        )
        self._interp = load_pstar_table(pstar_path)

        schneider_path = PROJECT_ROOT / method_cfg.get(
            "schneider_calibration",
            "data/proton_tables/schneider_default.yaml",
        )
        self._calibration = load_schneider_calibration(schneider_path)

    def estimate(
        self,
        ct_hu: np.ndarray,
        flux: np.ndarray,
        energy_mev: float,
        dose: Optional[np.ndarray],
        resolution_mm: Tuple[float, float, float],
    ) -> float:
        range_water_cm = float(self._interp(energy_mev))
        range_water_mm = range_water_cm * 10.0

        rsp = hu_to_rsp(ct_hu, self._calibration)

        # Flux-weighted mean RSP along beam path
        # Use central beam profile: sum flux laterally per depth slice,
        # weight each depth slice's mean RSP by that flux
        flux_per_depth = np.abs(flux).sum(axis=(1, 2))  # (D,)
        total_flux = flux_per_depth.sum()
        if total_flux < 1e-12:
            return range_water_mm  # fallback to water

        # Mean RSP per depth slice (lateral average weighted by flux)
        rsp_per_depth = np.zeros(ct_hu.shape[0])
        for k in range(ct_hu.shape[0]):
            fl_slice = np.abs(flux[k])
            fl_sum = fl_slice.sum()
            if fl_sum > 1e-12:
                rsp_per_depth[k] = (rsp[k] * fl_slice).sum() / fl_sum
            else:
                rsp_per_depth[k] = rsp[k].mean()

        # Walk from entrance accumulating WET until we reach range_water_mm
        dz = resolution_mm[0]
        wet_accum = 0.0
        for k in range(ct_hu.shape[0]):
            wet_accum += rsp_per_depth[k] * dz
            if wet_accum >= range_water_mm:
                # Linearly interpolate within this voxel
                overshoot = wet_accum - range_water_mm
                frac = (
                    overshoot / (rsp_per_depth[k] * dz)
                    if rsp_per_depth[k] * dz > 1e-12
                    else 0.0
                )
                return (float(k) - frac) * dz
        # Beam exits the volume
        return float(ct_hu.shape[0] - 1) * dz


# ═══════════════════════════════════════════════════════════════════════════
#  Method 4: CT density gradient
# ═══════════════════════════════════════════════════════════════════════════


@register("ct_density_gradient")
class CTDensityGradientEstimator:
    """Heuristic: deepest large CT gradient within the beam path."""

    name = "ct_density_gradient"

    def __init__(self, method_cfg: dict, global_cfg: dict):
        self.gradient_percentile = method_cfg.get("gradient_percentile", 95.0)
        self.min_depth_fraction = method_cfg.get("min_depth_fraction", 0.3)

    def estimate(
        self,
        ct_hu: np.ndarray,
        flux: np.ndarray,
        energy_mev: float,
        dose: Optional[np.ndarray],
        resolution_mm: Tuple[float, float, float],
    ) -> float:
        # Flux-weighted axial CT profile
        flux_per_depth = np.abs(flux).sum(axis=(1, 2))
        total_flux = flux_per_depth.sum()
        if total_flux < 1e-12:
            return 0.0

        ct_profile = np.zeros(ct_hu.shape[0])
        for k in range(ct_hu.shape[0]):
            fl_slice = np.abs(flux[k])
            fl_sum = fl_slice.sum()
            if fl_sum > 1e-12:
                ct_profile[k] = (ct_hu[k] * fl_slice).sum() / fl_sum
            else:
                ct_profile[k] = ct_hu[k].mean()

        # Axial gradient
        grad = np.abs(np.gradient(ct_profile))
        n_slices = len(grad)
        min_idx = int(self.min_depth_fraction * n_slices)

        # Threshold: only keep gradients above the percentile
        threshold = np.percentile(grad[min_idx:], self.gradient_percentile)
        candidates = np.where((grad >= threshold) & (np.arange(n_slices) >= min_idx))[0]

        if len(candidates) == 0:
            # Fallback: just use the deepest large gradient
            return float(np.argmax(grad[min_idx:]) + min_idx) * resolution_mm[0]

        # Return the deepest candidate
        return float(candidates[-1]) * resolution_mm[0]


# ═══════════════════════════════════════════════════════════════════════════
#  Method 5: Grevillot r80 + density-corrected WEPL  (OpenTPS-inspired)
# ═══════════════════════════════════════════════════════════════════════════


@register("r80_density_corrected")
class R80DensityCorrectedEstimator:
    """Grevillot r80 in water, walked through density-based RSP field.

    Approach (mirrors analytical TPS range calculation):
    1. energy → r80 via Grevillot analytical fit.
    2. CT HU → mass density → RSP ≈ ρ / ρ_water.
    3. Walk along depth (axis 0) accumulating WET = Σ RSP_k · Δz,
       weighted laterally by the flux profile.
    4. BP depth = geometric depth where WET reaches r80.
    """

    name = "r80_density_corrected"

    def __init__(self, method_cfg: dict, global_cfg: dict):
        pass  # no external files needed

    def estimate(
        self,
        ct_hu: np.ndarray,
        flux: np.ndarray,
        energy_mev: float,
        dose: Optional[np.ndarray],
        resolution_mm: Tuple[float, float, float],
    ) -> float:
        r80_mm = energy_to_r80_mm(energy_mev)

        rsp = hu_to_rsp_density(ct_hu)  # (D, H, W)

        # Flux-weighted mean RSP per depth slice
        dz = resolution_mm[0]
        n_depth = ct_hu.shape[0]
        rsp_per_depth = np.empty(n_depth)

        for k in range(n_depth):
            fl_slice = np.abs(flux[k])
            fl_sum = fl_slice.sum()
            if fl_sum > 1e-12:
                rsp_per_depth[k] = (rsp[k] * fl_slice).sum() / fl_sum
            else:
                rsp_per_depth[k] = rsp[k].mean()

        # Walk from entrance accumulating WET
        wet = 0.0
        for k in range(n_depth):
            wet += rsp_per_depth[k] * dz
            if wet >= r80_mm:
                overshoot = wet - r80_mm
                step = rsp_per_depth[k] * dz
                frac = overshoot / step if step > 1e-12 else 0.0
                return (float(k) - frac) * dz

        # Beam exits the volume
        return float(n_depth - 1) * dz


# ═══════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════


def denormalize_energy(energy_normalized: float, scale: dict) -> float:
    return (
        energy_normalized * (scale["max_energy"] - scale["min_energy"])
        + scale["min_energy"]
    )


def setup_run_directory(runs_dir: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = runs_dir / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "figures").mkdir(exist_ok=True)
    return run_dir


def setup_logging(run_dir: Path, verbose: bool = False) -> Path:
    log_file = run_dir / "bp_estimation.log"
    log_level = logging.DEBUG if verbose else logging.INFO

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(log_level)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    fmt = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(fmt)
    root_logger.addHandler(console_handler)

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)
    file_handler.setFormatter(fmt)
    root_logger.addHandler(file_handler)

    return log_file


def load_yaml_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


# ═══════════════════════════════════════════════════════════════════════════
#  Plotting
# ═══════════════════════════════════════════════════════════════════════════


def plot_scatter(
    df: pd.DataFrame,
    gt_col: str,
    pred_col: str,
    method_name: str,
    run_dir: Path,
) -> None:
    """Scatter: GT BP depth vs estimated BP depth."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(df[gt_col], df[pred_col], alpha=0.15, s=8, edgecolors="none")
    lims = [
        min(df[gt_col].min(), df[pred_col].min()) - 5,
        max(df[gt_col].max(), df[pred_col].max()) + 5,
    ]
    ax.plot(lims, lims, "k--", lw=0.8, label="identity")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("GT BP depth [mm]")
    ax.set_ylabel(f"{method_name} BP depth [mm]")
    ax.set_title(f"{method_name} vs GT IDD")

    # Correlation text
    valid = df[[gt_col, pred_col]].dropna()
    if len(valid) > 2:
        r, _ = pearsonr(valid[gt_col], valid[pred_col])
        rho, _ = spearmanr(valid[gt_col], valid[pred_col])
        mae = np.mean(np.abs(valid[gt_col] - valid[pred_col]))
        ax.text(
            0.05,
            0.95,
            f"r={r:.3f}  ρ={rho:.3f}\nMAE={mae:.1f} mm\nn={len(valid)}",
            transform=ax.transAxes,
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round", fc="white", alpha=0.8),
        )
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(run_dir / "figures" / f"scatter_{method_name}.png", dpi=150)
    plt.close(fig)


def plot_error_histogram(
    df: pd.DataFrame,
    gt_col: str,
    pred_col: str,
    method_name: str,
    run_dir: Path,
) -> None:
    """Histogram of signed error (estimated − GT)."""
    valid = df[[gt_col, pred_col]].dropna()
    if len(valid) < 2:
        return
    error = valid[pred_col] - valid[gt_col]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(error, bins=60, edgecolor="black", linewidth=0.3, alpha=0.75)
    ax.axvline(0, color="red", ls="--", lw=0.8)
    ax.set_xlabel("Error [mm] (estimated − GT)")
    ax.set_ylabel("Count")
    ax.set_title(f"{method_name} – BP depth error distribution")
    mean_err = error.mean()
    std_err = error.std()
    ax.text(
        0.95,
        0.95,
        f"mean={mean_err:.1f} mm\nstd={std_err:.1f} mm",
        transform=ax.transAxes,
        va="top",
        ha="right",
        fontsize=9,
        bbox=dict(boxstyle="round", fc="white", alpha=0.8),
    )
    fig.tight_layout()
    fig.savefig(run_dir / "figures" / f"error_hist_{method_name}.png", dpi=150)
    plt.close(fig)


def plot_energy_stratified(
    df: pd.DataFrame,
    gt_col: str,
    method_cols: List[str],
    energy_bins: List[float],
    run_dir: Path,
) -> None:
    """Per-energy-bin MAE bar chart for each method."""
    df = df.copy()
    df["energy_bin"] = pd.cut(
        df["energy_mev"],
        bins=energy_bins,
        right=False,
        labels=[
            f"{energy_bins[i]}-{energy_bins[i+1]}" for i in range(len(energy_bins) - 1)
        ],
    )

    rows = []
    for method_col in method_cols:
        method_name = method_col.replace("bp_", "")
        for bin_label, grp in df.groupby("energy_bin", observed=True):
            valid = grp[[gt_col, method_col]].dropna()
            if len(valid) < 2:
                continue
            mae = np.mean(np.abs(valid[gt_col] - valid[method_col]))
            rho, _ = spearmanr(valid[gt_col], valid[method_col])
            rows.append(
                {
                    "method": method_name,
                    "energy_bin": bin_label,
                    "MAE_mm": mae,
                    "spearman_rho": rho,
                    "n": len(valid),
                }
            )

    if not rows:
        return

    strat_df = pd.DataFrame(rows)
    strat_df.to_csv(run_dir / "energy_stratified_summary.csv", index=False)

    # Bar chart – MAE
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=strat_df, x="energy_bin", y="MAE_mm", hue="method", ax=ax)
    ax.set_xlabel("Energy bin [MeV]")
    ax.set_ylabel("MAE [mm]")
    ax.set_title("Energy-stratified BP estimation MAE")
    ax.legend(title="Method", fontsize=8)
    fig.tight_layout()
    fig.savefig(run_dir / "figures" / "energy_stratified_mae.png", dpi=150)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════


@app.command()
def main(
    h5_path: Annotated[
        Optional[Path],
        typer.Argument(help="Path to the HDF5 dataset file"),
    ] = None,
    config: Annotated[
        Optional[Path],
        typer.Option(help="Path to YAML configuration file"),
    ] = None,
    excluded_indexes_file: Annotated[
        Optional[Path],
        typer.Option(help="Path to file listing record IDs to exclude"),
    ] = None,
    max_energy_mev: Annotated[
        Optional[float],
        typer.Option(help="Max initial energy [MeV]"),
    ] = None,
    n_samples: Annotated[
        Optional[int],
        typer.Option(help="Limit to first N samples"),
    ] = None,
    no_progress: Annotated[
        Optional[bool], typer.Option(help="Disable progress bar")
    ] = None,
    verbose: Annotated[
        Optional[bool], typer.Option(help="Enable verbose output")
    ] = None,
) -> None:
    """Estimate Bragg-peak depth using multiple methods and compare."""

    # ── Load & merge config ─────────────────────────────────────────────
    yaml_config: dict = {}
    config_path: Optional[Path] = None
    if config is not None:
        config_path = config if config.is_absolute() else PROJECT_ROOT / config
        yaml_config = load_yaml_config(config_path)

    h5_path = h5_path or (
        Path(yaml_config["h5_path"]) if "h5_path" in yaml_config else None
    )
    excluded_indexes_file = excluded_indexes_file or (
        Path(yaml_config["excluded_indexes_file"])
        if "excluded_indexes_file" in yaml_config
        else None
    )
    max_energy_mev = (
        max_energy_mev
        if max_energy_mev is not None
        else yaml_config.get("max_energy_mev", 250.0)
    )
    n_samples = n_samples if n_samples is not None else yaml_config.get("n_samples")
    no_progress = (
        no_progress
        if no_progress is not None
        else yaml_config.get("no_progress", False)
    )
    verbose = verbose if verbose is not None else yaml_config.get("verbose", False)
    resolution = tuple(yaml_config.get("resolution", [2.0, 2.0, 2.0]))
    energy_bins = yaml_config.get("energy_bins", [70, 100, 130, 160, 190, 220, 250])
    n_diagnostic_figures = yaml_config.get("n_diagnostic_figures", 5)

    if h5_path is None:
        raise typer.BadParameter("H5_PATH is required (CLI or YAML)")

    # ── Setup run directory & logging ───────────────────────────────────
    runs_dir = PROJECT_ROOT / "runs"
    run_dir = setup_run_directory(runs_dir)
    log_file = setup_logging(run_dir, verbose=verbose)

    if config_path is not None:
        shutil.copy2(config_path, run_dir / config_path.name)
    logger.info(f"Run directory: {run_dir}")

    # ── Build estimators ────────────────────────────────────────────────
    method_cfgs = yaml_config.get("methods", [{"name": "gt_idd"}])
    estimators: List[BPEstimator] = []
    for mcfg in method_cfgs:
        est = build_estimator(mcfg, yaml_config)
        estimators.append(est)
        logger.info(f"Registered method: {est.name}")

    # ── Discover samples ────────────────────────────────────────────────
    if not h5_path.is_absolute():
        h5_path = PROJECT_ROOT / h5_path

    excluded_indexes: List[str] = []
    if excluded_indexes_file is not None:
        exc_path = (
            excluded_indexes_file
            if excluded_indexes_file.is_absolute()
            else PROJECT_ROOT / excluded_indexes_file
        )
        if exc_path.exists():
            with open(exc_path, "r") as f:
                excluded_indexes = [line.strip() for line in f if line.strip()]
            logger.info(f"Loaded {len(excluded_indexes)} excluded indexes")

    with h5py.File(h5_path, "r") as ds:
        all_record_ids = list(ds.keys())
    logger.info(f"Total records: {len(all_record_ids)}")

    record_ids = [rid for rid in all_record_ids if rid not in excluded_indexes]
    logger.info(f"After exclusion: {len(record_ids)}")

    if n_samples is not None:
        record_ids = record_ids[:n_samples]
        logger.info(f"Limited to {len(record_ids)} samples")

    # ── Build dataset (no augmentation for evaluation) ──────────────────
    dataset = H5PYGenerator(
        file_path=str(h5_path),
        indexes=record_ids,
        augmentation=False,
        cropp=True,
        normalize=False,
        normalize_flux_only=True,
    )

    scale = DEFAULT_SCALE.copy()

    # ── Column names ────────────────────────────────────────────────────
    method_names = [est.name for est in estimators]
    bp_cols = [f"bp_{name}" for name in method_names]

    # ── Main loop ───────────────────────────────────────────────────────
    rows: List[dict] = []
    skipped = 0
    t0 = perf_counter()

    pbar = tqdm(
        range(len(record_ids)),
        desc="BP estimation",
        disable=no_progress,
    )

    for i in pbar:
        record_id = record_ids[i]
        x, energy, y = dataset[i]

        energy_mev_val = denormalize_energy(energy.item(), scale)

        # Guards
        flux_channel = x[1]
        if torch.abs(flux_channel).max().item() < 1e-9:
            skipped += 1
            continue
        if energy_mev_val > max_energy_mev:
            skipped += 1
            continue

        # De-normalise
        ct_hu = inverse_minmax(x[0].cpu().numpy(), scale["min_ct"], scale["max_ct"])
        flux_np = x[1].cpu().numpy()
        gt_dose_norm = y.cpu().numpy()
        gt_dose = inverse_minmax(
            gt_dose_norm if gt_dose_norm.ndim == 4 else gt_dose_norm[np.newaxis],
            scale["min_ds"],
            scale["max_ds"],
        ).squeeze()

        row = {"sample_id": record_id, "energy_mev": energy_mev_val}

        for est, col in zip(estimators, bp_cols):
            try:
                bp_mm = est.estimate(
                    ct_hu, flux_np, energy_mev_val, gt_dose, resolution
                )
                row[col] = bp_mm
            except Exception as exc:
                logger.warning(f"Method {est.name} failed on {record_id}: {exc}")
                row[col] = float("nan")

        rows.append(row)

        # ── Diagnostic figure for first N samples ───────────────────────
        if len(rows) <= n_diagnostic_figures:
            bp_est_dict = {est.name: row[col] for est, col in zip(estimators, bp_cols)}
            fig_path = run_dir / "figures" / f"bp_diag_{record_id}.png"
            try:
                plot_bp_estimation_diagnostic(
                    ct_hu=ct_hu,
                    gt_dose=gt_dose,
                    sample_id=record_id,
                    energy_mev=energy_mev_val,
                    output_path=fig_path,
                    bp_estimates=bp_est_dict,
                    voxel_spacing_mm=resolution[0],
                )
                logger.debug(f"Diagnostic figure saved: {fig_path}")
            except Exception as exc:
                logger.warning(f"Diagnostic figure failed for {record_id}: {exc}")

    elapsed = perf_counter() - t0
    logger.info(f"Processed {len(rows)} samples in {elapsed:.1f}s (skipped {skipped})")

    # ── Build DataFrame & save CSV ──────────────────────────────────────
    df = pd.DataFrame(rows)
    csv_path = run_dir / "bp_estimation_results.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Results saved to {csv_path}")

    # ── Summary statistics ──────────────────────────────────────────────
    gt_col = "bp_gt_idd"
    if gt_col not in df.columns:
        logger.warning("gt_idd not among methods — skipping comparison plots")
        logger.info("Done.")
        return

    pred_cols = [c for c in bp_cols if c != gt_col and c in df.columns]

    summary_rows = []
    for pred_col in pred_cols:
        method_name = pred_col.replace("bp_", "")
        valid = df[[gt_col, pred_col]].dropna()
        if len(valid) < 3:
            continue
        error = valid[pred_col] - valid[gt_col]
        mae = np.mean(np.abs(error))
        rmse = np.sqrt(np.mean(error**2))
        bias = error.mean()
        r, _ = pearsonr(valid[gt_col], valid[pred_col])
        rho, _ = spearmanr(valid[gt_col], valid[pred_col])
        summary_rows.append(
            {
                "method": method_name,
                "n": len(valid),
                "MAE_mm": round(mae, 2),
                "RMSE_mm": round(rmse, 2),
                "bias_mm": round(bias, 2),
                "pearson_r": round(r, 4),
                "spearman_rho": round(rho, 4),
            }
        )

        # Plots
        plot_scatter(df, gt_col, pred_col, method_name, run_dir)
        plot_error_histogram(df, gt_col, pred_col, method_name, run_dir)

    if summary_rows:
        sum_df = pd.DataFrame(summary_rows)
        sum_df.to_csv(run_dir / "method_summary.csv", index=False)
        logger.info("Method summary:")
        for _, r in sum_df.iterrows():
            logger.info(
                f"  {r['method']:30s}  MAE={r['MAE_mm']:.1f}mm  "
                f"RMSE={r['RMSE_mm']:.1f}mm  bias={r['bias_mm']:.1f}mm  "
                f"r={r['pearson_r']:.3f}  ρ={r['spearman_rho']:.3f}"
            )

    # Energy-stratified analysis
    if pred_cols:
        plot_energy_stratified(df, gt_col, pred_cols, energy_bins, run_dir)

    logger.info(f"All outputs in {run_dir}")
    logger.info("Done.")


if __name__ == "__main__":
    app()
