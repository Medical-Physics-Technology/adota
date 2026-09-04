"""Deviation-ladder harness for the plan-level gamma corpus.

The GPU gamma work is validated against eight OpenTPS plan directories that each
carry a ``gamma_metrics.json`` recording the pass rates a previous CPU run
produced. Those recordings are **rung 0** of a four-rung ladder; every later rung
is measured against the one above it so a moved pass rate can be attributed:

===== ============================================================ ================
Rung   What it is                                                   Isolates
===== ============================================================ ================
0      the recorded ``gamma_metrics.json`` (pymedphys 0.40, py3.9)  --
1      a CPU re-run on this machine at the current pymedphys        the interpolator
2      ``gamma_torch`` on torch-CPU, float64                        the kernel
3      ``gamma_torch`` on one GPU, float64                          the device
4      ``gamma_torch`` on one GPU, float32                          precision
===== ============================================================ ================

Two rules shape this module:

* **The recipe comes from each plan's own JSON, not from adota's defaults.**
  ``gamma_params_base`` and ``criteria`` are read back from the recording;
  :data:`src.adota.config.DEFAULT_GAMMA_PARAMS` has drifted since
  (``interp_fraction`` is 10 there and was 5 in the recorded runs).
* **MCsquare is the reference grid and ADoTA the evaluation grid.** Gamma is not
  symmetric. ``scripts/run_plan_opentps.py`` calls ``plan_gamma(dose_adota,
  dose_mc)``, whose signature is ``plan_gamma(dose_eval, dose_ref)``; this module
  goes through :func:`src.metrics.plan_gamma.plan_gamma` rather than re-deriving
  the direction.

The corpus lives outside the repository, at ``$ADOTA_GAMMA_CORPUS`` (default
``/scratch/mstryja/opentps_plans``), following the ``$ADOTA_GOLDEN_DIR`` pattern.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import socket
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import SimpleITK as sitk

from src.metrics.plan_gamma import criterion_label, parse_criteria

logger = logging.getLogger(__name__)

__all__ = [
    "BENCHMARK_PLANS",
    "VOXEL_PARITY_PLANS",
    "DEFAULT_CORPUS_DIR",
    "PlanCase",
    "corpus_dir",
    "corpus_skip_reason",
    "load_plan_case",
    "environment_stamp",
    "write_rung_json",
    "voxel_parity",
]

DEFAULT_CORPUS_DIR = Path(
    os.environ.get("ADOTA_GAMMA_CORPUS", "/scratch/mstryja/opentps_plans")
)

# The eight plans the GPU gamma brief validates against, in the order its table
# lists them. Every one carries a gamma_metrics.json from a prior CPU run.
BENCHMARK_PLANS: Tuple[str, ...] = (
    "LUNG1-062_Publication_Plan_1",
    "LUNG1-195_Publication_Plan_2",
    "LUNG1-250_Publication_Plan_3",
    "LUNG1-364_Publication_Plan_5",
    "Prostate-AEC-004_Publication_Plan_1",
    "Prostate-AEC-069_Publication_Plan_2",
    "Prostate-AEC-006_Publication_Plan_3",
    "Prostate-AEC-007_Publication_Plan_4",
)

# Persisting gamma maps costs ~2 GB per plan (5 criteria x ~400 MB), so
# voxel-level parity is only checked on the fastest plan and one prostate case.
VOXEL_PARITY_PLANS: Tuple[str, ...] = (
    "LUNG1-195_Publication_Plan_2",
    "Prostate-AEC-007_Publication_Plan_4",
)


def corpus_dir() -> Path:
    """Root of the plan corpus (``$ADOTA_GAMMA_CORPUS``)."""
    return DEFAULT_CORPUS_DIR


def corpus_skip_reason(plans: Sequence[str] = BENCHMARK_PLANS) -> Optional[str]:
    """Return a skip message naming what is missing, or ``None`` when usable.

    A test that needs the corpus must skip rather than fail, so this names both
    the environment variable and the files it looked for.
    """
    root = corpus_dir()
    if not root.is_dir():
        return (
            f"Gamma plan corpus not found at {root}. Set $ADOTA_GAMMA_CORPUS to a "
            "directory of OpenTPS plan directories (each with CT.mhd, "
            "PlanPencil.txt, config.txt, bdl.txt, Dose.mhd, Dose_ADoTA.mhd and "
            "gamma_metrics.json)."
        )
    missing = [
        name
        for name in plans
        if not (root / name / "gamma_metrics.json").is_file()
        or not (root / name / "Dose_ADoTA.mhd").is_file()
        or not (root / name / "Dose.mhd").is_file()
    ]
    if missing:
        return (
            f"Gamma plan corpus at {root} is missing Dose.mhd / Dose_ADoTA.mhd / "
            f"gamma_metrics.json for: {', '.join(missing)}."
        )
    return None


@dataclass
class PlanCase:
    """One plan's doses plus the recipe and results recorded for it.

    Attributes:
        name: Plan directory name.
        plan_dir: Path to the plan directory.
        dose_eval: ADoTA dose in Gy, ``(z, y, x)`` -- the *evaluation* grid.
        dose_ref: MCsquare dose in Gy, same shape -- the *reference* grid.
        spacing_zyx: Voxel spacing in mm, ``(z, y, x)``.
        criteria: ``(dose%, distance_mm, cutoff%)`` tuples read from the JSON.
        gamma_params_base: The recorded ``gamma_params_base`` (interp_fraction,
            max_gamma, local_gamma, random_subset).
        recorded: ``{criterion label: recorded pass_rate_pct}`` -- rung 0.
        recorded_elapsed_s: The JSON's ``elapsed_s`` (provenance, not a
            controlled measurement: another machine, another pymedphys).
    """

    name: str
    plan_dir: Path
    dose_eval: np.ndarray
    dose_ref: np.ndarray
    spacing_zyx: Tuple[float, float, float]
    criteria: List[Tuple[float, float, float]]
    gamma_params_base: dict
    recorded: Dict[str, float]
    recorded_elapsed_s: Optional[float]

    @property
    def n_voxels(self) -> int:
        """Total voxel count of the (shared) dose grid."""
        return int(np.prod(self.dose_ref.shape))


def load_plan_case(plan_dir: Path) -> PlanCase:
    """Load a plan's ADoTA and MCsquare doses in Gy plus its recorded recipe.

    Doses go through :func:`src.beamlets.dose_scaling.load_dose_gy`, which applies
    the MU->Gy factor; reading the ``.raw`` files directly would give plausible
    but unscaled numbers.

    Args:
        plan_dir: An OpenTPS plan directory holding ``gamma_metrics.json``.

    Returns:
        The populated :class:`PlanCase`.

    Raises:
        FileNotFoundError: If the directory, its ``gamma_metrics.json`` or either
            dose file is missing.
    """
    # Imported here so the module can be imported (for its constants and the
    # skip reason) without pulling in the plan loader stack.
    from src.beamlets.bdl import BeamDataLibrary
    from src.beamlets.dose_scaling import load_dose_gy
    from src.loaders.plan_directory import load_plan_directory

    plan_dir = Path(plan_dir)
    metrics_path = plan_dir / "gamma_metrics.json"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"No gamma_metrics.json in {plan_dir}")

    recorded_json = json.loads(metrics_path.read_text())
    criteria = parse_criteria(
        [entry["criterion"] for entry in recorded_json.get("criteria", [])]
    )
    recorded = {
        criterion_label(tuple(entry["criterion"])): float(entry["pass_rate_pct"])
        for entry in recorded_json.get("criteria", [])
    }

    plan_directory = load_plan_directory(plan_dir)
    if plan_directory.mc_dose_path is None:
        raise FileNotFoundError(f"No MCsquare Dose.mhd in {plan_dir}")
    adota_dose_path = plan_dir / "Dose_ADoTA.mhd"
    if not adota_dose_path.is_file():
        raise FileNotFoundError(f"No Dose_ADoTA.mhd in {plan_dir}")

    bdl = BeamDataLibrary.from_file(plan_directory.bdl_path)
    dose_eval = sitk.GetArrayFromImage(
        load_dose_gy(adota_dose_path, plan_directory.plan, bdl)
    )
    dose_ref = sitk.GetArrayFromImage(
        load_dose_gy(plan_directory.mc_dose_path, plan_directory.plan, bdl)
    )
    spacing_zyx = tuple(float(s) for s in plan_directory.ct.GetSpacing()[::-1])

    return PlanCase(
        name=plan_dir.name,
        plan_dir=plan_dir,
        dose_eval=dose_eval,
        dose_ref=dose_ref,
        spacing_zyx=spacing_zyx,
        criteria=criteria,
        gamma_params_base=dict(recorded_json.get("gamma_params_base", {})),
        recorded=recorded,
        recorded_elapsed_s=recorded_json.get("elapsed_s"),
    )


def environment_stamp(device: str, dtype: Optional[str]) -> dict:
    """Provenance for a rung: host, interpreter, library versions, device."""
    import pymedphys
    import torch

    stamp = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "host": socket.gethostname(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pymedphys": pymedphys.__version__,
        "torch": torch.__version__,
        "device": device,
        "dtype": dtype,
    }
    if device.startswith("cuda") and torch.cuda.is_available():
        index = int(device.split(":")[1]) if ":" in device else 0
        stamp["gpu_name"] = torch.cuda.get_device_name(index)
    return stamp


def write_rung_json(
    path: Path,
    rung: str,
    backend: str,
    device: str,
    dtype: Optional[str],
    plans: Dict[str, dict],
) -> Path:
    """Write one rung's results as JSON and return the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "rung": rung,
        "backend": backend,
        "environment": environment_stamp(device, dtype),
        "plans": plans,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")
    logger.info("Wrote %s", path)
    return path


def voxel_parity(reference_map: np.ndarray, other_map: np.ndarray) -> dict:
    """Compare two gamma maps voxel by voxel.

    Pass rates can agree while the maps differ, so the number that matters is the
    fraction of evaluated voxels that cross the gamma = 1 decision boundary: a
    voxel moving 1.5 -> 1.6 is irrelevant, one moving 0.999 -> 1.001 is not.

    A voxel counts as *not evaluated* when its gamma is NaN or exactly 0. Both
    forms occur: the kernel returns NaN, and the maps the pipeline hands on have
    been through ``np.nan_to_num``, which turns those NaNs into zeros. The
    pass-rate definition already treats a zero as not evaluated, so the two are
    the same category here and disagreements about it are counted together.

    Args:
        reference_map: Baseline gamma map.
        other_map: Comparison gamma map, same shape.

    Returns:
        ``{"n_evaluated", "max_abs_delta", "p999_abs_delta", "boundary_cross_frac",
        "n_boundary_cross", "n_evaluated_disagree"}``.

    Raises:
        ValueError: If the shapes differ.
    """
    if reference_map.shape != other_map.shape:
        raise ValueError(
            f"gamma maps must share shape, got {reference_map.shape} vs "
            f"{other_map.shape}"
        )
    flat_a = reference_map.astype(np.float64).ravel()
    flat_b = other_map.astype(np.float64).ravel()

    unevaluated_a = np.isnan(flat_a) | (flat_a == 0)
    unevaluated_b = np.isnan(flat_b) | (flat_b == 0)
    n_evaluated_disagree = int(np.count_nonzero(unevaluated_a ^ unevaluated_b))

    both = ~unevaluated_a & ~unevaluated_b
    n_evaluated = int(np.count_nonzero(both))
    if n_evaluated == 0:
        return {
            "n_evaluated": 0,
            "max_abs_delta": 0.0,
            "p999_abs_delta": 0.0,
            "boundary_cross_frac": 0.0,
            "n_boundary_cross": 0,
            "n_evaluated_disagree": n_evaluated_disagree,
        }

    delta = np.abs(flat_a[both] - flat_b[both])
    crossings = int(np.count_nonzero((flat_a[both] > 1.0) != (flat_b[both] > 1.0)))
    return {
        "n_evaluated": n_evaluated,
        "max_abs_delta": float(delta.max()),
        "p999_abs_delta": float(np.percentile(delta, 99.9)),
        "boundary_cross_frac": crossings / n_evaluated,
        "n_boundary_cross": crossings,
        "n_evaluated_disagree": n_evaluated_disagree,
    }
