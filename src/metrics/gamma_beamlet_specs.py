# Copyright (C) 2026 Medical Physics & Technology
#
# Licensed under the MIT License. See the repository LICENSE file.

"""What the gamma benchmarks measure: the rungs and the criteria.

Split from :mod:`src.metrics.gamma_beamlet_benchmark` by role. A rung is one
backend, device and precision; a case is one gamma recipe. Both are plain
frozen dataclasses so that a run's settings can be written into its results
verbatim and compared across runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

__all__ = ["RungSpec", "RUNGS", "GammaCase", "criterion_label", "default_cases"]


@dataclass(frozen=True)
class RungSpec:
    """One backend / device / precision combination under test.

    Attributes:
        rung: Ladder position, matching :mod:`src.metrics.gamma_benchmark`.
        backend: ``"pymedphys"`` or ``"torch"``.
        device: Device string for the torch backend; ``None`` for pymedphys.
        dtype: ``"float32"`` or ``"float64"`` for the torch backend.
    """

    rung: int
    backend: str
    device: Optional[str] = None
    dtype: Optional[str] = None

    @property
    def label(self) -> str:
        """Short human-readable name, used as a table column."""
        if self.backend == "pymedphys":
            return "pymedphys cpu"
        return f"torch {self.device} {self.dtype}"

    def resolve_device(self, device: Optional[str]) -> Optional[str]:
        """The device this rung runs on, given the CLI's ``--device``.

        The override applies only to the GPU rungs. Rung 2 is the torch-CPU
        rung by definition, so ``--device cuda:0`` must not silently move it
        onto the GPU and make rungs 2 and 3 the same measurement.
        """
        if self.device is None:
            return None
        if self.device.startswith("cuda") and device:
            return device
        return self.device

    def backend_options(self, device: Optional[str] = None) -> Optional[Dict[str, str]]:
        """Options dict for :func:`src.metrics.gamma_pass_rate.gamma_index`."""
        if self.backend == "pymedphys":
            return None
        return {"device": self.resolve_device(device), "dtype": self.dtype}


# The rungs, keyed by the name the CLI accepts. ``cuda`` is a placeholder that
# the CLI's --device argument replaces, so the same table can be produced on any
# GPU index without editing the specs.
RUNGS: Dict[str, RungSpec] = {
    "rung1": RungSpec(1, "pymedphys"),
    "rung2": RungSpec(2, "torch", "cpu", "float64"),
    "rung3": RungSpec(3, "torch", "cuda", "float64"),
    "rung4": RungSpec(4, "torch", "cuda", "float32"),
}


@dataclass(frozen=True)
class GammaCase:
    """One gamma recipe: a criterion plus the search parameters.

    Attributes:
        dose_percent_threshold: Dose-difference criterion, in percent.
        distance_mm_threshold: Distance-to-agreement criterion, in millimetres.
        lower_percent_dose_cutoff: Percent of the normalisation below which
            gamma is not evaluated.
        interp_fraction: Steps the distance threshold is divided into.
        max_gamma: Largest gamma searched for.
    """

    dose_percent_threshold: float
    distance_mm_threshold: float
    lower_percent_dose_cutoff: float = 10.0
    interp_fraction: int = 10
    max_gamma: float = 2.0

    def as_params(self) -> Dict[str, Any]:
        """The dict both backends take, in ``DEFAULT_GAMMA_PARAMS`` form."""
        return {
            "dose_percent_threshold": self.dose_percent_threshold,
            "distance_mm_threshold": self.distance_mm_threshold,
            "interp_fraction": self.interp_fraction,
            "max_gamma": self.max_gamma,
            "lower_percent_dose_cutoff": self.lower_percent_dose_cutoff,
            "random_subset": None,
            "local_gamma": False,
            "quiet": True,
        }


def criterion_label(case: GammaCase) -> str:
    """``"3%/3mm/10%"``-style label, matching the plan-level tables."""
    return (
        f"{case.dose_percent_threshold:g}%/{case.distance_mm_threshold:g}mm/"
        f"{case.lower_percent_dose_cutoff:g}%"
    )


def default_cases() -> Tuple[GammaCase, ...]:
    """The three headline criteria, at the repository's default search settings.

    3%/3mm is the reported headline; 2%/2mm is what the training configs use;
    1%/1mm is the tightest criterion, and the most expensive, because a tighter
    distance threshold makes the search radius grow in smaller steps.
    """
    return (
        GammaCase(1.0, 1.0),
        GammaCase(2.0, 2.0),
        GammaCase(3.0, 3.0),
    )
