"""Single-beamlet MCsquare runner (vendored, self-contained).

Mirrors datagenerator's ``DoseMCSquare.run_sitk`` without any nvidia.dali /
totalsegmentator dependency. Each call runs in a fresh per-run working directory
on scratch (never in the repo): engine data (Materials/Scanners/BDL) is symlinked
in from the independent install (see docs/mcsquare_engine.md), the CT / PlanPencil
/ config are written there, the binary is executed with that dir as CWD, and the
resulting ``Outputs/Dose.mhd`` is read back. The gantry angle is carried by the
PlanPencil (MCsquare rotates internally), matching the training pipeline.
"""
from __future__ import annotations

import shutil
import subprocess
import uuid
from pathlib import Path
from time import perf_counter
from typing import Iterator, Optional, Sequence, Tuple

import numpy as np
import SimpleITK as sitk

from src.mc_generation.config_writer import (
    build_simulation_config,
    write_config,
    write_plan_beamlets,
    write_plan_pencil,
)


def _mean_stat_uncertainty(outputs_dir: Path) -> float:
    """Mean statistical uncertainty (%) parsed from Simulation_progress.txt."""
    prog = outputs_dir / "Simulation_progress.txt"
    if not prog.exists():
        return float("nan")
    vals = []
    for line in prog.read_text().splitlines():
        if line.startswith(" 100.0 %"):
            parts = line.split(" ")
            try:
                vals.append(float(parts[-3]))
            except (ValueError, IndexError):
                pass
    return float(np.mean(vals)) if vals else float("nan")


class MCSquareRunner:
    """Runs one proton beamlet through MCsquare and returns the dose."""

    def __init__(
        self,
        install_dir: str,
        work_root: str,
        bdl_file: str = "hptc_beam_model_rsnone.txt",
        scanner: str = "default",
    ):
        self.install_dir = Path(install_dir)
        self.binary = self.install_dir / "MCsquare_linux"
        self.work_root = Path(work_root)
        self.bdl_file = bdl_file
        self.scanner = scanner
        if not self.binary.exists():
            raise FileNotFoundError(f"MCsquare binary not found: {self.binary}")
        for sub in ("Materials", "Scanners", "BDL"):
            if not (self.install_dir / sub).exists():
                raise FileNotFoundError(f"MCsquare install missing {sub}/ under {self.install_dir}")
        if not (self.install_dir / "BDL" / self.bdl_file).exists():
            raise FileNotFoundError(f"BDL file not found: {self.install_dir/'BDL'/self.bdl_file}")

    @staticmethod
    def default_isocenter(ct: sitk.Image) -> list:
        """Center-of-CT isocenter (physical mm), matching datagenerator's convention."""
        size = np.asarray(ct.GetSize())
        spacing = np.asarray(ct.GetSpacing())
        return (size * spacing // 2).tolist()

    def _make_workdir(self) -> Path:
        wd = self.work_root / f"beamlet_{uuid.uuid4().hex[:12]}"
        wd.mkdir(parents=True, exist_ok=False)
        for sub in ("Materials", "Scanners", "BDL"):
            (wd / sub).symlink_to(self.install_dir / sub)
        return wd

    def run_beamlet_field(
        self,
        ct: sitk.Image,
        energy: float,
        gantry_angle: float,
        spots_xy: Sequence[Sequence[float]],
        isocenter: Optional[Sequence[float]] = None,
        num_primaries: float = 1e7,
        num_threads: int = 0,
        rng_seed: int = 0,
        weight: float = 1000.0,
        keep_workdir: bool = False,
    ) -> Iterator[Tuple[int, sitk.Image, dict]]:
        """Run every spot of one field in a single MCsquare beamlet-mode call.

        Beamlet mode pays the per-run cost -- CT read, HU->material conversion,
        scoring allocation, and the CT write on our side -- once for the whole
        field instead of once per spot, and ``Beamlet_Parallelization`` gives one
        OpenMP thread per spot with a private scoring array (dynamic schedule, so
        spots self-balance) rather than splitting threads inside one beamlet.
        ``Num_Primaries`` applies **per spot**, so each beamlet's statistics and
        dose scale match :meth:`run_beamlet` exactly.

        Yields ``(spot_index, dose_image, sim_res)`` in plan order, deleting each
        dose file as it is yielded: a field's dense dose grids run to tens of GB.
        The working directory is removed when the generator is exhausted or closed,
        so consume it fully (or close it) rather than abandoning it.

        Note: MCsquare writes ``Simulation_progress.txt`` only outside beamlet
        mode, so ``sim_res["stat_uncertainty"]`` is NaN here.
        """
        iso = list(map(float, isocenter)) if isocenter is not None else self.default_isocenter(ct)
        spots = [(float(x), float(y)) for x, y in spots_xy]
        wd = self._make_workdir()
        try:
            sitk.WriteImage(ct, str(wd / "CT.mhd"))
            write_plan_beamlets(
                wd / "PlanPencil.txt", energy=float(energy), spots_xy=spots,
                gantry_angle=float(gantry_angle), isocenter=iso, weight=float(weight))
            cfg = build_simulation_config(
                ct_file="CT.mhd", pencil_plan_path="PlanPencil.txt",
                bdl_file_path=self.bdl_file, output_dir="Outputs",
                sim_params={
                    "Num_Primaries": num_primaries, "Num_Threads": num_threads,
                    "RNG_Seed": rng_seed,
                },
                scanner=self.scanner, compute_uncertainty=False,
                beamlet_mode=True, beamlet_parallelization=True,
            )
            write_config(wd / "config.txt", cfg)

            t0 = perf_counter()
            proc = subprocess.run(
                [str(self.binary), "config.txt"], cwd=str(wd),
                capture_output=True, text=True,
            )
            elapsed = perf_counter() - t0
            if proc.returncode != 0:
                raise RuntimeError(
                    f"MCsquare beamlet-mode run failed (code {proc.returncode}).\n"
                    f"stdout tail:\n{proc.stdout[-2000:]}\nstderr tail:\n{proc.stderr[-1000:]}"
                )

            outputs = wd / "Outputs"
            for index, spot in enumerate(spots):
                dose_path = outputs / f"Dose_Beamlet_0_0_{index}.mhd"
                if not dose_path.exists():
                    raise RuntimeError(
                        f"beamlet-mode run produced no {dose_path.name}; "
                        f"{len(list(outputs.glob('Dose_Beamlet_*.mhd')))} of "
                        f"{len(spots)} spots present")
                dose = sitk.ReadImage(str(dose_path))
                sim_res = {
                    "simulation_log": {
                        "energy": [float(energy)],
                        "isocenter": iso,
                        "bixelgrid_shifts_xy": [spot],
                        "gantry_angle": float(gantry_angle),
                    },
                    "initial_energy": float(energy),
                    "gantry_angle": float(gantry_angle),
                    "num_primaries": float(num_primaries),
                    "rng_seed": int(rng_seed),
                    "stat_uncertainty": float("nan"),   # not reported in beamlet mode
                    "mc_seconds": elapsed / len(spots),  # field cost shared over its spots
                    "mc_field_seconds": elapsed,
                    "mc_beamlet_mode": True,
                    "workdir": str(wd) if keep_workdir else None,
                }
                yield index, dose, sim_res
                if not keep_workdir:
                    for suffix in (".mhd", ".raw", ".zraw"):
                        dose_path.with_suffix(suffix).unlink(missing_ok=True)
        finally:
            if not keep_workdir:
                shutil.rmtree(wd, ignore_errors=True)

    def run_beamlet(
        self,
        ct: sitk.Image,
        energy: float,
        gantry_angle: float,
        spot_xy: Sequence[float],
        isocenter: Optional[Sequence[float]] = None,
        num_primaries: float = 1e7,
        num_threads: int = 0,
        rng_seed: int = 0,
        weight: float = 1000.0,
        keep_workdir: bool = False,
    ) -> tuple[sitk.Image, dict]:
        """Run one beamlet; return ``(dose_sitk, sim_res)``.

        ``sim_res`` carries the plan params (energy, isocenter, spot shift, gantry),
        the mean statistical uncertainty, and timing. The dose grid matches the CT.
        """
        iso = list(map(float, isocenter)) if isocenter is not None else self.default_isocenter(ct)
        wd = self._make_workdir()
        try:
            sitk.WriteImage(ct, str(wd / "CT.mhd"))
            write_plan_pencil(
                wd / "PlanPencil.txt", energy=float(energy),
                spot_xy=(float(spot_xy[0]), float(spot_xy[1])),
                gantry_angle=float(gantry_angle), isocenter=iso, weight=float(weight),
            )
            cfg = build_simulation_config(
                ct_file="CT.mhd", pencil_plan_path="PlanPencil.txt",
                bdl_file_path=self.bdl_file, output_dir="Outputs",
                sim_params={
                    "Num_Primaries": num_primaries, "Num_Threads": num_threads,
                    "RNG_Seed": rng_seed,
                },
                scanner=self.scanner, compute_uncertainty=True,
            )
            write_config(wd / "config.txt", cfg)

            t0 = perf_counter()
            proc = subprocess.run(
                [str(self.binary), "config.txt"], cwd=str(wd),
                capture_output=True, text=True,
            )
            elapsed = perf_counter() - t0
            dose_path = wd / "Outputs" / "Dose.mhd"
            if proc.returncode != 0 or not dose_path.exists():
                raise RuntimeError(
                    f"MCsquare failed (code {proc.returncode}); no {dose_path}.\n"
                    f"stdout tail:\n{proc.stdout[-2000:]}\nstderr tail:\n{proc.stderr[-1000:]}"
                )
            dose = sitk.ReadImage(str(dose_path))
            sim_res = {
                "simulation_log": {
                    "energy": [float(energy)],
                    "isocenter": iso,
                    "bixelgrid_shifts_xy": [(float(spot_xy[0]), float(spot_xy[1]))],
                    "gantry_angle": float(gantry_angle),
                },
                "initial_energy": float(energy),
                "gantry_angle": float(gantry_angle),
                "num_primaries": float(num_primaries),
                "rng_seed": int(rng_seed),
                "stat_uncertainty": _mean_stat_uncertainty(wd / "Outputs"),
                "mc_seconds": elapsed,
                "workdir": str(wd) if keep_workdir else None,
            }
            return dose, sim_res
        finally:
            if not keep_workdir:
                shutil.rmtree(wd, ignore_errors=True)
