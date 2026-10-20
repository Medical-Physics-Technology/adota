"""The commissioned pencil-beam model: DoseCUDA's tables, readable and writable.

The on-disk layout is DoseCUDA's own, so a commissioned model also drops into
DoseCUDA's ``lookuptables/protons/<machine>/`` unchanged::

    <model_dir>/
        machine_geometry.csv        VSADX / VSADY (virtual source to isocenter, mm)
        rangeshifter_labels.csv     one entry, "0" -> RS0 (no range shifter)
        HU_RLSP.csv                 written by :func:`src.pencil_beam.rlsp.write_rlsp_csv`
        commissioning.json          provenance (engine, BDL, scanner, phantom, fits)
        RS0/energies.csv            index, energy_label, sigma (in-air sigma at isocenter, mm)
        RS0/energy_NNN.csv          "energy,R80,A,B,C" row, then depth,sigma,idd rows

Per energy the kernel uses ``sigma_total = sigma_air(d) + sigma_mcs(z_w)`` with
``d = distance_to_source - z_w + 0.7 R80`` and ``sigma_air = A d^2 + B d + C``;
``idd`` is the integrated depth dose per unit spot weight. DoseCUDA compiles a
fixed table length of 400 but looks depths up by value, so each energy carries its
own 400-point depth grid (0 .. ~1.3 R80): fine steps at low energy, enough reach
at 270 MeV (R80 ~ 41 cm), and no kernel patch.

DoseCUDA only accepts energies that are in the table. ADoTA's beamlets have
continuous energies, so :meth:`PencilBeamModel.at_energy` builds the tables for any
energy inside the commissioned span by interpolating its two neighbouring nodes in
range-scaled depth (``z / R80``), with R80 interpolated log-log in energy.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import numpy as np

__all__ = ["LUT_LENGTH", "PencilBeamModel", "hu_to_rlsp", "read_rlsp_curve"]

LUT_LENGTH = 400          # compiled into DoseCUDA (IMPTClasses.cuh: LUT_LENGTH)
_RS_FOLDER = "RS0"
_RS_LABEL = "0"


@dataclass
class PencilBeamModel:
    """Per-energy pencil-beam tables plus the machine's virtual source distances.

    Attributes:
        vsadx / vsady: Virtual source (scanning magnet) to isocenter distances, mm.
            Spot ``x`` diverges about ``vsadx``, spot ``y`` about ``vsady``.
        energies: Nominal energies, MeV, ascending, shape ``(n,)``.
        r80: Distal 80% depth in water, mm, ``(n,)``.
        sigma_air_coef: ``(A, B, C)`` of the in-air width quadratic, ``(n, 3)``.
        depths: Water-equivalent depth grid, mm, ``(n, LUT_LENGTH)``.
        sigmas: ``sigma_mcs`` on that grid, mm (``sigma - sigma(0.7 R80)``).
        idds: Integrated depth dose per unit spot weight on that grid.
        reference_sigma: In-air spot sigma at the isocenter, mm (DICOM export only).
        meta: Provenance written to ``commissioning.json``.
    """

    vsadx: float
    vsady: float
    energies: np.ndarray
    r80: np.ndarray
    sigma_air_coef: np.ndarray
    depths: np.ndarray
    sigmas: np.ndarray
    idds: np.ndarray
    reference_sigma: np.ndarray
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        order = np.argsort(self.energies)
        for name in ("energies", "r80", "sigma_air_coef", "depths", "sigmas", "idds",
                     "reference_sigma"):
            setattr(self, name, np.asarray(getattr(self, name), dtype=float)[order])
        if self.depths.shape[1] != LUT_LENGTH:
            raise ValueError(f"depth grid must have {LUT_LENGTH} points, got {self.depths.shape[1]}")

    @property
    def source_distance(self) -> float:
        """Point-source distance the kernel ray-traces from (mean of the two VSADs)."""
        return 0.5 * (self.vsadx + self.vsady)

    def sigma_air(self, index: int, d: np.ndarray) -> np.ndarray:
        """In-air width of node ``index`` at kernel distance ``d`` (mm)."""
        a, b, c = self.sigma_air_coef[index]
        return (a * d + b) * d + c

    # ------------------------------------------------------------------ energies
    def at_energy(self, energy: float, tol: float = 1e-3) -> "PencilBeamModel":
        """A one-energy model for ``energy``: the node itself, or interpolated.

        Raises:
            ValueError: Outside the commissioned span (no extrapolation).
        """
        e = np.asarray(self.energies)
        hit = np.flatnonzero(np.abs(e - energy) <= tol)
        if hit.size:
            return self._subset(int(hit[0]))
        if not e[0] < energy < e[-1]:
            raise ValueError(f"energy {energy} MeV outside the commissioned span [{e[0]}, {e[-1]}]")
        hi = int(np.searchsorted(e, energy))
        lo = hi - 1
        w = (energy - e[lo]) / (e[hi] - e[lo])
        r80 = float(np.exp((1 - w) * np.log(self.r80[lo]) + w * np.log(self.r80[hi])))
        z_max = (1 - w) * self.depths[lo, -1] / self.r80[lo] + w * self.depths[hi, -1] / self.r80[hi]
        depths = np.linspace(0.0, z_max * r80, LUT_LENGTH)

        def blend(table: np.ndarray) -> np.ndarray:
            # sample both nodes at the same reduced depth z / R80
            a = np.interp(depths / r80 * self.r80[lo], self.depths[lo], table[lo])
            b = np.interp(depths / r80 * self.r80[hi], self.depths[hi], table[hi])
            return (1 - w) * a + w * b

        return PencilBeamModel(
            vsadx=self.vsadx, vsady=self.vsady, energies=np.array([energy]), r80=np.array([r80]),
            sigma_air_coef=((1 - w) * self.sigma_air_coef[lo] + w * self.sigma_air_coef[hi])[None],
            depths=depths[None], sigmas=blend(self.sigmas)[None], idds=blend(self.idds)[None],
            reference_sigma=np.array([(1 - w) * self.reference_sigma[lo] + w * self.reference_sigma[hi]]),
            meta={"interpolated_from": [float(e[lo]), float(e[hi])]},
        )

    def _subset(self, index: int) -> "PencilBeamModel":
        sl = slice(index, index + 1)
        return PencilBeamModel(
            vsadx=self.vsadx, vsady=self.vsady, energies=self.energies[sl], r80=self.r80[sl],
            sigma_air_coef=self.sigma_air_coef[sl], depths=self.depths[sl], sigmas=self.sigmas[sl],
            idds=self.idds[sl], reference_sigma=self.reference_sigma[sl], meta={})

    def kernel_model(self) -> SimpleNamespace:
        """The object ``dose_kernels.proton_*_cuda`` reads its model from.

        Row order is energy id; ``divergence_params`` rows are
        ``[energy, R80, A, B, C]`` (the kernel's ``dvp_len = 5``).
        """
        f32 = np.float32
        dvp = np.column_stack([self.energies, self.r80, self.sigma_air_coef])
        return SimpleNamespace(
            VSADX=float(self.vsadx), VSADY=float(self.vsady),
            divergence_params=np.ascontiguousarray(dvp, dtype=f32),
            lut_depths=np.ascontiguousarray(self.depths, dtype=f32),
            lut_sigmas=np.ascontiguousarray(self.sigmas, dtype=f32),
            lut_idds=np.ascontiguousarray(self.idds, dtype=f32),
        )

    # ------------------------------------------------------------------ disk I/O
    def write(self, model_dir: Path) -> Path:
        """Write the model in DoseCUDA's layout (``HU_RLSP.csv`` is written separately)."""
        model_dir = Path(model_dir)
        rs = model_dir / _RS_FOLDER
        rs.mkdir(parents=True, exist_ok=True)
        (model_dir / "machine_geometry.csv").write_text(f"VSADX,{self.vsadx}\nVSADY,{self.vsady}\n")
        (model_dir / "rangeshifter_labels.csv").write_text(
            f"dicom_rangeshifter_label,folder_rangeshifter_label\n{_RS_LABEL},{_RS_FOLDER}\n")
        rows = ["index,energy_label,sigma"] + [
            f"{i},{e:.2f},{s:.4f}" for i, (e, s) in enumerate(zip(self.energies, self.reference_sigma))]
        (rs / "energies.csv").write_text("\n".join(rows) + "\n")
        for i in range(len(self.energies)):
            a, b, c = self.sigma_air_coef[i]
            lines = ["energy, R80,A,B,C", f"{self.energies[i]:.2f},{self.r80[i]:.4f},{a:.6e},{b:.6e},{c:.6e}",
                     "", "depth,sigma,idd"]
            lines += [f"{d:.4f},{s:.5f},{v:.6e}"
                      for d, s, v in zip(self.depths[i], self.sigmas[i], self.idds[i])]
            (rs / f"energy_{i:03d}.csv").write_text("\n".join(lines) + "\n")
        (model_dir / "commissioning.json").write_text(json.dumps(self.meta, indent=2, default=float) + "\n")
        return model_dir

    @classmethod
    def read(cls, model_dir: Path) -> "PencilBeamModel":
        """Read a model written by :meth:`write` (or any DoseCUDA RS0 model)."""
        model_dir = Path(model_dir)
        geom = dict(line.split(",") for line in
                    (model_dir / "machine_geometry.csv").read_text().split() if "," in line)
        table = np.genfromtxt(model_dir / _RS_FOLDER / "energies.csv", delimiter=",", names=True)
        table = np.atleast_1d(table)
        energies, r80, coef, depths, sigmas, idds = [], [], [], [], [], []
        for index in table["index"].astype(int):
            lines = (model_dir / _RS_FOLDER / f"energy_{index:03d}.csv").read_text().splitlines()
            head = [float(v) for v in lines[1].split(",")]
            body = np.array([[float(v) for v in ln.split(",")] for ln in lines[4:4 + LUT_LENGTH] if ln])
            energies.append(head[0])
            r80.append(head[1])
            coef.append(head[2:5])
            depths.append(body[:, 0])
            sigmas.append(body[:, 1])
            idds.append(body[:, 2])
        meta_path = model_dir / "commissioning.json"
        return cls(
            vsadx=float(geom["VSADX"]), vsady=float(geom["VSADY"]), energies=np.array(energies),
            r80=np.array(r80), sigma_air_coef=np.array(coef), depths=np.array(depths),
            sigmas=np.array(sigmas), idds=np.array(idds), reference_sigma=np.asarray(table["sigma"]),
            meta=json.loads(meta_path.read_text()) if meta_path.exists() else {},
        )


def read_rlsp_curve(model_dir: Path) -> tuple:
    """``(hu, rlsp)`` from a model directory's ``HU_RLSP.csv``."""
    data = np.loadtxt(Path(model_dir) / "HU_RLSP.csv", delimiter=",", skiprows=1)
    return data[:, 0], data[:, 1]


def hu_to_rlsp(hu_array: np.ndarray, curve: tuple, dtype: Optional[type] = np.float32) -> np.ndarray:
    """Map an HU volume through an ``(hu, rlsp)`` curve (clamped at the ends, as DoseCUDA)."""
    return np.interp(hu_array, curve[0], curve[1]).astype(dtype)
