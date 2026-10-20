"""MC vs pencil beam vs ADoTA on beamlet directories that were already simulated.

The MC generator (:mod:`src.mc_generation.robustness`) saved, per beamlet, the
ADoTA crop of the field's canonical CT and of the MC dose (``{id}_ct.npy``,
``{id}_ds.npy``, ``{id}_flux.npy``, ``{id}_sim_res.json``), but not the full CT the
pencil beam must ray-trace through. The full field is therefore rebuilt from the
source DICOM with the same code and generation config
(:func:`~src.mc_generation.robustness.field_geometry`), and **checked against the
saved crops** (it must reproduce ``{id}_ct.npy`` exactly) before any dose is
trusted. The pencil beam then runs on that grid, once per field for the ray trace,
and per beamlet only inside its crop window, so ``{id}_ds_pba.npy`` has exactly the
layout of ``{id}_ds.npy``.

Source directories are **never written to**. Each one is mirrored into the output
root as symlinks to its inputs; every new file (``_ds_pba.npy``, a fresh
``_ds_pred.npy`` from :func:`~src.beamlets.inference.run_inference`, metrics)
lives in the mirror.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Sequence, Tuple

import numpy as np

from src.beamlets.cropping import extract_beamlet_roi
from src.pencil_beam.metrics import GammaCriterion, dose_agreement, dose_bbox, gamma_pass_rates

logger = logging.getLogger(__name__)

__all__ = [
    "INPUT_SUFFIXES",
    "SourceDir",
    "compute_field_pba",
    "load_triplet",
    "mirror_inputs",
    "rank_beamlets",
    "score_beamlet",
    "survey_dirs",
]

INPUT_SUFFIXES = ("_ct.npy", "_ds.npy", "_flux.npy", "_sim_res.json")
_STEM = re.compile(r"^(a\d{2}_\d{2})_sim_res\.json$")


@dataclass(frozen=True)
class SourceDir:
    """One saved (patient, energy, field) block of beamlets."""

    path: Path
    stems: Tuple[str, ...]
    lattice_n: int            # 1 + the largest grid index (18 for an 18 x 18 sweep)
    anatomy: str
    patient_id: str
    provenance_uid: str
    energy: float
    gantry: float

    @property
    def name(self) -> str:
        return self.path.name


def survey_dirs(paths: Sequence[Path], grid_n: int = 18, min_beamlets: int = 200
                ) -> Tuple[List[SourceDir], List[Tuple[Path, str]]]:
    """Split candidate directories into full-lattice blocks and excluded ones (with a reason).

    A directory qualifies when its grid indices span the ``grid_n x grid_n`` lattice
    and it holds at least ``min_beamlets`` beamlets: the 18 x 18 sweeps lose a few
    beamlets to the QA gates, the smoke directories hold four or five.
    """
    selected, excluded = [], []
    for path in sorted(Path(p) for p in paths):
        stems = sorted(m.group(1) for f in path.iterdir() if (m := _STEM.match(f.name)))
        if not stems:
            excluded.append((path, "no beamlets"))
            continue
        first = json.loads((path / f"{stems[0]}_sim_res.json").read_text())
        idx = np.array([[int(s[1:3]), int(s[4:6])] for s in stems])
        lattice = int(idx.max()) + 1
        if lattice != grid_n:
            excluded.append((path, f"lattice {lattice} x {lattice}, not {grid_n} x {grid_n}"))
        elif len(stems) < min_beamlets:
            excluded.append((path, f"{len(stems)} beamlets < {min_beamlets}"))
        else:
            selected.append(SourceDir(
                path=path, stems=tuple(stems), lattice_n=lattice, anatomy=first["anatomy"],
                patient_id=first["patient_id"], provenance_uid=first["provenance_uid"],
                energy=float(first["initial_energy"]), gantry=float(first["gantry_angle"])))
    return selected, excluded


def mirror_inputs(src: SourceDir, out_dir: Path) -> Path:
    """Symlink a source block's inputs into ``out_dir`` (the source is never written)."""
    out_dir = Path(out_dir).resolve()
    src_path = src.path.resolve()
    if out_dir == src_path or src_path in out_dir.parents:
        raise ValueError(f"output {out_dir} must not lie inside the source {src_path}")
    out_dir.mkdir(parents=True, exist_ok=True)
    for stem in src.stems:
        for suffix in INPUT_SUFFIXES:
            link = out_dir / f"{stem}{suffix}"
            if not link.is_symlink():
                link.symlink_to(src_path / f"{stem}{suffix}")
    return out_dir


def compute_field_pba(engine, geom, bdl, roi_size, src: SourceDir, out_dir: Path, overwrite: bool = False,
                      verify_ct: bool = True) -> dict:
    """Pencil-beam dose of every beamlet of one block, written as ``{stem}_ds_pba.npy``.

    ``geom`` is the rebuilt :class:`~src.mc_generation.robustness.FieldGeometry`.
    Each beamlet's crop window is recomputed with the dataset's own
    :func:`~src.beamlets.cropping.extract_beamlet_roi`, and (``verify_ct``) its CT
    crop must equal the saved ``{stem}_ct.npy`` bit for bit.

    Raises:
        RuntimeError: If a rebuilt CT crop differs from the saved one.
    """
    out_dir = Path(out_dir)
    todo = [s for s in src.stems if overwrite or not (out_dir / f"{s}_ds_pba.npy").exists()]
    if not todo:
        return {"computed": 0, "field_seconds": 0.0, "spot_seconds": 0.0}
    field = engine.prepare_field(geom.ct, geom.iso_mc, geom.mc_gantry)
    h, w, d = roi_size
    spot_seconds = 0.0
    for stem in todo:
        sim = json.loads((src.path / f"{stem}_sim_res.json").read_text())
        spot = sim["spot_position"]
        crop, _, crp, oob = extract_beamlet_roi(geom.ct, *bdl.distances, spot, geom.iso_ext, roi_size,
                                                ct_array=geom.ct_array)
        if oob:
            raise RuntimeError(f"{src.name}/{stem}: crop out of bounds, but it was saved in-bounds")
        if verify_ct and not np.array_equal(crop, np.load(src.path / f"{stem}_ct.npy")):
            raise RuntimeError(f"{src.name}/{stem}: rebuilt CT crop differs from the saved one")
        iz, iy, _ = crp
        window = (slice(iz - h // 2, iz - h // 2 + h), slice(iy - w // 2, iy - w // 2 + w), slice(0, d))
        result = engine.dose_in_window(field, src.energy, [spot], window)
        np.save(out_dir / f"{stem}_ds_pba.npy", result.dose)
        spot_seconds += result.seconds
    return {"computed": len(todo), "field_seconds": field.seconds, "spot_seconds": spot_seconds,
            "pba_model_energy": result.model_energy, "pba_interpolated_from": result.interpolated_from}


def load_triplet(out_dir: Path, stem: str) -> Dict[str, np.ndarray]:
    """``{"CT", "MC", "PBA", "ADoTA"}`` crops ``(H, W, D) = (z, y, x)`` of one mirrored beamlet."""
    out_dir = Path(out_dir)
    pred = np.squeeze(np.load(out_dir / f"{stem}_ds_pred.npy")).astype(np.float32)   # (D, H, W)
    return {"CT": np.load(out_dir / f"{stem}_ct.npy"),
            "MC": np.load(out_dir / f"{stem}_ds.npy").astype(np.float32),
            "PBA": np.load(out_dir / f"{stem}_ds_pba.npy").astype(np.float32),
            "ADoTA": np.clip(np.moveaxis(pred, 0, -1), 0.0, None)}


def score_beamlet(task) -> dict:
    """Worker: gamma and dose agreement of PBA and ADoTA against MC for one beamlet."""
    out_dir, stem, criteria, block = task
    logging.getLogger("pymedphys").setLevel(logging.WARNING)
    t0 = perf_counter()
    doses = load_triplet(out_dir, stem)
    mc = doses["MC"]
    max_dist = max(c.dist for c in criteria)
    min_cut = min(c.cutoff for c in criteria) / 100.0
    row = {"block": block, "stem": stem, "grid_ix": int(stem[1:3]), "grid_iy": int(stem[4:6]),
           "mc_max": float(mc.max())}
    for label in ("PBA", "ADoTA"):
        test = doses[label]
        tag = label.lower()
        box = dose_bbox([mc, test], low_frac=0.5 * min_cut, margin=int(np.ceil(max_dist)) + 2)
        for key, value in gamma_pass_rates(mc[box], test[box], criteria).items():
            row[f"{tag}_{key}"] = value
        for key, value in dose_agreement(mc, test).items():
            row[f"{tag}_{key}"] = value
    row["score_seconds"] = perf_counter() - t0
    return row


def rank_beamlets(rows: Sequence[dict], engine: str, criteria: Sequence[GammaCriterion], n: int = 10
                  ) -> Tuple[List[dict], List[dict]]:
    """``(best, worst)`` ``n`` beamlets of one engine by the first criterion.

    Ties (many beamlets pass 3%/3mm fully) are broken by the next criteria, then by
    the mean absolute dose difference, so the order is deterministic and meaningful.
    """
    tag = engine.lower()
    keys = [f"{tag}_{c.key}" for c in criteria]

    def order(r):
        return tuple(-float(r[k]) for k in keys) + (float(r[f"{tag}_mean_abs_diff_pct"]),)

    usable = [r for r in rows if np.isfinite(float(r[keys[0]]))]
    ranked = sorted(usable, key=order)
    return ranked[:n], ranked[::-1][:n]
