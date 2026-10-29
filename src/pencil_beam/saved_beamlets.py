"""Pencil beam on beamlet directories that were already simulated, scored with their saved MC and ADoTA.

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

The saved MC (``{id}_ds.npy``) and ADoTA prediction (``{id}_ds_pred.npy``) are used
as they are. Files are only ever *added* to a source directory, and each is created
exclusively (:func:`save_new_file`): **no existing file there is modified or
replaced**, by a rerun or by a crash. What may be added:

* ``{id}_ds_pba.npy`` and one ``pba_provenance.json`` per directory, recording the
  model fingerprint (:func:`~src.pencil_beam.beam_model.model_fingerprint`), the
  kernel build and the code commit. A directory whose provenance names a different
  model is refused, so pencil-beam doses of two models never mix in one directory.
* For directories without ADoTA predictions only: the missing ``{id}_ds_pred.npy``
  and an ``adota_provenance.json`` (:func:`write_missing_predictions`).

Metrics and figures go to a separate output root.

**Engines.** What is scored against the MC is a list of :class:`Engine`: a label,
a kind (``pencil_beam`` or ``adota``) and the file it reads or adds next to
``{id}_ds.npy``, for example ``_ds_pba.npy`` or ``_ds_pred_<model>.npy``. Several
ADoTA models therefore sit side by side in one directory, each with its own
provenance file (:attr:`Engine.provenance_file`). :func:`engines_from_config`
reads an ``engines:`` list, or the older ``pencil_beam:`` + ``adota:`` blocks as
the two engines ``PBA`` and ``ADoTA``.
"""
from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from src.beamlets.cropping import extract_beamlet_roi
from src.pencil_beam.metrics import GammaCriterion, dose_agreement, dose_bbox, gamma_pass_rates

logger = logging.getLogger(__name__)

__all__ = [
    "ENGINE_KINDS",
    "Engine",
    "INPUT_SUFFIXES",
    "engine_tag",
    "engines_from_config",
    "engines_to_score",
    "load_doses",
    "SourceDir",
    "compute_field_pba",
    "load_triplet",
    "rank_beamlets",
    "score_beamlet",
    "PBA_PROVENANCE",
    "ADOTA_PROVENANCE",
    "check_or_write_provenance",
    "field_windows",
    "save_new_file",
    "save_new_npy",
    "survey_dirs",
    "write_missing_predictions",
]

INPUT_SUFFIXES = ("_ct.npy", "_ds.npy", "_flux.npy", "_sim_res.json")    # required: written by the generator
PREDICTION_SUFFIX = "_ds_pred.npy"                                      # ADoTA; may be added if missing
PBA_PROVENANCE = "pba_provenance.json"
ADOTA_PROVENANCE = "adota_provenance.json"
_STEM = re.compile(r"^(a\d{2}_\d{2})_sim_res\.json$")
ENGINE_KINDS = ("pencil_beam", "adota")
_ENGINE_SUFFIX = re.compile(r"^_ds_[A-Za-z0-9_]+\.npy$")
_DEFAULT_SUFFIX = {"pencil_beam": "_ds_pba.npy", "adota": PREDICTION_SUFFIX}


def engine_tag(name: str) -> str:
    """The column prefix of an engine's scores: its name, lower case, ``[a-z0-9_]``."""
    return re.sub(r"[^0-9a-z]+", "_", name.lower()).strip("_")


@dataclass(frozen=True)
class Engine:
    """One dose engine scored against the saved MC.

    ``suffix`` names the file the engine reads or adds next to ``{id}_ds.npy``;
    ``settings`` holds what produces it (for ``pencil_beam``: ``dosecuda_kernels``,
    ``model_dir``, ``gpu_id``; for ``adota``: ``model_dir``, ``fname``,
    ``device_index``, ``batch_size``).
    """

    name: str
    kind: str
    suffix: str
    settings: Mapping[str, Any] = field(default_factory=dict, compare=False)

    @property
    def tag(self) -> str:
        return engine_tag(self.name)

    @property
    def provenance_file(self) -> str:
        """``pba_provenance.json`` / ``adota_provenance.json`` for the default files,
        else the kind's name with the file's suffix, one per engine output."""
        base = PBA_PROVENANCE if self.kind == "pencil_beam" else ADOTA_PROVENANCE
        if self.suffix == _DEFAULT_SUFFIX[self.kind]:
            return base
        return base.replace(".json", f"{self.suffix[:-len('.npy')]}.json")

    def load(self, path: Path) -> np.ndarray:
        """The engine's dose as ``(H, W, D) = (z, y, x)``, the layout of ``_ds.npy``.

        ADoTA writes ``(D, H, W)`` (possibly with a leading channel axis) and may
        predict small negatives, which are clipped to zero.
        """
        dose = np.load(path).astype(np.float32)
        if self.kind == "adota":
            dose = np.clip(np.moveaxis(np.squeeze(dose), 0, -1), 0.0, None)
        return dose


def engines_from_config(cfg: Mapping[str, Any]) -> List[Engine]:
    """The engines of a config: its ``engines:`` list, or the older ``pencil_beam:``
    + ``adota:`` blocks as ``PBA`` (``_ds_pba.npy``) and ``ADoTA`` (``_ds_pred.npy``).

    Raises:
        ValueError: On an unknown kind, a malformed suffix, the MC's own file, or
            two engines sharing a name, a column prefix or a file.
    """
    if "engines" in cfg:
        engines = []
        for item in cfg["engines"]:
            item = dict(item)
            name, kind = str(item.pop("name")), str(item.pop("kind"))
            if kind not in ENGINE_KINDS:
                raise ValueError(f"engine {name!r}: kind {kind!r} is not one of {ENGINE_KINDS}")
            suffix = str(item.pop("suffix", _DEFAULT_SUFFIX[kind]))
            engines.append(Engine(name=name, kind=kind, suffix=suffix, settings=item))
    else:
        engines = [Engine("PBA", "pencil_beam", "_ds_pba.npy", dict(cfg.get("pencil_beam") or {})),
                   Engine("ADoTA", "adota", PREDICTION_SUFFIX, dict(cfg.get("adota") or {}))]
    for attr in ("name", "tag", "suffix"):
        values = [getattr(e, attr) for e in engines]
        if len(set(values)) != len(values):
            raise ValueError(f"two engines share a {attr}: {values}")
    for e in engines:
        if not _ENGINE_SUFFIX.match(e.suffix) or e.suffix == "_ds.npy":
            raise ValueError(f"engine {e.name!r}: suffix {e.suffix!r} must look like _ds_<name>.npy")
        if e.tag in ("mc", "ct", ""):
            raise ValueError(f"engine name {e.name!r} is reserved or empty")
    return engines


def engines_to_score(rows: Sequence[Mapping[str, Any]], engines: Sequence[Engine], keys: Sequence[str]
                     ) -> List[Engine]:
    """The engines whose scores ``rows`` (one block's cached metrics) do not hold for
    every beamlet and every key; all of them when ``rows`` is empty."""
    if not rows:
        return list(engines)
    return [e for e in engines if any(f"{e.tag}_{k}" not in r for r in rows for k in keys)]


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
    n_missing_predictions: int = 0

    @property
    def name(self) -> str:
        return self.path.name


def survey_dirs(paths: Sequence[Path], grid_n: int = 18, min_beamlets: int = 200
                ) -> Tuple[List[SourceDir], List[Tuple[Path, str]]]:
    """Split candidate directories into full-lattice blocks and excluded ones (with a reason).

    A directory qualifies when its grid indices span the ``grid_n x grid_n`` lattice,
    it holds at least ``min_beamlets`` beamlets (the 18 x 18 sweeps lose a few to the
    QA gates, the smoke directories hold four or five), and every beamlet has the
    generator's four files. Missing ADoTA predictions are counted, not excluded.
    """
    selected, excluded = [], []
    for path in sorted(Path(p) for p in paths):
        stems = sorted(m.group(1) for f in path.iterdir() if (m := _STEM.match(f.name)))
        if not stems:
            excluded.append((path, "no beamlets"))
            continue
        missing = [f"{s}{x}" for s in stems for x in INPUT_SUFFIXES if not (path / f"{s}{x}").exists()]
        if missing:
            excluded.append((path, f"{len(missing)} input files missing (e.g. {missing[0]})"))
            continue
        n_pred = sum(not (path / f"{s}{PREDICTION_SUFFIX}").exists() for s in stems)
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
                energy=float(first["initial_energy"]), gantry=float(first["gantry_angle"]),
                n_missing_predictions=n_pred))
    return selected, excluded


def save_new_file(path: Path, write) -> bool:
    """Create ``path`` with ``write(fh)`` only if nothing exists there; ``True`` if it was written.

    Writes a hidden temporary file in the same directory and hard-links it to the
    target, which fails if the target exists. An existing file is never opened for
    writing, and an interrupted run leaves no half-written target.
    """
    path = Path(path)
    if path.exists():
        return False
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        with open(tmp, "xb") as fh:
            write(fh)
        try:
            os.link(tmp, path)
        except FileExistsError:
            return False
        return True
    finally:
        tmp.unlink(missing_ok=True)


def save_new_npy(path: Path, array: np.ndarray) -> bool:
    """:func:`save_new_file` for one array (``np.save``)."""
    return save_new_file(path, lambda fh: np.save(fh, array))


def check_or_write_provenance(path: Path, record: dict, identity_keys: Sequence[str]) -> dict:
    """Write ``record`` to ``path`` if absent; if present, require the same identity.

    Returns the record in force. The identity (for example the model fingerprint)
    decides whether new files may join a directory that already has some.

    Raises:
        RuntimeError: If an existing provenance names a different identity.
    """
    path = Path(path)
    payload = (json.dumps(record, indent=2, default=str) + "\n").encode()
    if save_new_file(path, lambda fh: fh.write(payload)):
        return record
    existing = json.loads(path.read_text())
    clash = {k: (existing.get(k), record.get(k)) for k in identity_keys if existing.get(k) != record.get(k)}
    if clash:
        raise RuntimeError(f"{path} was written by a different setup {clash}; "
                           "refusing to mix results in one directory")
    return existing


def field_windows(geom, bdl, roi_size, src: SourceDir, verify_ct: bool = True) -> Dict[str, tuple]:
    """``{stem: (spot, window)}`` for every beamlet of a block, verified before anything is written.

    Each crop window is recomputed with the dataset's own
    :func:`~src.beamlets.cropping.extract_beamlet_roi` on the rebuilt field, and
    (``verify_ct``) every CT crop must equal the saved ``{stem}_ct.npy`` bit for bit.
    A wrong generation config, CT, or code version therefore fails here, loudly.

    Raises:
        RuntimeError: Naming the first beamlets whose rebuilt crop differs.
    """
    h, w, d = roi_size
    windows, bad = {}, []
    for stem in src.stems:
        sim = json.loads((src.path / f"{stem}_sim_res.json").read_text())
        spot = sim["spot_position"]
        crop, _, crp, oob = extract_beamlet_roi(geom.ct, *bdl.distances, spot, geom.iso_ext, roi_size,
                                                ct_array=geom.ct_array)
        if oob or (verify_ct and not np.array_equal(crop, np.load(src.path / f"{stem}_ct.npy"))):
            bad.append(stem)
            continue
        iz, iy, _ = crp
        windows[stem] = (spot, (slice(iz - h // 2, iz - h // 2 + h), slice(iy - w // 2, iy - w // 2 + w),
                                slice(0, d)))
    if bad:
        raise RuntimeError(f"{src.name}: {len(bad)}/{len(src.stems)} rebuilt CT crops differ from the saved ones "
                           f"(e.g. {bad[:3]}); check generation_config, the CT source and the code version")
    return windows


def compute_field_pba(engine, geom, bdl, roi_size, src: SourceDir, provenance: dict,
                      suffix: str = "_ds_pba.npy", provenance_file: str = PBA_PROVENANCE) -> dict:
    """Pencil-beam dose of every beamlet of one block, added to the source as ``{stem}{suffix}``.

    ``geom`` is the rebuilt :class:`~src.mc_generation.robustness.FieldGeometry`. All
    crops are verified (:func:`field_windows`) and the directory's
    ``provenance_file`` is checked or written before the first dose is saved.
    Beamlets that already have one are skipped (never replaced).
    """
    windows = field_windows(geom, bdl, roi_size, src)
    check_or_write_provenance(src.path / provenance_file, provenance, ("model_fingerprint",))
    todo = [s for s in src.stems if not (src.path / f"{s}{suffix}").exists()]
    if not todo:
        return {"computed": 0, "field_seconds": 0.0, "spot_seconds": 0.0}
    field = engine.prepare_field(geom.ct, geom.iso_mc, geom.mc_gantry)
    spot_seconds = 0.0
    for stem in todo:
        spot, window = windows[stem]
        result = engine.dose_in_window(field, src.energy, [spot], window)
        save_new_npy(src.path / f"{stem}{suffix}", result.dose)
        spot_seconds += result.seconds
    return {"computed": len(todo), "field_seconds": field.seconds, "spot_seconds": spot_seconds,
            "pba_model_energy": result.model_energy, "pba_interpolated_from": result.interpolated_from}


def write_missing_predictions(src: SourceDir, model, device, provenance: dict, batch_size: int = 56,
                              tmp_root: Optional[Path] = None, suffix: str = PREDICTION_SUFFIX,
                              provenance_file: str = ADOTA_PROVENANCE) -> int:
    """Add ADoTA predictions as ``{stem}{suffix}`` for the beamlets of a block that
    have none; return how many.

    Inference runs in a temporary directory of symlinks (``run_inference`` writes
    ``{stem}_ds_pred.npy`` next to its inputs and would overwrite), and each
    prediction is then added to the source exclusively under ``suffix``, with a
    ``provenance_file`` that must match on reruns. Existing files are never touched.
    """
    from src.beamlets.inference import InferenceConfig, run_inference

    missing = [s for s in src.stems if not (src.path / f"{s}{suffix}").exists()]
    if not missing:
        return 0
    check_or_write_provenance(src.path / provenance_file, provenance, ("checkpoint_sha256",))
    with tempfile.TemporaryDirectory(dir=tmp_root) as tmp:
        tmp = Path(tmp)
        for stem in missing:
            for name in INPUT_SUFFIXES:
                (tmp / f"{stem}{name}").symlink_to(src.path / f"{stem}{name}")
        run_inference(tmp, model, device, InferenceConfig(batch_size=batch_size, grid_factor=1))
        return sum(save_new_npy(src.path / f"{stem}{suffix}", np.load(tmp / f"{stem}{PREDICTION_SUFFIX}"))
                   for stem in missing)


def load_doses(block_dir: Path, stem: str, engines: Sequence[Engine]) -> Dict[str, np.ndarray]:
    """``{"CT", "MC", <engine name>...}`` crops ``(H, W, D) = (z, y, x)`` of one saved beamlet (read only)."""
    block_dir = Path(block_dir)
    out = {"CT": np.load(block_dir / f"{stem}_ct.npy"),
           "MC": np.load(block_dir / f"{stem}_ds.npy").astype(np.float32)}
    for e in engines:
        out[e.name] = e.load(block_dir / f"{stem}{e.suffix}")
    return out


def load_triplet(block_dir: Path, stem: str) -> Dict[str, np.ndarray]:
    """``{"CT", "MC", "PBA", "ADoTA"}``: :func:`load_doses` with the two default engines."""
    return load_doses(block_dir, stem, engines_from_config({}))


def score_beamlet(task) -> dict:
    """Worker: gamma and dose agreement against MC of each engine for one beamlet.

    ``task`` is ``(block_dir, stem, criteria, block[, engines])``; without
    ``engines``, the two default engines (``PBA``, ``ADoTA``).
    """
    block_dir, stem, criteria, block = task[:4]
    engines = task[4] if len(task) > 4 else engines_from_config({})
    logging.getLogger("pymedphys").setLevel(logging.WARNING)
    t0 = perf_counter()
    doses = load_doses(block_dir, stem, engines)
    mc = doses["MC"]
    max_dist = max(c.dist for c in criteria)
    min_cut = min(c.cutoff for c in criteria) / 100.0
    row = {"block": block, "stem": stem, "grid_ix": int(stem[1:3]), "grid_iy": int(stem[4:6]),
           "mc_max": float(mc.max())}
    for e in engines:
        test = doses[e.name]
        tag = e.tag
        box = dose_bbox([mc, test], low_frac=0.5 * min_cut, margin=int(np.ceil(max_dist)) + 2)
        for key, value in gamma_pass_rates(mc[box], test[box], criteria).items():
            row[f"{tag}_{key}"] = value
        for key, value in dose_agreement(mc, test).items():
            row[f"{tag}_{key}"] = value
    row["score_seconds"] = perf_counter() - t0
    return row


def rank_beamlets(rows: Sequence[dict], engine: str, criteria: Sequence[GammaCriterion], n: int = 10
                  ) -> Tuple[List[dict], List[dict]]:
    """``(best, worst)`` ``n`` beamlets of one engine (its name) by the first criterion.

    Ties (many beamlets pass 3%/3mm fully) are broken by the next criteria, then by
    the mean absolute dose difference, so the order is deterministic and meaningful.
    """
    tag = engine_tag(engine)
    keys = [f"{tag}_{c.key}" for c in criteria]

    def order(r):
        return tuple(-float(r[k]) for k in keys) + (float(r[f"{tag}_mean_abs_diff_pct"]),)

    usable = [r for r in rows if np.isfinite(float(r[keys[0]]))]
    ranked = sorted(usable, key=order)
    return ranked[:n], ranked[::-1][:n]
