"""The scorer interface, and the input-only difficulty scorer over HDF5 records.

The loop calls ``scorer.score(pool)`` once per cycle on the records still in the
pool and does not assume the result is stable between cycles: a future scorer
will depend on the current model weights and must be recomputed each cycle, and
the loop is built for that from the start. The frozen difficulty score does not
change between cycles, and is still recomputed, on purpose.

**Input only, enforced.** :func:`input_only_features` reads the CT, the flux and
the beam energy of a record and nothing else; the Bragg peak is located by the
analytic surrogate of :mod:`src.acquisition.surrogate`, never by the stored dose
or by anything derived from it. ``tests/retrospective/test_scoring.py`` fails if
that path ever touches the dose array.
"""
from __future__ import annotations

import logging
from multiprocessing import get_context
from pathlib import Path
from typing import Dict, List, Optional, Protocol, Union, runtime_checkable

import h5py
import numpy as np
import pandas as pd

from src.acquisition.features import FeatureConfig
from src.acquisition.reference import StoredRecord, record_features
from src.acquisition.scorer import DEPLOYED_SCORER, FULL, DifficultyScorer
from src.adota.config import DEFAULT_SCALE, denormalize_energy
from src.utils.scallers import inverse_minmax

logger = logging.getLogger(__name__)

INPUT_DATASETS = ("ct", "flux")
"""The only datasets of a record the scoring path may open."""


@runtime_checkable
class PoolScorer(Protocol):
    """What the loop needs from a scorer: a name for the manifest and one call
    that returns the pool with a ``score`` column (higher is harder, NaN when a
    record cannot be scored) and any diagnostic columns it wants recorded."""

    name: str

    def score(self, pool: pd.DataFrame) -> pd.DataFrame: ...


# ── Input-only features of one stored record ────────────────────────────────


def read_stored_inputs(group, sample_id: str,
                       scale: Dict[str, float] = DEFAULT_SCALE) -> StoredRecord:
    """The model inputs of one record: CT in HU, the normalised flux, the energy.

    ``dose`` is left ``None``. :func:`src.acquisition.reference.read_stored_record`
    is the study's loader and reads the ground truth too; this one exists so the
    scoring path cannot.
    """
    ct = np.asarray(group["ct"][:], dtype=np.float32).transpose(2, 0, 1)
    flux = np.asarray(group["flux"][:], dtype=np.float32).transpose(2, 0, 1)
    span = float(flux.max() - flux.min())
    if span > 0:
        flux = (flux - flux.min()) / span
    return StoredRecord(
        sample_id=sample_id,
        ct_hu=inverse_minmax(ct, scale["min_ct"], scale["max_ct"]),
        flux=flux, dose=None,
        energy_mev=float(denormalize_energy(float(group.attrs["initial_energy"]), scale)))


def input_only_features(group, sample_id: str, scale: Dict[str, float] = DEFAULT_SCALE,
                        config: FeatureConfig = FeatureConfig()) -> Optional[Dict[str, float]]:
    """The thirty metrics of a record from its inputs and the analytic dose, or
    ``None`` for the records the reference study skipped (no flux, energy above
    250 MeV)."""
    record = read_stored_inputs(group, sample_id, scale)
    return record_features(record, mode="analytic", config=config)


# ── The difficulty scorer over the HDF5 pool ────────────────────────────────


def _score_chunk(args) -> List[Dict]:
    """Worker: features and score for a chunk of record ids. Opens the file
    itself, so it is safe under ``spawn`` and ``fork`` alike."""
    dataset_path, ids, scorer_path, variant, arm, scale = args
    scorer = DifficultyScorer.load(scorer_path, variant=variant, arm=arm)
    rows = []
    with h5py.File(dataset_path, "r") as handle:
        for sample_id in ids:
            row = {"sample_id": sample_id, "score": np.nan, "peak_inside_crop": False,
                   "scored": False}
            try:
                features = input_only_features(handle[sample_id], sample_id, scale)
            except Exception as exc:  # one broken record must not end the cycle
                row["error"] = repr(exc)
                rows.append(row)
                continue
            if features is not None:
                row.update(score=float(scorer.score(features)),
                           peak_inside_crop=bool(features["peak_inside_crop"]),
                           scored=True, bp_range_max_mm=float(features["bp_range_max_mm"]),
                           wepl_mean=float(features["wepl_mean"]))
            rows.append(row)
    return rows


class DifficultyPoolScorer:
    """The deployed input-only difficulty score (EXP-0006) applied to HDF5 records.

    Args:
        dataset_path: The HDF5 file the pool ids index.
        scorer_path: The frozen scorer JSON; the vendored deployed one by default.
        variant: ``"full (ridge)"`` (deployed) or ``"sparse (Lasso)"``.
        arm: For an arms-keyed scorer file only; ``None`` for the vendored one.
        n_workers: Processes for the feature extraction (about a second per
            record per process).
        chunk_size: Records per worker task.
    """

    name = "difficulty"

    def __init__(self, dataset_path: Union[str, Path], *,
                 scorer_path: Union[str, Path] = DEPLOYED_SCORER, variant: str = FULL,
                 arm: Optional[str] = None, n_workers: int = 8, chunk_size: int = 128,
                 scale: Dict[str, float] = DEFAULT_SCALE):
        self.dataset_path = str(dataset_path)
        self.scorer_path = str(scorer_path)
        self.variant = variant
        self.arm = arm
        self.n_workers = max(1, int(n_workers))
        self.chunk_size = max(1, int(chunk_size))
        self.scale = dict(scale)
        # Fail at construction, not in a worker, if the scorer file is wrong.
        self.scorer = DifficultyScorer.load(self.scorer_path, variant=variant, arm=arm)

    def describe(self) -> Dict:
        return {"name": self.name, "scorer_path": self.scorer_path, "variant": self.variant,
                "arm": self.arm, "source": self.scorer.source,
                "n_metrics": len(self.scorer.metrics)}

    def score(self, pool: pd.DataFrame) -> pd.DataFrame:
        ids = pool["sample_id"].astype(str).tolist()
        # Small pools still spread over every worker: at least four tasks per worker.
        chunk = max(1, min(self.chunk_size, int(np.ceil(len(ids) / (4 * self.n_workers)))))
        chunks = [ids[i:i + chunk] for i in range(0, len(ids), chunk)]
        tasks = [(self.dataset_path, chunk, self.scorer_path, self.variant, self.arm, self.scale)
                 for chunk in chunks]
        logger.info("scoring %d pool records with %s in %d chunks on %d workers",
                    len(ids), self.name, len(chunks), self.n_workers)
        if self.n_workers == 1 or len(chunks) == 1:
            rows = [row for task in tasks for row in _score_chunk(task)]
        else:
            with get_context("spawn").Pool(self.n_workers) as workers:
                rows = [row for part in workers.imap_unordered(_score_chunk, tasks)
                        for row in part]
        scored = pd.DataFrame(rows)
        out = pool.merge(scored, on="sample_id", how="left")
        n_ok = int(out["scored"].fillna(False).sum())
        logger.info("scored %d of %d records (%d unscorable); score median %.3f, "
                    "peak inside crop for %.1f%%", n_ok, len(out), len(out) - n_ok,
                    float(out["score"].median()) if n_ok else float("nan"),
                    100.0 * float(out["peak_inside_crop"].fillna(False).mean()))
        return out


SCORERS = {"difficulty": DifficultyPoolScorer}
"""Registry of scorers by config name. A model-uncertainty or hybrid scorer is a
new entry here that satisfies :class:`PoolScorer`; the loop does not change."""


def build_scorer(name: str, dataset_path: Union[str, Path], **kwargs) -> PoolScorer:
    if name not in SCORERS:
        raise ValueError(f"unknown scorer {name!r}; expected one of {sorted(SCORERS)}")
    return SCORERS[name](dataset_path, **kwargs)


def score_distribution(pool: pd.DataFrame) -> Dict[str, float]:
    """The numbers the smoke test prints for the first scored pool."""
    scores = pool["score"].to_numpy(dtype=float)
    finite = scores[np.isfinite(scores)]
    if finite.size == 0:
        return {"n": int(len(pool)), "n_scored": 0}
    quantiles = np.percentile(finite, [0, 5, 25, 50, 75, 95, 100])
    return {"n": int(len(pool)), "n_scored": int(finite.size),
            "mean": float(finite.mean()), "std": float(finite.std()),
            **{f"p{p:02d}": float(v) for p, v in zip((0, 5, 25, 50, 75, 95, 100), quantiles)},
            "peak_inside_crop_fraction": float(pool["peak_inside_crop"].fillna(False).mean())
            if "peak_inside_crop" in pool else float("nan")}
