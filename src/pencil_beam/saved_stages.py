"""The prediction and scoring stages on saved beamlet directories, shared by the scripts that run them.

:func:`add_adota_predictions` adds a model's doses next to the saved MC, and
:func:`score_blocks` scores every engine against that MC, keeping what a block's
``metrics.json`` already holds. An ADoTA engine names its model in one of two ways:

* ``model_dir`` + ``fname``: a deployed model with a ``hyperparams.json``.
* ``checkpoint`` + ``model_config``: a training run's snapshot and its resolved
  ``config.yaml``, which also supplies the normalisation scale.
"""
from __future__ import annotations

import hashlib
import json
import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from src.pencil_beam.pipeline import write_rows_csv
from src.pencil_beam.saved_beamlets import Engine, engines_to_score, score_beamlet, write_missing_predictions

logger = logging.getLogger(__name__)


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_adota_engine(settings: Mapping[str, Any], device) -> Tuple[Any, Optional[Dict[str, float]], Dict[str, str]]:
    """``(model in eval mode, its scale or None for the default, provenance fields)``.

    Raises:
        ValueError: If ``settings`` names neither form of model.
    """
    if settings.get("checkpoint") and settings.get("model_config"):
        from src.active_learning.ensemble.members import training_config_from_yaml
        from src.training.checkpoints import CheckpointManager
        from src.training.factory import build_adota_model

        checkpoint, model_config = Path(settings["checkpoint"]), Path(settings["model_config"])
        config = training_config_from_yaml(model_config, "")
        model = build_adota_model(config, device)
        CheckpointManager.load_weights_only(checkpoint, model=model, device=device)
        model.eval()
        return model, dict(config.scale), {"checkpoint": str(checkpoint), "checkpoint_sha256": _sha256(checkpoint),
                                           "model_config": str(model_config),
                                           "model_config_sha256": _sha256(model_config)}
    if settings.get("model_dir"):
        from src.adota.utils import load_model

        model_dir = Path(settings["model_dir"])
        checkpoint = model_dir / settings.get("fname", "best_model.pth")
        model = load_model(checkpoint, model_dir / "hyperparams.json", device)
        return model, None, {"checkpoint": str(checkpoint), "checkpoint_sha256": _sha256(checkpoint),
                             "hyperparams_sha256": _sha256(model_dir / "hyperparams.json")}
    raise ValueError("an ADoTA engine needs model_dir, or checkpoint and model_config")


def add_adota_predictions(blocks: Sequence, engines: Sequence[Engine], run_provenance: Mapping[str, Any]) -> None:
    """For each ADoTA engine, add its prediction only where a beamlet has none."""
    from src.evaluation.cli import resolve_device

    for e in (e for e in engines if e.kind == "adota"):
        todo = [b for b in blocks if any(not (b.path / f"{s}{e.suffix}").exists() for s in b.stems)]
        if not todo:
            logger.info("%s: every beamlet already has a %s; nothing to add", e.name, e.suffix)
            continue
        device = resolve_device(int(e.settings.get("device_index", 0)))
        try:
            model, scale, files = load_adota_engine(e.settings, device)
        except ValueError as err:
            raise ValueError(f"{len(todo)} blocks lack {e.suffix}; engine {e.name!r}: {err}") from err
        record = {"engine": e.name, "suffix": e.suffix, **files, **run_provenance}
        for b in todo:
            n = write_missing_predictions(b, model, device, record, int(e.settings.get("batch_size", 56)),
                                          suffix=e.suffix, provenance_file=e.provenance_file, scale=scale)
            logger.info("%s %s: added %d missing predictions (%s)", e.name, b.name, n, e.suffix)


def score_blocks(blocks: Sequence, engines: Sequence[Engine], out_root: Path, criteria: Sequence, overwrite: bool,
                 workers: int = 32) -> List[dict]:
    """Every engine against the saved MC, per block, cached in ``<out_root>/<block>/metrics.json``.

    Only the engines a block's cache does not hold are scored; a cache for another
    beamlet selection is discarded. Writes ``<out_root>/all_beamlets.csv``.

    Raises:
        ValueError: If an engine to score has no file for a beamlet.
    """
    keys = [c.key for c in criteria] + ["r80_delta_mm"]
    rows: List[dict] = []
    for b in blocks:
        path = Path(out_root) / b.name / "metrics.json"
        cached = [] if overwrite or not path.exists() else json.loads(path.read_text())
        if cached and {r["stem"] for r in cached} != set(b.stems):
            cached = []                                   # another beamlet selection: score again
        todo = engines_to_score(cached, engines, keys)
        if not todo:
            rows += cached
            continue
        for e in todo:
            missing = [s for s in b.stems if not (b.path / f"{s}{e.suffix}").exists()]
            if missing:
                stage = "pba" if e.kind == "pencil_beam" else "adota"
                raise ValueError(f"{b.name}: {len(missing)} beamlets have no {e.suffix} ({e.name}); "
                                 f"run the {stage} stage")
        tasks = [(str(b.path), s, tuple(criteria), b.name, tuple(todo)) for s in b.stems]
        with ProcessPoolExecutor(max_workers=workers) as ex:
            scored = {r["stem"]: r for r in ex.map(score_beamlet, tasks, chunksize=2)}
        meta = {"anatomy": b.anatomy, "patient_id": b.patient_id, "energy": b.energy, "gantry": b.gantry,
                "block_dir": str(b.path)}
        by_stem = {r["stem"]: r for r in cached}
        block_rows = [dict(by_stem.get(s, {}), **scored[s], **meta) for s in b.stems]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(block_rows, indent=1, default=float))
        head = criteria[0].key
        logger.info("score %s (%s): median %s %s", b.name, ", ".join(e.name for e in todo), head, ", ".join(
            f"{e.name} {np.nanmedian([r[f'{e.tag}_{head}'] for r in block_rows]):.1f}%" for e in engines))
        rows += block_rows
    write_rows_csv(rows, Path(out_root) / "all_beamlets.csv")
    return rows
